#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"
# The NCCL_SOCKET_IFNAME variable specifies which IP interface to use for nccl
# communication. More details can be found in
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=INFO
stage=5 # start from 0 if you need to start from data preparation
stop_stage=5
# The num of nodes or machines used for multi-machine training
# Default 1 for single machine/node
# NFS will be needed if you want run multi-machine training
num_nodes=1
# The rank of each node or machine, range from 0 to num_nodes -1
# The first node/machine sets node_rank 0, the second one sets node_rank 1
# the third one set node_rank 2, and so on. Default 0
node_rank=0

data=/node4/speech/data
data_url=www.openslr.org/resources/33

# modify this to your AISHELL-2 data path
trn_set=/ssd/wenjing.xia/AISHELL-2/AISHELL-2/iOS/data
dev_set=/node4/wenjing.xia/data/AISHELL-2/AISHELL-DEV-TEST-SET/iOS/dev
tst_set=/node4/wenjing.xia/data/AISHELL-2/AISHELL-DEV-TEST-SET/iOS/test

nj=16
feat_dir=raw_wav
dict=data/dict/lang_char.txt

train_set=aishell
candidate_set=aishell2
train_dir=train
candidate_dir=candidate
dev_dir=dev_combine
num_split=8 # number of dev splits to run on one GPU
# Optional train_config
# 1. conf/train_transformer.yaml: Standard transformer
# 2. conf/train_conformer.yaml: Standard conformer
# 3. conf/train_unified_conformer.yaml: Unified dynamic chunk causal conformer
# 4. conf/train_unified_transformer.yaml: Unified dynamic chunk transformer
train_config=conf/train_unified_conformer.yaml
cmvn=true
dir=exp_tune_all/conformer
checkpoint=

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=10
decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"

. tools/parse_options.sh || exit 1;

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    local/download_and_untar.sh ${data} ${data_url} data_aishell
    #local/download_and_untar.sh ${data} ${data_url} resource_aishell
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # Data preparation
    local/aishell_data_prep.sh ${data}/data_aishell/wav ${data}/data_aishell/transcript

    local/prepare_data.sh ${trn_set} data/local/${candidate_set} data/${candidate_set} || exit 1;
    local/prepare_data.sh ${dev_set} data/local/dev data/aishell2_dev || exit 1;
    local/prepare_data.sh ${tst_set} data/local/test data/aishell2_test || exit 1;

     tools/combine_data.sh data/${dev_dir} data/dev data/aishell2_dev || exit 1;
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # remove the space between the text labels for Mandarin dataset
    for x in ${train_set} ${dev_dir} test ${candidate_set} aishell2_test; do
        sed 's/\t/ /' data/${x}/text > data/${x}/text.org
        paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " data/${x}/text.org \
             | tr 'a-z' 'A-Z' | sed 's/\([A-Z]\) \([A-Z]\)/\1▁\2/g' | tr -d " ") \
            > data/${x}/text
        #rm data/${x}/text.org
    done
    # For wav feature, just copy the data. Fbank extraction is done in training
    mkdir -p $feat_dir
    for x in ${train_set} ${dev_dir} test ${candidate_set} aishell2_test; do
        cp -r data/$x $feat_dir
    done

    tools/compute_cmvn_stats.py --num_workers 16 --train_config $train_config \
        --in_scp data/${train_set}/wav.scp \
        --out_cmvn $feat_dir/$train_set/global_cmvn

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # Make train dict
    echo "Make a dictionary"
    mkdir -p $(dirname $dict)
    echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
    echo "<unk> 1" >> ${dict} # <unk> must be 1
    tools/text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -a -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    num_token=$(cat $dict | wc -l)
    echo "<sos/eos> $num_token" >> $dict # <eos>
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    nj=32
    # Prepare wenet requried data
    echo "Prepare data, prepare requried format"
    for x in ${dev_dir} test ${train_set} ${candidate_set} aishell2_test; do
        tools/format_data.sh --nj ${nj} \
            --feat-type wav --feat $feat_dir/$x/wav.scp \
            $feat_dir/$x ${dict} > $feat_dir/$x/format.data.tmp

        tools/remove_longshortdata.py \
            --min_input_len 0.5 \
            --max_input_len 20 \
            --max_output_len 400 \
            --max_output_input_ratio 10.0 \
            --data_file $feat_dir/$x/format.data.tmp \
            --output_data_file $feat_dir/$x/format.data
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    mkdir $feat_dir/$train_dir
    cp $feat_dir/$train_set/format.data $feat_dir/$train_dir/
    cp $feat_dir/$train_set/global_cmvn $feat_dir/$train_dir/
    # Training
    mkdir -p $dir
    INIT_FILE=$dir/ddp_init
    # You had better rm it manually before you start run.sh on first node.
    # rm -f $INIT_FILE # delete old one before starting
    init_method=file://$(readlink -f $INIT_FILE)
    echo "$0: init method is $init_method"
    # The number of gpus runing on each node/machine
    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
    # Use "nccl" if it works, otherwise use "gloo"
    dist_backend="nccl"
    # The total number of processes/gpus, so that the master knows
    # how many workers to wait for.
    # More details about ddp can be found in
    # https://pytorch.org/tutorials/intermediate/dist_tuto.html
    world_size=`expr $num_gpus \* $num_nodes`
    echo "total gpus is: $world_size"
    cmvn_opts=
    $cmvn && cp $feat_dir/${train_dir}/global_cmvn $dir
    $cmvn && cmvn_opts="--cmvn ${dir}/global_cmvn"

    # train.py will write $train_config to $dir/train.yaml with model input
    # and output dimension, train.yaml will be used for inference or model
    # export later
    for ((i = 0; i < $num_gpus; ++i)); do
    {
        gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
    # Rank of each gpu/process used for knowing whether it is
    # the master of a worker.
    rank=`expr $node_rank \* $num_gpus + $i`
    python wenet/bin/train.py --gpu $gpu_id \
            --config $train_config \
            --train_data $feat_dir/$train_dir/format.data \
            --cv_data $feat_dir/${dev_dir}/format.data \
            ${checkpoint:+--checkpoint $checkpoint} \
            --model_dir $dir \
            --ddp.init_method $init_method \
            --ddp.world_size $world_size \
            --ddp.rank $rank \
            --ddp.dist_backend $dist_backend \
            --num_workers 2 \
            $cmvn_opts \
            --pin_memory
    } &
    done
    wait
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
    for cutoff in 1.0 0.5 0.0 -1.0 -99999999.0; do
    if [ ${average_checkpoint} == true ]; then
        decode_checkpoint=$dir/avg_${average_num}.pt
        if [ $(echo "$cutoff < 1"|bc) = 1 ]; then
        echo "do model average and final checkpoint is $decode_checkpoint"
        python wenet/bin/average_model.py \
            --dst_model $decode_checkpoint \
            --src_path $dir  \
            --num ${average_num} \
            --val_best
        fi
    fi

    decoding_chunk_size=
    ctc_weight=0.5
    mode="attention_rescoring"
    for test_data in ${dev_dir} $candidate_set; do
        test_num=`wc -l $feat_dir/$test_data/format.data | awk '{print $1}'`
        num_utts=$[($test_num+$num_gpus*$num_split-1)/($num_gpus*$num_split)]
        split -l $num_utts --numeric-suffixes $feat_dir/$test_data/format.data $feat_dir/$test_data/format.data_
        gpu=0
        index=0
        for f in $feat_dir/$test_data/format.data_*; do
            if [ $index -ge $num_split ]; then
                gpu=$[$gpu+1]
                index=0
            fi
            {
            split=${f##*_}
            test_dir=$dir/pred_${test_data}_${split}
            mkdir -p $test_dir
            python wenet/bin/recognize.py --gpu $gpu \
                --mode $mode \
                --config $dir/train.yaml \
                --test_data $feat_dir/$test_data/format.data_${split} \
                --checkpoint $decode_checkpoint \
                --beam_size 10 \
                --batch_size 1 \
                --penalty 0.0 \
                --dict $dict \
                --ctc_weight $ctc_weight \
                --score_file $test_dir/score \
                --result_file $test_dir/text \
                ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
             } &
             index=$[$index+1]
        done
        wait
        test_dir=$dir/pred_${test_data}
        rm -r $test_dir
        mkdir $test_dir
        cat $dir/pred_${test_data}_*/score > $test_dir/score
        cat $dir/pred_${test_data}_*/text > $test_dir/text
    done
    sleep 10s

    candidate_dir=${candidate_dir}_cutoff${cutoff}
    mkdir -p $feat_dir/$candidate_dir/selected
    python wenet/filter/filter.py \
       --cutoff $cutoff \
       --test_data $dir/pred_${dev_dir}/text \
       --test_score $dir/pred_${dev_dir}/score \
       --candidate_data $dir/pred_$candidate_set/text \
       --candidate_score $dir/pred_${candidate_set}/score \
       --source_data $feat_dir/$candidate_set/wav.scp \
       --result_dir $feat_dir/$candidate_dir/selected || exit 1

       tools/format_data.sh --nj ${nj} \
           --feat-type wav --feat $feat_dir/$candidate_dir/selected/wav.scp \
           $feat_dir/$candidate_dir/selected ${dict} > $feat_dir/$candidate_dir/format.data.tmp || exit 1
       tools/remove_longshortdata.py \
           --min_input_len 0.5 \
           --max_input_len 20 \
           --max_output_len 400 \
           --max_output_input_ratio 10.0 \
           --data_file $feat_dir/$candidate_dir/format.data.tmp \
           --output_data_file $feat_dir/$candidate_dir/format.data || exit 1


    cp $feat_dir/$train_set/format.data $feat_dir/$train_dir/
    if [ -f $feat_dir/$candidate_dir/format.data ]; then
        cat $feat_dir/$candidate_dir/format.data >> $feat_dir/$train_dir/format.data
    fi
    dir=${dir}_cutoff${cutoff}
    mkdir -p $dir
    # Training
    INIT_FILE=$dir/ddp_init
    # You had better rm it manually before you start run.sh on first node.
    # rm -f $INIT_FILE # delete old one before starting
    init_method=file://$(readlink -f $INIT_FILE)
    echo "$0: init method is $init_method"
    # The number of gpus runing on each node/machine
    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
    # Use "nccl" if it works, otherwise use "gloo"
    dist_backend="nccl"
    # The total number of processes/gpus, so that the master knows
    # how many workers to wait for.
    # More details about ddp can be found in
    # https://pytorch.org/tutorials/intermediate/dist_tuto.html
    world_size=`expr $num_gpus \* $num_nodes`
    echo "total gpus is: $world_size"
    cmvn_opts=
    $cmvn && cp $feat_dir/${train_dir}/global_cmvn $dir
    $cmvn && cmvn_opts="--cmvn ${dir}/global_cmvn"
    cp $train_config $dir/train_tuning.yaml
    sed -i 's/max_epoch: 120/max_epoch: 40/g' $dir/train_tuning.yaml

    # train.py will write $train_config to $dir/train.yaml with model input
    # and output dimension, train.yaml will be used for inference or model
    # export later
    for ((i = 0; i < $num_gpus; ++i)); do
    {
        gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
    # Rank of each gpu/process used for knowing whether it is
    # the master of a worker.
    rank=`expr $node_rank \* $num_gpus + $i`
    python wenet/bin/train.py --gpu $gpu_id \
            --config $dir/train_tuning.yaml \
            --train_data $feat_dir/$train_dir/format.data \
            --cv_data $feat_dir/${dev_dir}/format.data \
            ${decode_checkpoint:+--checkpoint $decode_checkpoint} \
            --model_dir $dir \
            --ddp.init_method $init_method \
            --ddp.world_size $world_size \
            --ddp.rank $rank \
            --ddp.dist_backend $dist_backend \
            --num_workers 2 \
            $cmvn_opts \
            --pin_memory \
            --tuning
    } &
    done
    wait
    done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    # Test model, please specify the model you want to test by --checkpoint
    dir=exp/conformer
    if [ ${average_checkpoint} == true ]; then
        decode_checkpoint=$dir/avg_${average_num}.pt
        echo "do model average and final checkpoint is $decode_checkpoint"
        if [ ! -f $decode_checkpoint ]; then
            python wenet/bin/average_model.py \
                --dst_model $decode_checkpoint \
                --src_path $dir  \
                --num ${average_num} \
                --val_best
        fi
    fi
    # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
    # -1 for full chunk
    decoding_chunk_size=
    ctc_weight=0.5
    mode="attention_rescoring"
    idx=0
    for testset in test aishell2_test; do
        for decoding_chunk_size in -1 16; do
        {
            test_dir=$dir/test_${mode}
            mkdir -p $test_dir
            gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$idx+1])
            python wenet/bin/recognize.py --gpu $gpu \
                --mode $mode \
                --config $dir/train.yaml \
                --test_data $feat_dir/test/format.data \
                --checkpoint $decode_checkpoint \
                --beam_size 10 \
                --batch_size 1 \
                --penalty 0.0 \
                --dict $dict \
                --ctc_weight $ctc_weight \
                --result_file $test_dir/text \
                ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
             python tools/compute-wer.py --char=1 --v=1 \
                $feat_dir/test/text $test_dir/text > $test_dir/wer
        } &
        ((idx+=1))
        done
    done
    wait

fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    # Export the best model you want
    python wenet/bin/export_jit.py \
        --config $dir/train.yaml \
        --checkpoint $dir/avg_${average_num}.pt \
        --output_file $dir/final.zip \
        --output_quant_file $dir/final_quant.zip
fi

