# Copyright (c) 2021 Mobvoi Inc. (authors: Wenjing Xia)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='filter candidate data')
    parser.add_argument('--cutoff', type=float, required=True, help='cut off scale')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--test_score', required=True, help='test score file')
    parser.add_argument('--candidate_data', required=True, help='candidate data file')
    parser.add_argument('--candidate_score', required=True, help='candidate score file')
    parser.add_argument('--source_data', required=True, help='source data file')
    parser.add_argument('--result_dir', required=True, help='result dir')
    args = parser.parse_args()
    return args

class Filter:
    def __init__(self, cutoff, predict_path, score_path):
        self.cutoff = cutoff
        self._init_compute(predict_path, score_path)

    def _init_compute(self, predict_path, score_path):
        predict_length = []
        score = []
        with open(predict_path, 'r') as preader, open(score_path, 'r') as sreader:
            for pline, sline in zip(preader, sreader):
                pline_split = pline.strip().split()
                sline_split = sline.strip().split()
                assert(len(pline_split) <= 2 and len(sline_split) <= 2 and pline_split[0] == sline_split[0])
                predict_length.append(len(pline_split[1]) if len(pline_split) == 2 else 0.000001)
                score.append(float(sline_split[1]))
        predict_length = np.asarray(predict_length)
        score = np.asarray(score)
        self.weight = np.sum(score * (predict_length - np.mean(predict_length))) / (np.sum(predict_length**2) - (1/predict_length.size) * (np.sum(predict_length))**2)
        self.bias = (1 / predict_length.size) * np.sum(score - self.weight * predict_length)
        deviation = np.asarray([(s-self.weight*l-self.bias)/np.sqrt(l) for s, l in zip(score, predict_length)])
        self.sigma = np.std(deviation)

    def select(self, predict_path, score_path):
        uttid_choosed = set()
        with open(predict_path, 'r') as preader, open(score_path, 'r') as sreader:
            for pline, sline in zip(preader, sreader):
                pline_split = pline.strip().split()
                sline_split = sline.strip().split()
                assert(len(pline_split) <= 2 and len(sline_split) == 2 and pline_split[0] == sline_split[0])
                predict_length = len(pline_split[1]) if len(pline_split) == 2 else 0.000001
                score = float(sline_split[1])
                filtering_score = (float(score) - predict_length * self.weight - self.bias) / (self.sigma * math.sqrt(predict_length))
                if filtering_score > self.cutoff:
                    uttid_choosed.add(pline_split[0])
        return uttid_choosed

def filtering(uttids, source_data, candidate_data, result_dir):
    with open(os.path.join(result_dir, 'wav.scp'), 'w') as wav_writer, \
         open(source_data, 'r') as wav_reader:
        for wav_line in wav_reader:
            wav_line_split = wav_line.strip().split()
            if wav_line_split[0] in uttids:
                wav_writer.write(wav_line)
    with open(os.path.join(result_dir, 'text'), 'w') as text_writer, \
         open(candidate_data, 'r') as text_reader:
        for text_line in text_reader:
            text_line_split = text_line.strip().split()
            if text_line_split[0] in uttids:
                text_writer.write(text_line)

if __name__ == "__main__":
    args = get_args()
    dev_filter = Filter(args.cutoff, args.test_data, args.test_score)
    uttids = dev_filter.select(args.candidate_data, args.candidate_score)
    print("choose {} utterances in candidate set".format(len(uttids)))
    filtering(uttids, args.source_data, args.candidate_data, args.result_dir)

