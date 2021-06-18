import torch
import torch.nn.functional as F

from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.utils.common import (IGNORE_ID, add_eos, softmax_T, get_attention_map)

class Loss(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        ctc_weight: float = 0.5,
        distill_weight: float = 0.02,
        att_distill_weight: float = 1000,
        temp_scalar: int = 1,
        ignore_id: int = IGNORE_ID,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        reduce: bool = True,
        device: torch.device = torch.device("cpu"),
        teacher: bool = False,
    ):
        super().__init__()
        self.eos = vocab_size - 1
        self.device = device
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.temp_scalar = temp_scalar
        self.distill_weight = distill_weight
        self.att_distill_weight = att_distill_weight
        self.teacher = teacher

        reduction_type = "sum" if reduce else "none"
        self.ctc_loss = torch.nn.CTCLoss(reduction=reduction_type)

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

    def get_losses(
        self,
        outputs: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ):
        loss = torch.zeros(1).to(self.device)
        loss_dict = dict()
        base_outputs = outputs[0]
        if base_outputs.get("encoder_proj_output") is not None and base_outputs.get("encoder_out_lens") is not None and self.ctc_weight != 0.0:
            loss_ctc = self._calc_ctc_loss(base_outputs['encoder_proj_output'], base_outputs['encoder_out_lens'], text, text_lengths)
            loss_dict['loss_ctc'] = loss_ctc
            loss += loss_ctc * self.ctc_weight
        if  base_outputs.get("decoder_output") is not None and self.ctc_weight != 1.0:
            loss_att = self._calc_att_loss(base_outputs['decoder_output'], text)
            loss_dict['loss_att'] = loss_att
            loss += loss_att * (1.0 - self.ctc_weight)
        if self.teacher:
            assert(len(outputs) > 1)
            if self.ctc_weight != 0.0:
                loss_distill = self._calc_distilling_loss(outputs[0]['encoder_proj_output'], outputs[1]['encoder_proj_output'])
                loss_dict['loss_distill'] = loss_distill
                loss += loss_distill * self.distill_weight
            loss_att_distill = self._calc_attention_distilling_loss(outputs[0]['encoder_proj_output'], outputs[0]['encoder_output'], outputs[1]['encoder_proj_output'], outputs[1]['encoder_output'], outputs[0]['proj_weight'], outputs[1]['proj_weight'])
            loss_dict['loss_att_distill'] = loss_att_distill
            loss += loss_att_distill * self.att_distill_weight
        loss_dict['loss'] = loss
        return loss_dict

    def _calc_ctc_loss(self, encoder_out, encoder_out_lens, text, text_lengths):
        encoder_out = F.log_softmax(encoder_out, dim=-1)
        # encoder_out: (B, L, D) -> (L, B, D)
        encoder_out = encoder_out.transpose(0, 1)
        loss = self.ctc_loss(encoder_out, text, encoder_out_lens, text_lengths)
        loss = loss / encoder_out.size(1)
        return loss

    def _calc_att_loss(self, decoder_out, ys_pad: torch.Tensor):
        ys_out_pad = add_eos(ys_pad, self.eos, self.ignore_id)
        loss = self.criterion_att(decoder_out, ys_out_pad)
        return loss

    def _calc_distilling_loss(self, student_encoder_proj_out, teacher_encoder_proj_out):
        p_student = softmax_T(student_encoder_proj_out, dim=-1, T=self.temp_scalar)
        p_teacher = softmax_T(teacher_encoder_proj_out, dim=-1, T=self.temp_scalar)
        loss = F.kl_div(p_student.log(), p_teacher)
        return loss

    def _calc_attention_distilling_loss(self, student_encoder_proj_out, student_encoder_out, teacher_encoder_proj_out, teacher_encoder_out, student_proj_weight, teacher_proj_weight):
        q_student = get_attention_map(student_encoder_proj_out, student_encoder_out, student_proj_weight)
        q_teacher = get_attention_map(teacher_encoder_proj_out, teacher_encoder_out, teacher_proj_weight)
        def at(x):
            return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
        loss = (at(q_teacher) - at(q_student)).pow(2).mean()
        return loss

