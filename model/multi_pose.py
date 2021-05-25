from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from model.losses import FocalLoss, RegL1Loss, RegLoss, RegWeightedL1Loss
from model.decode import multi_pose_decode
from model.utils import _sigmoid, flip_tensor, flip_lr_off, flip_lr
# from utils.debugger import Debugger
# from utils.post_process import multi_pose_post_process
from model.oracle_utils import gen_oracle_map
# from .base_trainer import BaseTrainer

class MultiPoseLoss(torch.nn.Module):
  def __init__(self, opt):
    super(MultiPoseLoss, self).__init__()
    self.crit = FocalLoss()
    self.crit_hm_hp = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_kp = RegWeightedL1Loss() if not opt.dense_hp else \
                   torch.nn.L1Loss(reduction='sum')
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
                    RegLoss() if opt.reg_loss == 'sl1' else None
    self.opt = opt

  def forward(self, outputs, batch):
    opt = self.opt
    hm_loss, wh_loss, off_loss = 0, 0, 0
    lm_loss, off_loss, hm_hp_loss, hp_offset_loss = 0, 0, 0, 0
    for s in range(opt.num_stacks):
      output = outputs[s]
      output['hm'] = output['hm']
      # if opt.hm_hp and not opt.mse_loss:
      #   output['hm_hp'] = _sigmoid(output['hm_hp'])
      
      if opt.eval_oracle_hmhp:
        output['hm_hp'] = batch['hm_hp']
      if opt.eval_oracle_hm:
        output['hm'] = batch['hm']
      if opt.eval_oracle_kps:
        if opt.dense_hp:
          output['hps'] = batch['dense_hps']
        else:
          output['hps'] = torch.from_numpy(gen_oracle_map(
            batch['hps'].detach().cpu().numpy(), 
            batch['ind'].detach().cpu().numpy(), 
            opt.output_res, opt.output_res)).to(opt.device)
      if opt.eval_oracle_hp_offset:
        output['hp_offset'] = torch.from_numpy(gen_oracle_map(
          batch['hp_offset'].detach().cpu().numpy(), 
          batch['hp_ind'].detach().cpu().numpy(), 
          opt.output_res, opt.output_res)).to(opt.device)


      hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks          # 1. focal loss,求目标的中心，
      if opt.wh_weight > 0:
        wh_loss += self.crit_reg(output['wh'], batch['reg_mask'],               # 2. 人脸bbox高度和宽度的loss
                                 batch['ind'], batch['wh'], batch['wight_mask']) / opt.num_stacks
      if opt.reg_offset and opt.off_weight > 0:
        off_loss += self.crit_reg(output['hm_offset'], batch['reg_mask'],             # 3. 人脸bbox中心点下采样，所需要的偏差补偿
                                  batch['ind'], batch['hm_offset'], batch['wight_mask']) / opt.num_stacks

      if opt.dense_hp:
        mask_weight = batch['dense_hps_mask'].sum() + 1e-4
        lm_loss += (self.crit_kp(output['hps'] * batch['dense_hps_mask'],
                                 batch['dense_hps'] * batch['dense_hps_mask']) / 
                                 mask_weight) / opt.num_stacks
      else:
        lm_loss += self.crit_kp(output['landmarks'], batch['hps_mask'],               # 4. 关节点的偏移
                                batch['ind'], batch['landmarks']) / opt.num_stacks

      # if opt.reg_hp_offset and opt.off_weight > 0:                              # 关节点的中心偏移
      #   hp_offset_loss += self.crit_reg(
      #     output['hp_offset'], batch['hp_mask'],
      #     batch['hp_ind'], batch['hp_offset']) / opt.num_stacks
      # if opt.hm_hp and opt.hm_hp_weight > 0:                                    # 关节点的热力图
      #   hm_hp_loss += self.crit_hm_hp(
      #     output['hm_hp'], batch['hm_hp']) / opt.num_stacks

    loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
           opt.off_weight * off_loss + opt.lm_weight * lm_loss
    
    # loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'lm_loss': lm_loss,
    #               'wh_loss': wh_loss, 'off_loss': off_loss}
    # return loss, loss_stats
    return loss