import torch
import numpy as np
from utils.image import flip, shuffle_lr
from utils.eval import accuracy, get_preds, mpjpe, get_preds_3d, accuracy_ocv
import cv2
from progress.bar import Bar
from utils.debugger import Debugger
from models.losses import RegLoss, FusionLoss
import time

import torch.nn as nn
import pdb

def step(split, epoch, opt, data_loader, model, optimizer=None):
  if split == 'train':
    model.train()
  else:
    model.eval()
  
  # crit = torch.nn.MSELoss()
  # crit_3d = FusionLoss(opt.device, opt.weight_3d, opt.weight_var)

  # crit_ocv = nn.BCEWithLogitsLoss()
  crit_ocv = nn.CrossEntropyLoss()

  # acc_idxs = data_loader.dataset.acc_idxs
  # edges = data_loader.dataset.edges
  # edges_3d = data_loader.dataset.edges_3d
  # shuffle_ref = data_loader.dataset.shuffle_ref
  # mean = data_loader.dataset.mean
  # std = data_loader.dataset.std
  # convert_eval_format = data_loader.dataset.convert_eval_format

  # Loss, Loss3D = AverageMeter(), AverageMeter()
  # Acc, MPJPE = AverageMeter(), AverageMeter()

  Loss_ocv, Acc_ocv = AverageMeter(), AverageMeter()

  data_time, batch_time = AverageMeter(), AverageMeter()
  preds = []
  time_str = ''

  nIters = len(data_loader)
  if opt.train_half:
    nIters = nIters/2
  bar = Bar('{}'.format(opt.exp_id), max=nIters)
  
  end = time.time()
  for i, batch in enumerate(data_loader):
    if i>=nIters:
      break

    data_time.update(time.time() - end)
    # for k in batch:
    #   if k != 'meta':
    #     batch[k] = batch[k].cuda(device=opt.device, non_blocking=True)
    # gt_2d = batch['meta']['pts_crop'].cuda(
    #   device=opt.device, non_blocking=True).float() / opt.output_h

    img, ocv_gt, info = batch

    if i==0:
      np.savez(split+'_debug.npz', img=img.numpy(), ocv_gt=ocv_gt.numpy(), info=info)

    img = img.cuda(device=opt.device, non_blocking=True)
    ocv_gt = ocv_gt.cuda(device=opt.device, non_blocking=True)
    output = model(img)



    # loss = crit(output[-1]['hm'], batch['target'])
    # loss_3d = crit_3d(
    #   output[-1]['depth'], batch['reg_mask'], batch['reg_ind'], 
    #   batch['reg_target'],gt_2d)
    # for k in range(opt.num_stacks - 1):
    #   loss += crit(output[k], batch['target'])
    #   loss_3d = crit_3d(
    #     output[-1]['depth'], batch['reg_mask'], batch['reg_ind'], 
    #     batch['reg_target'], gt_2d)
    # loss += loss_3d
    # loss = crit_ocv(output, ocv_gt)
    loss = crit_ocv(output, torch.argmax(ocv_gt, 1))
    preds = torch.argmax(output, 1)

    if split == 'train':
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    # else:
    #   input_ = batch['input'].cpu().numpy().copy()
    #   input_[0] = flip(input_[0]).copy()[np.newaxis, ...]
    #   input_flip_var = torch.from_numpy(input_).cuda(
    #     device=opt.device, non_blocking=True)
    #   output_flip_ = model(input_flip_var)
    #   output_flip = shuffle_lr(
    #     flip(output_flip_[-1]['hm'].detach().cpu().numpy()[0]), shuffle_ref)
    #   output_flip = output_flip.reshape(
    #     1, opt.num_output, opt.output_h, opt.output_w)
    #   output_depth_flip = shuffle_lr(
    #     flip(output_flip_[-1]['depth'].detach().cpu().numpy()[0]), shuffle_ref)
    #   output_depth_flip = output_depth_flip.reshape(
    #     1, opt.num_output, opt.output_h, opt.output_w)
    #   output_flip = torch.from_numpy(output_flip).cuda(
    #     device=opt.device, non_blocking=True)
    #   output_depth_flip = torch.from_numpy(output_depth_flip).cuda(
    #     device=opt.device, non_blocking=True)
    #   output[-1]['hm'] = (output[-1]['hm'] + output_flip) / 2
    #   output[-1]['depth'] = (output[-1]['depth'] + output_depth_flip) / 2
      # pred = get_preds(output[-1]['hm'].detach().cpu().numpy())
      # preds.append(convert_eval_format(pred, conf, meta)[0])
    
    acc = accuracy_ocv(preds, torch.argmax(ocv_gt, 1))
    Loss_ocv.update(loss.item(), img.size(0))
    Acc_ocv.update(acc, img.size(0))
    # Loss.update(loss.item(), batch['input'].size(0))
    # Loss3D.update(loss_3d.item(), batch['input'].size(0))
    # Acc.update(accuracy(output[-1]['hm'].detach().cpu().numpy(), 
    #                     batch['target'].detach().cpu().numpy(), acc_idxs))
    # mpeje_batch, mpjpe_cnt = mpjpe(output[-1]['hm'].detach().cpu().numpy(),
    #                                output[-1]['depth'].detach().cpu().numpy(),
    #                                batch['meta']['gt_3d'].detach().numpy(),
    #                                convert_func=convert_eval_format)
    # MPJPE.update(mpeje_batch, mpjpe_cnt)
   
    batch_time.update(time.time() - end)
    end = time.time()
    if not opt.hide_data_time:
      time_str = ' |Data {dt.avg:.3f}s({dt.val:.3f}s)' \
                 ' |Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
      
    # Bar.suffix = '{split}: [{0}][{1}/{2}] |Total {total:} |ETA {eta:} '\
    #              '|Loss {loss.avg:.5f} |Loss3D {loss_3d.avg:.5f}'\
    #              '|Acc {Acc.avg:.4f} |MPJPE {MPJPE.avg:.2f}'\
    #              '{time_str}'.format(epoch, i, nIters, total=bar.elapsed_td, 
    #                                  eta=bar.eta_td, loss=Loss, Acc=Acc, 
    #                                  split=split, time_str=time_str,
                                     # MPJPE=MPJPE, loss_3d=Loss3D)

    Bar.suffix = '{split}: [{0}][{1}/{2}] |Total {total:} |ETA {eta:} '\
                 '|Loss_ocv {loss.avg:.5f}'\
                 '|Acc_ocv {Acc.avg:.4f}'\
                 '|loss_batch {loss_batch:.4f}'\
                 '|acc_batch {acc_batch:.4f}'\
                 '{time_str}'.format(epoch, i, nIters, total=bar.elapsed_td, 
                                     eta=bar.eta_td, loss=Loss_ocv, Acc=Acc_ocv,
                                     loss_batch=loss.item(), acc_batch=acc, 
                                     split=split, time_str=time_str)

    if opt.print_iter > 0:
      if i % opt.print_iter == 0:
        print('{}| {}'.format(opt.exp_id, Bar.suffix))
    else:
      bar.next()
    if opt.debug >= 2:
      gt = get_preds(batch['target'].cpu().numpy()) * 4
      pred = get_preds(output[-1]['hm'].detach().cpu().numpy()) * 4
      debugger = Debugger(ipynb=opt.print_iter > 0, edges=edges)
      img = (
        batch['input'][0].cpu().numpy().transpose(1, 2, 0) * std + mean) * 256
      img = img.astype(np.uint8).copy()
      debugger.add_img(img)
      debugger.add_mask(
        cv2.resize(batch['target'][0].cpu().numpy().max(axis=0), 
                   (opt.input_w, opt.input_h)), img, 'target')
      debugger.add_mask(
        cv2.resize(output[-1]['hm'][0].detach().cpu().numpy().max(axis=0), 
                   (opt.input_w, opt.input_h)), img, 'pred')
      debugger.add_point_2d(gt[0], (0, 0, 255))
      debugger.add_point_2d(pred[0], (255, 0, 0))
      debugger.add_point_3d(
        batch['meta']['gt_3d'].detach().numpy()[0], 'r', edges=edges_3d)
      pred_3d = get_preds_3d(output[-1]['hm'].detach().cpu().numpy(), 
                             output[-1]['depth'].detach().cpu().numpy())
      debugger.add_point_3d(convert_eval_format(pred_3d[0]), 'b',edges=edges_3d)
      debugger.show_all_imgs(pause=False)
      debugger.show_3d()

    # pdb.set_trace()

  bar.finish()
  # return {'loss': Loss.avg, 
  #         'acc': Acc.avg, 
  #         'mpjpe': MPJPE.avg,
  #         'time': bar.elapsed_td.total_seconds() / 60.}, preds
  return {'loss': Loss_ocv.avg, 
          'acc': Acc_ocv.avg, 
          'time': bar.elapsed_td.total_seconds() / 60.}, preds
  
def train_3d(epoch, opt, train_loader, model, optimizer):
  return step('train', epoch, opt, train_loader, model, optimizer)
  
def val_3d(epoch, opt, val_loader, model):
  return step('val', epoch, opt, val_loader, model)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
