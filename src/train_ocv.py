import torch
import numpy as np
from utils.utils import AverageMeter
from utils.eval import Accuracy, getPreds, MPJPE
from utils.debugger import Debugger
from models.layers.FusionCriterion import FusionCriterion
import cv2
import ref
from progress.bar import Bar

import torch.nn.functional as F
import torch.nn as nn
import os
import pdb

def step(split, epoch, opt, dataLoader, model, criterion, optimizer = None):
  if split == 'train':
    model.train()
  else:
    model.eval()
  Loss, Acc, Loss_ocv, Acc_ocv = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
  Mpjpe, Loss3D = AverageMeter(), AverageMeter()

  crit_ocv = nn.CrossEntropyLoss()
  crit_multi_margin = nn.MultiLabelMarginLoss()
  
  nIters = len(dataLoader)
  if opt.train_half:
    nIters = int(nIters/3)
  bar = Bar('==>', max=nIters)
  
  if split == "val":
    total_preds = np.zeros((dataLoader.dataset.nSamples, opt.num_views))
    total_gt = np.zeros((dataLoader.dataset.nSamples, opt.num_views))

  for i, (input, target2D, target3D, meta, ocv_gt, multi_cam_ind, info) in enumerate(dataLoader):
    if i>=nIters:
      break
    input_var = torch.autograd.Variable(input).float().cuda()
    target2D_var = torch.autograd.Variable(target2D).float().cuda()
    target3D_var = torch.autograd.Variable(target3D).float().cuda()

    # target_ocv_var = torch.autograd.Variable(target_ocv).float().cuda()
    
    if opt.err_reg:
      output, preds_ocv, preds_ocv_cls = model(input_var)
    else:
      output, preds_ocv = model(input_var)
    reg = output[opt.nStack]
    if opt.DEBUG >= 2:
      gt = getPreds(target2D.cpu().numpy()) * 4
      pred = getPreds((output[opt.nStack - 1].data).cpu().numpy()) * 4
      debugger = Debugger()
      debugger.addImg((input[0].numpy().transpose(1, 2, 0)*256).astype(np.uint8))
      debugger.addPoint2D(pred[0], (255, 0, 0))
      debugger.addPoint2D(gt[0], (0, 0, 255))
      # debugger.showImg()
      debugger.saveImg('debug/{}.png'.format(i))

    # print(preds_ocv.shape, ocv_gt.shape)
    ocv_gt = ocv_gt.cuda().float()
    # pdb.set_trace()
    # ocv_gt = ocv_gt.cuda()
    # print(preds_ocv.shape, ocv_gt.shape)

    loss = FusionCriterion(opt.regWeight, opt.varWeight)(reg, target3D_var)
    Loss3D.update(loss.data[0], input.size(0))
    # pdb.set_trace()
    # loss += loss_3d.item()

    # pdb.set_trace()
    if opt.multi_class:
      # multi_cam_ind = multi_cam_ind.cuda().long()
      # preds_ocv = F.sigmoid(preds_ocv)
      # loss_ocv = crit_multi_margin(preds_ocv, multi_cam_ind)
      loss_ocv = F.binary_cross_entropy_with_logits(preds_ocv.float(), ocv_gt.float())
      preds_ocv = F.sigmoid(preds_ocv)
    elif opt.err_reg:
      loss_ocv = 0.01 * F.mse_loss(preds_ocv, ocv_gt)
      loss_ocv += crit_ocv(preds_ocv_cls, ocv_gt)
    else:
      loss_ocv = crit_ocv(preds_ocv, torch.argmax(ocv_gt, 1))
    Loss_ocv.update(loss_ocv.item(), input.size(0))
    loss += loss_ocv

    for k in range(opt.nStack):
      loss += criterion(output[k], target2D_var)

    Loss.update(loss.data[0], input.size(0))
    Acc.update(Accuracy((output[opt.nStack - 1].data).cpu().numpy(), (target2D_var.data).cpu().numpy()))
    mpjpe, num3D = MPJPE((output[opt.nStack - 1].data).cpu().numpy(), (reg.data).cpu().numpy(), meta)

    if opt.multi_class:
      acc_ocv = ((preds_ocv >= opt.ocv_thresh).float() == ocv_gt.float()).float().mean()
    elif opt.err_reg:
      acc_ocv = (torch.argmax(preds_ocv_cls, 1) == torch.argmin(ocv_gt, 1)).float().mean()
    else:
      acc_ocv = (torch.argmax(preds_ocv, 1) == torch.argmax(ocv_gt, 1)).float().mean()
    Acc_ocv.update(acc_ocv, input.size(0))

    # pdb.set_trace()

    if num3D > 0:
      Mpjpe.update(mpjpe, num3D)
    if split == 'train':
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    else:
      preds_ocv_np = preds_ocv.detach().cpu().numpy()
      ocv_gt_np = ocv_gt.detach().cpu().numpy()
      for j in range(input.size(0)):
        ind = info['index'][j]*opt.num_views + info['cam_num'][j]
        total_preds[ind] = preds_ocv_np[j]
        total_gt[ind] = ocv_gt_np[j]

    # pdb.set_trace()
 
    Bar.suffix = '{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss {loss.avg:.6f} | Loss3D {loss3d.avg:.6f} | Loss_ocv {Loss_ocv.avg:.6f} ({Loss_ocv.val:.6f})| Acc {Acc.avg:.6f} | Mpjpe {Mpjpe.avg:.6f} ({Mpjpe.val:.6f})| Acc_ocv {Acc_ocv.avg:.6f} ({Acc_ocv.val:.6f})'.format(epoch, i, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss, Acc=Acc, split = split, Acc_ocv = Acc_ocv, Loss_ocv = Loss_ocv,  Mpjpe=Mpjpe, loss3d = Loss3D)
    # Bar.suffix = '{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss {loss.avg:.6f} | Loss3D {loss3d.avg:.6f} | Acc {Acc.avg:.6f} | Mpjpe {Mpjpe.avg:.6f} ({Mpjpe.val:.6f})'.format(epoch, i, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss, Acc=Acc, split = split, Mpjpe=Mpjpe, loss3d = Loss3D)

    bar.next()

  bar.finish()

  if split == "val":
    save_dic = {'preds_ocv': total_preds, 'gt': total_gt}
    np.save(os.path.join(opt.saveDir, "preds_ocv.npy"), save_dic)

  return Loss.avg, Acc.avg, Mpjpe.avg, Loss3D.avg, Acc_ocv.avg, Loss_ocv.avg
  

def train(epoch, opt, train_loader, model, criterion, optimizer):
  return step('train', epoch, opt, train_loader, model, criterion, optimizer)
  
def val(epoch, opt, val_loader, model, criterion):
  return step('val', epoch, opt, val_loader, model, criterion)
