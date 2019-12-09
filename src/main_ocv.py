import os
import time
import datetime

import torch
import torch.utils.data
from opts import opts
import ref
from utils.utils import adjust_learning_rate
# from datasets.fusion import Fusion
# from datasets.h36m import H36M
# from datasets.mpii import MPII
from utils.logger import Logger
# from train import train, val

# from train_ocv import train, val
# from models.hg_3d_ocv import HourglassNet3D
# from datasets.h36m_ocv import H36M

from train_ocj import train, val
from models.hg_3d_ocj import HourglassNet3D
from datasets.h36m_ocj import H36M


def freeze_model(model, opt):
  print('Freezing initial layers')
  modules = list(model.children())
  # freeze last but one module
  for child in modules[:-opt.num_freeze]:
    # print(child)
    for name, param in child.named_parameters():
      param.requires_grad = False
      # print(name)

  print('Unfreezed parameters')
  print(modules[-opt.num_freeze:])

def main():
  opt = opts().parse()
  now = datetime.datetime.now()
  logger = Logger(opt.saveDir + '/logs_{}'.format(now.isoformat()))

  # if opt.loadModel != 'none':
  #   model = torch.load(opt.loadModel).cuda()
  # else:
  # model = HourglassNet3D(opt.nStack, opt.nModules, opt.nFeats, opt.nRegModules).cuda()
  model = HourglassNet3D(opt.nStack, opt.nModules, opt.nFeats, opt.nRegModules, opt)


  if opt.loadModel != '':
    # checkpoint = torch.load(opt.loadModel, map_location=lambda storage, loc: storage)
    checkpoint = torch.load(opt.loadModel, map_location=lambda storage, loc: storage, encoding='latin1')
    print('loaded {}'.format(opt.loadModel))
    # print('loaded {}, epoch {}'.format(opt.loadModel, checkpoint['epoch']))
    if type(checkpoint) == type({}):
      state_dict = checkpoint['state_dict']
    else:
      state_dict = checkpoint.state_dict()
    print(model.load_state_dict(state_dict, strict=False))

  if opt.freeze_layers:
    freeze_model(model, opt)

  opt.device = torch.device('cuda:{}'.format(opt.gpus[0]))
  if len(opt.gpus) > 1:
    model = torch.nn.DataParallel(model, device_ids=opt.gpus).cuda(opt.device)
  else:
    model = model.cuda(opt.device)
  
  criterion = torch.nn.MSELoss().cuda()
  optimizer = torch.optim.RMSprop(model.parameters(), opt.LR, 
                                  alpha = ref.alpha, 
                                  eps = ref.epsilon, 
                                  weight_decay = ref.weightDecay, 
                                  momentum = ref.momentum)

  # if opt.ratio3D < ref.eps:
  #   val_loader = torch.utils.data.DataLoader(
  #       MPII(opt, 'val', returnMeta = True), 
  #       batch_size = opt.trainBatch, 
  #       shuffle = False,
  #       num_workers = int(ref.nThreads)
  #   )
  # else:
  if opt.test:
    val_shuffle = False
  else:
    val_shuffle = True
  val_loader = torch.utils.data.DataLoader(
      H36M(opt, 'val'), 
      batch_size = opt.trainBatch, 
      # batch_size = 1, 
      shuffle = True,
      num_workers = int(ref.nThreads)
    )
  

  if opt.test:
    val(0, opt, val_loader, model, criterion)
    return

  # train_loader = torch.utils.data.DataLoader(
  #     Fusion(opt, 'train'), 
  #     batch_size = opt.trainBatch, 
  #     shuffle = True if opt.DEBUG == 0 else False,
  #     num_workers = int(ref.nThreads)
  # )

  train_loader = torch.utils.data.DataLoader(
      H36M(opt, 'train'), 
      batch_size = opt.trainBatch, 
      shuffle = True if opt.DEBUG == 0 else False,
      num_workers = int(ref.nThreads)
  )

  for epoch in range(1, opt.nEpochs + 1):
    loss_train, acc_train, mpjpe_train, loss3d_train, acc_ocv_train, loss_ocv_train = train(epoch, opt, train_loader, model, criterion, optimizer)
    logger.scalar_summary('loss_train', loss_train, epoch)
    logger.scalar_summary('acc_train', acc_train, epoch)
    logger.scalar_summary('acc_ocv_train', acc_ocv_train, epoch)
    logger.scalar_summary('loss_ocv_train', loss_ocv_train, epoch)
    logger.scalar_summary('mpjpe_train', mpjpe_train, epoch)
    logger.scalar_summary('loss3d_train', loss3d_train, epoch)
    if epoch % opt.valIntervals == 0:
      loss_val, acc_val, mpjpe_val, loss3d_val, acc_ocv_val, loss_ocv_val = val(epoch, opt, val_loader, model, criterion)
      logger.scalar_summary('loss_val', loss_val, epoch)
      logger.scalar_summary('acc_val', acc_val, epoch)
      logger.scalar_summary('acc_ocv_val', acc_ocv_val, epoch)
      logger.scalar_summary('loss_ocv_val', loss_ocv_val, epoch)
      logger.scalar_summary('mpjpe_val', mpjpe_val, epoch)
      logger.scalar_summary('loss3d_val', loss3d_val, epoch)
      torch.save(model, os.path.join(opt.saveDir, 'model_{}.pth'.format(epoch)))
      logger.write('{:8f} {:8f} {:8f} {:8f} {:8f} {:8f} {:8f} {:8f} {:8f} {:8f} {:8f} {:8f} \n'.format(loss_train, acc_train,  mpjpe_train, loss3d_train, acc_ocv_train, loss_ocv_train, loss_val, acc_val, mpjpe_val, loss3d_val, acc_ocv_val, loss_ocv_val))
    else:
      logger.write('{:8f} {:8f} {:8f} {:8f} {:8f} {:8f} \n'.format(loss_train, acc_train, mpjpe_train, loss3d_train, acc_ocv_train, loss_ocv_train))
    adjust_learning_rate(optimizer, epoch, opt.dropLR, opt.LR)
  logger.close()

if __name__ == '__main__':
  main()
