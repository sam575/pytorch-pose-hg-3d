import sys
import torch
from opts import opts
import ref
from utils.debugger import Debugger
from utils.eval import getPreds
import cv2
import numpy as np

# def recursion_change_bn(module):
#     if isinstance(module, torch.nn.BatchNorm2d):
#         module.track_running_stats = 1
#     else:
#         for i, (name, module1) in enumerate(module._modules.items()):
#             module1 = recursion_change_bn(module1)
#     return module

def main():
  opt = opts().parse()
  if opt.loadModel != 'none':
    model = torch.load(opt.loadModel).cuda()
  else:
    model = torch.load('model_weights/hgreg-3d.pth').cuda()
    # model = torch.load('model_weights/hgreg-3d.pth', encoding='latin1').cuda()


  # ## tackling batchnorm error
  # for i, (name, module) in enumerate(model._modules.items()):
  #   module = recursion_change_bn(model)
  # model.eval()

  img = cv2.imread(opt.demo)
  input = torch.from_numpy(img.transpose(2, 0, 1)).float() / 256.
  input = input.view(1, input.size(0), input.size(1), input.size(2))
  input_var = torch.autograd.Variable(input).float().cuda()
  output = model(input_var)
  pred = getPreds((output[-2].data).cpu().numpy())[0] * 4
  reg = (output[-1].data).cpu().numpy().reshape(pred.shape[0], 1)
  debugger = Debugger()
  debugger.addImg((input[0].numpy().transpose(1, 2, 0)*256).astype(np.uint8))
  debugger.addPoint2D(pred, (255, 0, 0))
  debugger.addPoint3D(np.concatenate([pred, (reg + 1) / 2. * 256], axis = 1))
  debugger.showImg(pause = True)
  debugger.show3D()

if __name__ == '__main__':
  main()
