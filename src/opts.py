import argparse
import os
import ref

class opts():
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    
  def init(self):
    self.parser.add_argument('-expID', default = 'all_cam_0', help = 'Experiment ID')
    self.parser.add_argument('-test', action = 'store_true', help = 'test')
    self.parser.add_argument('-DEBUG', type = int, default = 0, help = 'DEBUG level')
    self.parser.add_argument('-demo', default = '', help = 'path/to/demo/image')
    
    self.parser.add_argument('-loadModel', default = 'model_weights/hgreg-3d.pth', help = 'Provide full path to a previously trained model')
    self.parser.add_argument('-nFeats', type = int, default = 256, help = '# features in the hourglass')
    self.parser.add_argument('-nStack', type = int, default = 2, help = '# hourglasses to stack')
    self.parser.add_argument('-nModules', type = int, default = 2, help = '# residual modules at each hourglass')

    self.parser.add_argument('-LR', type = float, default = 2.5e-4, help = 'Learning Rate')
    self.parser.add_argument('-dropLR', type = int, default = 1000000, help = 'drop LR')
    self.parser.add_argument('-nEpochs', type = int, default = 20, help = '#training epochs')
    self.parser.add_argument('-valIntervals', type = int, default = 1, help = '#valid intervel')
    self.parser.add_argument('-trainBatch', type = int, default = 24, help = 'Mini-batch size')
    
    self.parser.add_argument('-nRegModules', type = int, default = 2, help = '#depth regression modules')
    self.parser.add_argument('-ratio3D', type = int, default = 1, help = 'weak label data ratio')
    self.parser.add_argument('-regWeight', type = float, default = 0.1, help = 'depth regression loss weight')
    self.parser.add_argument('-varWeight', type = float, default = 0.01, help = 'variance loss weight')
    
    self.parser.add_argument('--gpus', default='0,1', help='-1 for CPU')
      
  def parse(self):
    self.init()  
    self.opt = self.parser.parse_args()
    self.opt.saveDir = os.path.join(ref.expDir, self.opt.expID)
    if self.opt.DEBUG > 0:
      ref.nThreads = 1

    args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                if not name.startswith('_'))
    refs = dict((name, getattr(ref, name)) for name in dir(ref)
                if not name.startswith('_'))
    if not os.path.exists(self.opt.saveDir):
      os.makedirs(self.opt.saveDir)
    file_name = os.path.join(self.opt.saveDir, 'opt.txt')
    with open(file_name, 'wt') as opt_file: 
      opt_file.write('==> Args:\n')
      for k, v in sorted(args.items()):
         opt_file.write('  %s: %s\n' % (str(k), str(v)))
      opt_file.write('==> Args:\n')
      for k, v in sorted(refs.items()):
         opt_file.write('  %s: %s\n' % (str(k), str(v)))

    self.opt.root_dir = os.path.join(os.path.dirname(__file__), '..')
    self.opt.data_dir = os.path.join(self.opt.root_dir, 'data')
    self.opt.gpus = [int(gpu) for gpu in self.opt.gpus.split(',')]
    self.opt.train_half = False
    # self.opt.scale_loss3d = 5
    self.opt.num_views = 4
    self.opt.freeze_layers = True
    self.opt.num_freeze = 2
    self.opt.multi_class = False
    self.opt.err_thresh = 2
    self.opt.ocv_thresh = 0.5
    self.opt.err_reg = False
    self.opt.err_cam = False
    if self.opt.err_reg:
      self.opt.num_freeze = 3

    return self.opt
