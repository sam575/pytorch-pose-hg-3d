import os
import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import pickle
from utils.image import flip, shuffle_lr
from utils.image import draw_gaussian, adjust_aspect_ratio
from utils.image import get_affine_transform, affine_transform
from utils.image import transform_preds

from utils.img_old import Crop, DrawGaussian, Transform3D
# from utils.image import Crop
import ref
import h5py

import pdb

class H36M(data.Dataset):
  def __init__(self, opt, split):
    self.split = split
    self.opt = opt

    print '==> initializing 3D {} data.'.format(split)
    annot = {}
    tags = ['action', 'bbox', 'id', 'joint_2d_gt', 'joint_2d_pred', 'joint_3d_gt', 'mpjpe', 'subaction', 'subject']

    # if self.split == 'train':
    self.annot_path = os.path.join(self.opt.data_dir, 'h36m', 'classifier_data_'+ self.split +'.h5')

    f = h5py.File(self.annot_path, 'r')
    for tag in tags:
      annot[tag] = np.asarray(f[tag]).copy()
    f.close()
    
    # ids = np.arange(annot['id'].shape[0])[annot['istrain'] == (1 if split == 'train' else 0)]
    # for tag in tags:
    #   annot[tag] = annot[tag][ids]
    
    self.root = 7
    
    self.annot = annot
    self.num_views = self.annot['mpjpe'].shape[-1]
    self.nSamples = len(self.annot['id']) * self.num_views
    self.image_dir = os.path.join(self.opt.data_dir, 'h36m', 'images')
    self.err_thresh = 0.5
    self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
    
    print 'Loaded 3D {} {} samples'.format(split, len(self.annot['id']))
  
  def LoadImage(self, index, cam_num):
    folder = 's_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}'.format(self.annot['subject'][index], self.annot['action'][index], \
              self.annot['subaction'][index], cam_num+1)
    path = '{}/{}/{}_{:06d}.jpg'.format(self.image_dir, folder, folder, self.annot['id'][index])
    # print 'path', path
    img = cv2.imread(path)

    if img is None:
      pass
      # print('Missing image:')
      # print(path)
      # img = np.zeros((256,256,3))

    return img
  
  def GetPartInfo(self, index, cam_num):
    pts = self.annot['joint_2d_gt'][index][cam_num].copy()
    pts_3d_mono = self.annot['joint_3d_gt'][index][cam_num].copy()
    pts_3d = self.annot['joint_3d_gt'][index][cam_num].copy()
    c = np.ones(2) * ref.h36mImgSize / 2
    s = ref.h36mImgSize * 1.0
    
    pts_3d = pts_3d - pts_3d[self.root]
    
    s2d, s3d = 0, 0
    for e in ref.edges:
      s2d += ((pts[e[0]] - pts[e[1]]) ** 2).sum() ** 0.5
      s3d += ((pts_3d[e[0], :2] - pts_3d[e[1], :2]) ** 2).sum() ** 0.5
    scale = s2d / s3d
      
    for j in range(ref.nJoints):
      pts_3d[j, 0] = pts_3d[j, 0] * scale + pts[self.root, 0]
      pts_3d[j, 1] = pts_3d[j, 1] * scale + pts[self.root, 1]
      pts_3d[j, 2] = pts_3d[j, 2] * scale + ref.h36mImgSize / 2
    return pts, c, s, pts_3d, pts_3d_mono
  
      
  def __getitem__(self, index):
    # if self.split == 'train':
      # index = np.random.randint(self.nSamples * self.num_views)

    index = index/self.num_views
    cam_num = index % self.num_views
    # cam_num = np.random.randint(self.num_views)
    err = self.annot['mpjpe'][index]
    min_err = np.min(err)
    min_err_ind = np.argmin(err)
    ocv_gt = np.zeros(self.num_views)
    ocv_gt[min_err_ind] = 1
    # ocv_gt[err < (min_err + self.err_thresh)] = 1

    # correct camera indices w.r.t actual rotation
    ## 2 4
    ## 1 3
    ## 1 2 4 3
    temp = ocv_gt[-1]
    ocv_gt[-1] = ocv_gt[-2]
    ocv_gt[-2] = temp
    # correct cam_num
    if cam_num == 3:
      roll = 2
    elif cam_num == 2:
      roll = 3
    else:
      roll = cam_num
    # rotate w.r.t cam_num 
    ocv_gt = np.roll(ocv_gt, -roll)

    info = {'index': index, 'cam_num': cam_num}

    img = self.LoadImage(index, cam_num)

    if img is None:
      img = np.zeros((224,224,3))
      # print(info)
      pass

    pts, c, s, pts_3d, pts_3d_mono = self.GetPartInfo(index, cam_num)   
    pts_3d[7] = (pts_3d[12] + pts_3d[13]) / 2
      
    # inp = Crop(img, c, s, 0, ref.inputRes) / 256
    img = Crop(img, c, s, 0, ref.inputRes)

    # outMap = np.zeros((ref.nJoints, ref.outputRes, ref.outputRes))
    # outReg = np.zeros((ref.nJoints, 3))
    # for i in range(ref.nJoints):
      # pt = Transform3D(pts_3d[i], c, s, 0, ref.outputRes)
      # if pts[i][0] > 1:
        # outMap[i] = DrawGaussian(outMap[i], pt[:2], ref.hmGauss) 
      # outReg[i, 2] = pt[2] / ref.outputRes * 2 - 1

    # remove division by 256 after crop
    if img.shape[0] == 3:
      # 3,224,224 -> 224,224,3
      img = img.transpose(1,2,0)
    img = (img.astype(np.float32) / 256. - self.mean) / self.std
    img = img.transpose(2, 0, 1) # commented transpose in Crop

    # inp = torch.from_numpy(inp).float()
    inp = torch.from_numpy(img)
    ocv_gt = torch.from_numpy(ocv_gt)
    # return inp, outMap, outReg, pts_3d_mono
    return inp, ocv_gt, info
    
  def __len__(self):
    return self.nSamples