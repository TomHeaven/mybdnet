import os.path as osp
import torch
import torch.utils.data as data
import data.util as util
import pdb,random
import numpy as np
import cv2
import lmdb,glob
import scipy.io as scio

DEBUG = False

class Real_dynamic(data.Dataset):

    def __init__(self, opt):
        super(Real_dynamic, self).__init__()
        self.opt = opt
        self.half_N_frames = opt['N_frames'] // 2
        # GS: No need LR
        #self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.GT_root = opt['dataroot_GT']
        self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': []}   ## They are flattened
        #### Generate data info and cache data
        
        # subfolders_LQ = util.glob_file_list(self.LQ_root)
        self.subfolders_GT = sorted(glob.glob(self.GT_root+'/*'))
        
                            

    def __getitem__(self, index):  ## The index indicates the frame index which is flattened
        # path_LQ = self.data_info['path_LQ'][index]
        # path_GT = self.data_info['path_GT'][index]
        GT_size = 256   # original = 512
        folder_path = self.subfolders_GT[index]
        img_paths_GT = sorted(glob.glob(folder_path+'/*mat'))[0:self.opt['N_frames']]
        rggb_l = []
        noise_l = []
        for img_path in img_paths_GT:
            mat_data = scio.loadmat(img_path)
            rggb = mat_data['param']['rggb'][0,0]
            noise = mat_data['param']['noise'][0,0]
            wb = mat_data['param']['lum'][0,0]
            noise_l.append(noise)
            rggb_l.append(rggb)
        
        wb = 1.0/wb
        rggb_case = np.stack(rggb_l,0)    
        rggb_in = np.transpose(rggb_case,(0,3,1,2)) # N,C,H,W
         
        noise = np.stack(noise_l,2) 
        #print(noise.shape)
        noise = noise[0,:,0]
        noise_p = (noise[0] + noise[2] + noise[4])/3.0
        noise_r = (noise[1] + noise[3] + noise[5])/3.0
        noise_map = (noise_p + noise_r)**0.5
        noise_map = np.tile(noise_map.reshape(1,1,1,1),[1,1,GT_size,GT_size])
        #LQ_size_tuple = (4,512,512)
        rggb_in = rggb_in[:,:,0:GT_size, 0:GT_size]
        noise_in_unpro_s = np.stack([noise[0],noise[2],noise[2],noise[4]],0).reshape(1,4,1,1)
        noise_in_unpro_r = np.stack([noise[1],noise[3],noise[3],noise[5]],0).reshape(1,4,1,1)
        noise_in_unpro = rggb_in*noise_in_unpro_s + noise_in_unpro_r
        noise_in_unpro = torch.from_numpy(np.ascontiguousarray(noise_in_unpro)).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(rggb_in)).float()
        noise_in = torch.from_numpy(np.ascontiguousarray(noise_map)).float()

        return {'LQs':img_LQs,'noise_in':noise_in,'folder':folder_path,'noise_in_unpro':noise_in_unpro,'wb':wb}

    def __len__(self):
        return len(self.subfolders_GT)
    
class Real_static(data.Dataset):


    def __init__(self, opt):
        super(Real_static, self).__init__()
        self.opt = opt
        self.half_N_frames = opt['N_frames'] // 2
        # GS: No need LR
        #self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.GT_root = opt['dataroot_GT']
        self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': []}   ## They are flattened
        #### Generate data info and cache data
        
        # subfolders_LQ = util.glob_file_list(self.LQ_root)
        self.subfolders_GT = sorted(glob.glob(self.GT_root+'/*'))
        
    def __getitem__(self, index):  ## The index indicates the frame index which is flattened
        # path_LQ = self.data_info['path_LQ'][index]
        # path_GT = self.data_info['path_GT'][index]
        GT_size = 384   # original = 512 
        folder_path = self.subfolders_GT[index]
        img_paths = sorted(glob.glob(folder_path+'/0*mat'))[0:self.opt['N_frames']]
        gt_path = folder_path+'/gt.mat'
        rggb_l = []
        noise_l = []
        for img_path in img_paths:
            mat_data = scio.loadmat(img_path)
            rggb = mat_data['param']['rggb'][0,0]
            noise = mat_data['param']['noise'][0,0]
            wb = mat_data['param']['lum'][0,0]
            noise_l.append(noise)
            if self.opt['pre_demosaic'] == True:
                rggb = util.demosaic(rggb)
            rggb_l.append(rggb)
            if DEBUG:
                print('img_path', img_path, 'rggb', rggb.shape)
            
        wb = 1.0/wb 
        rggb_case = np.stack(rggb_l,0)    
        rggb_in = np.transpose(rggb_case,(0,3,1,2)) # N,C,H,W
        if DEBUG:
            print('rggb_in 0', len(rggb_l), rggb_in.shape)
        
        gt_data = scio.loadmat(gt_path)
        gt = gt_data['gt']
        gt = np.transpose(gt,(2,0,1))
         
        noise = np.stack(noise_l,2) 
        
        noise = noise[0,:,0]
        noise_p = (noise[0] + noise[2] + noise[4])/3.0
        noise_r = (noise[1] + noise[3] + noise[5])/3.0
        noise_map = (noise_p + noise_r)**0.5
        noise_map = np.tile(noise_map.reshape(1,1,1,1),[1,1,GT_size,GT_size]) if self.opt['pre_demosaic'] == False else np.tile(noise_map.reshape(1,1,1,1),[1,1,GT_size*2,GT_size*2])
        #LQ_size_tuple = (4,512,512)
        rggb_in = rggb_in[:,:,0:GT_size, 0:GT_size] if self.opt['pre_demosaic'] == False else rggb_in[:,:,0:GT_size*2, 0:GT_size*2]
        if self.opt['pre_demosaic'] == False:
            noise_in_unpro_s = np.stack([noise[0],noise[2],noise[2],noise[4]],0).reshape(1,4,1,1)
            noise_in_unpro_r = np.stack([noise[1],noise[3],noise[3],noise[5]],0).reshape(1,4,1,1)
        else:
            noise_in_unpro_s = np.stack([noise[0],noise[2],noise[4]],0).reshape(1,3,1,1)
            noise_in_unpro_r = np.stack([noise[1],noise[3],noise[5]],0).reshape(1,3,1,1)
        noise_in_unpro = rggb_in*noise_in_unpro_s + noise_in_unpro_r
        gt = gt[:,0:GT_size*2,0:GT_size*2]

        if DEBUG:
            print('noise', noise.shape, 'noise_l', noise_l[0].shape, 'rggb_in', rggb_in.shape)
            print('noise_map', noise_map.shape)
         
        img_LQs = torch.from_numpy(np.ascontiguousarray(rggb_in)).float()
        noise_in = torch.from_numpy(np.ascontiguousarray(noise_map)).float()
        gt = torch.from_numpy(np.ascontiguousarray(gt)).float()
        noise_in_unpro = torch.from_numpy(np.ascontiguousarray(noise_in_unpro)).float()

        return {'LQs':img_LQs,'noise_in':noise_in,'GT':gt,'folder':folder_path,'noise_in_unpro':noise_in_unpro,'wb':wb}

    def __len__(self):
        return len(self.subfolders_GT)

def bayer2bayer3d(image, inBayerType='rggb'):
        """
        convert bayer image to specified bayer 3D type
        :param image: input RGB image
        :param inBayerType: input Bayer image
        :return: 3D Bayer image [H x W x C], where C is channel for 'rggb'
        """

        assert(image.ndim == 2)
        assert(len(inBayerType) == 4)

        height, width = image.shape[:2]
        image = image[:height-height%4, :width-width%4]
        out = np.zeros((image.shape[0]//2, image.shape[1]//2, 4), dtype=image.dtype)

        if DEBUG:
            print('out.shape = ', out.shape)

        c = np.zeros(4, dtype=np.uint8)
        g = 1
        for i in range(4):
            if inBayerType[i] == 'R' or inBayerType[i] == 'r':
                c[0] = i
            elif inBayerType[i] == 'G' or inBayerType[i] == 'g':
                c[g] = i
                g += 1
            elif inBayerType[i] == 'B' or inBayerType[i] == 'b':
                c[3] = i

        out[:, :, c[0]] = image[::2,::2]
        out[:, :, c[1]] = image[::2, 1::2]
        out[:, :, c[2]] = image[1::2, ::2]
        out[:, :, c[3]] = image[1::2, 1::2]
        return out


class My_real(data.Dataset):
    def __init__(self, opt):
        super(My_real, self).__init__()
        self.opt = opt
        self.half_N_frames = opt['N_frames'] // 2
        # GS: No need LR
        #self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.GT_root = opt['dataroot_GT']
        self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': []}   ## They are flattened
        #### Generate data info and cache data
        
        # subfolders_LQ = util.glob_file_list(self.LQ_root)
        self.subfolders_GT = sorted(glob.glob(self.GT_root+'/*'))
    
    def fix_bayer_size(self, bayer):
        h, w = bayer.shape
        new_h = h - h%8
        new_w = w - w%8
        bayer = bayer[:new_h,:new_w]
        return bayer
        
    def __getitem__(self, index):  ## The index indicates the frame index which is flattened
        #GT_size = 384
        folder_path = self.subfolders_GT[index]
        img_paths = sorted(glob.glob(folder_path+'/*.tiff'))[0:self.opt['N_frames']]
        rggb_l = []
        noise_l = []
        sigma = 5 / 255.0
        for img_path in img_paths:
            if DEBUG:
                print('img_path', img_path)
            bayer = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) / 255.0
            if DEBUG:
                print('bayer', bayer.shape)
            bayer = self.fix_bayer_size(bayer)
            if DEBUG:
                print('fixed bayer', bayer.shape)
            rggb = bayer2bayer3d(bayer, 'grbg')
            noise = np.ones(rggb.shape[:2]) * sigma
            noise_l.append(noise)
            if self.opt['pre_demosaic'] == True:
                rggb = util.demosaic(rggb)
            rggb_l.append(rggb)
            
        rggb_case = np.stack(rggb_l,0)    
        rggb_in = np.transpose(rggb_case,(0,3,1,2)) # N,C,H,W
        
        # Fake gt_data
        gt_data = rggb
        gt = gt_data
        gt = np.transpose(gt,(2,0,1))
         
        noise = np.stack(noise_l,2) 
        noise_map = np.mean(noise)
        noise_map = np.tile(noise_map.reshape(1,1,1,1),[1,1,rggb.shape[0],rggb.shape[1]]) if self.opt['pre_demosaic'] == False else np.tile(noise_map.reshape(1,1,1,1),[1,1,rggb.shape[0]*2,rggb.shape[1]*2])
        #LQ_size_tuple = (4,512,512)
        rggb_in = rggb_in[:,:,0:rggb.shape[0], 0:rggb.shape[1]] if self.opt['pre_demosaic'] == False else rggb_in[:,:,0:rggb.shape[0]*2, 0:rggb.shape[1]*2]
        noise_in_unpro = noise_map
        #gt = gt[:,0:GT_size*2,0:GT_size*2]
        if DEBUG:
            print('noise', noise.shape, 'noise_l', noise_l[0].shape, 'rggb_in', rggb_in.shape)
            print('noise_map', noise_map.shape)
         
        img_LQs = torch.from_numpy(np.ascontiguousarray(rggb_in)).float()
        noise_in = torch.from_numpy(np.ascontiguousarray(noise_map)).float()
        gt = torch.from_numpy(np.ascontiguousarray(gt)).float()
        noise_in_unpro = torch.from_numpy(np.ascontiguousarray(noise_in_unpro)).float()

        return {'LQs':img_LQs,'noise_in':noise_in,'GT':gt,'folder':folder_path,'noise_in_unpro':noise_in_unpro}

    def __len__(self):
        return len(self.subfolders_GT)

class My_sim(data.Dataset):
    def __init__(self, opt):
        super(My_sim, self).__init__()
        self.opt = opt
        self.half_N_frames = opt['N_frames'] // 2
        # GS: No need LR
        #self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.GT_root = opt['dataroot_GT']
        self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': []}   ## They are flattened
        #### Generate data info and cache data
        
        # subfolders_LQ = util.glob_file_list(self.LQ_root)
        self.subfolders_GT = sorted(glob.glob(self.GT_root+'/*'))
    
    def fix_bayer_size(self, bayer):
        h, w = bayer.shape
        new_h = h - h%8
        new_w = w - w%8
        bayer = bayer[:new_h,:new_w]
        return bayer
        
    def __getitem__(self, index):  ## The index indicates the frame index which is flattened
        #GT_size = 384
        folder_path = self.subfolders_GT[index]
        img_paths = sorted(glob.glob(folder_path+'/*.tiff'))[0:self.opt['N_frames']]
        rggb_l = []
        noise_l = []
        sigma = 5 / 255.0
        for img_path in img_paths:
            if DEBUG:
                print('img_path', img_path)
            bayer = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) / 255.0
            if DEBUG:
                print('bayer', bayer.shape)
            bayer = self.fix_bayer_size(bayer)
            if DEBUG:
                print('fixed bayer', bayer.shape)
            rggb = bayer2bayer3d(bayer, 'grbg')
            noise = np.ones(rggb.shape[:2]) * sigma
            noise_l.append(noise)
            if self.opt['pre_demosaic'] == True:
                rggb = util.demosaic(rggb)
            rggb_l.append(rggb)
            
        rggb_case = np.stack(rggb_l,0)    
        rggb_in = np.transpose(rggb_case,(0,3,1,2)) # N,C,H,W
        
        # Fake gt_data
        gt_data = rggb
        gt = gt_data
        gt = np.transpose(gt,(2,0,1))
         
        noise = np.stack(noise_l,2) 
        noise_map = np.mean(noise)
        noise_map = np.tile(noise_map.reshape(1,1,1,1),[1,1,rggb.shape[0],rggb.shape[1]]) if self.opt['pre_demosaic'] == False else np.tile(noise_map.reshape(1,1,1,1),[1,1,rggb.shape[0]*2,rggb.shape[1]*2])
        #LQ_size_tuple = (4,512,512)
        rggb_in = rggb_in[:,:,0:rggb.shape[0], 0:rggb.shape[1]] if self.opt['pre_demosaic'] == False else rggb_in[:,:,0:rggb.shape[0]*2, 0:rggb.shape[1]*2]
        noise_in_unpro = noise_map
        #gt = gt[:,0:GT_size*2,0:GT_size*2]
        if DEBUG:
            print('noise', noise.shape, 'noise_l', noise_l[0].shape, 'rggb_in', rggb_in.shape)
            print('noise_map', noise_map.shape)
         
        img_LQs = torch.from_numpy(np.ascontiguousarray(rggb_in)).float()
        noise_in = torch.from_numpy(np.ascontiguousarray(noise_map)).float()
        gt = torch.from_numpy(np.ascontiguousarray(gt)).float()
        noise_in_unpro = torch.from_numpy(np.ascontiguousarray(noise_in_unpro)).float()

        return {'LQs':img_LQs,'noise_in':noise_in,'GT':gt,'folder':folder_path,'noise_in_unpro':noise_in_unpro}

    def __len__(self):
        return len(self.subfolders_GT)

