import os
import math
import random
import logging,time
from typing_extensions import runtime
import data.util as datautil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
import cv2
import numpy as np
import argparse
import time

DEBUG = False

def main():
    parser = argparse.ArgumentParser(description='Process Arguments.')
    parser.add_argument('--opt_file', type=str, default='options/test/test_BDNet.yml', 
                    help='path for option file')
    args = parser.parse_args()
    #### options
    opt = args.opt_file 
    opt = option.parse(opt, is_train=True)
    log_dir = '../tb_logger/test_' + opt['name']
    #### distributed training settings
    opt['dist'] = False
    print('Disabled distributed training.')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase.startswith('val'):
            # pdb.set_trace()
            print('Test: phase', phase)
            if DEBUG:
                print('dataset_opt', dataset_opt)
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)

            #### create model
            model = create_model(opt)

            psnr_rlt = {}  # with border and center frames
            psnr_total_avg = 0.
            ssim_rlt = {}  # with border and center frames
            ssim_total_avg = 0.
            save_path = "%s/%s"%(log_dir, dataset_opt['name'])
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            is_test_gt = opt['is_test_gt']
            
            total_time = 0.0
            for idx, val_data in enumerate(val_loader):
                folder = val_data['folder'][0]
                model.feed_data(val_data,need_GT=is_test_gt)
                starttime = time.time()
                model.test(flag='real')
                total_time += time.time() - starttime
                visuals = model.get_current_visuals(need_GT=is_test_gt)
                lq_img = visuals['LQ'][2,:,:,:].permute(1,2,0).numpy()
                lq_img = datautil.demosaic(lq_img)
                rlt_img = util.tensor2img(visuals['rlt'],out_type=np.float32)  # uint8
                gt_img = util.tensor2img(visuals['GT'],out_type=np.float32) if is_test_gt == True else None # uint8

                if DEBUG:
                    if is_test_gt:
                        print('lq_img', lq_img.shape, 'rlt_img', rlt_img.shape, 'gt_img', gt_img.shape)
                    else:
                        print('lq_img', lq_img.shape, 'rlt_img', rlt_img.shape, 'folder', folder)
                
                #out_img = np.concatenate([lq_img,rlt_img,gt_img],1) if is_test_gt == True else np.concatenate([lq_img,rlt_img],1) # uint8
                out_img = rlt_img
                #wb = val_data['wb'][0].unsqueeze(0).numpy()
                #out_img = out_img*wb
                #out_img = (np.clip(out_img,0.0,1.0-1e-4)+1e-4)**(1.0/2.2)
                out_img = (np.clip(out_img,0.0,1.0-1e-4)+1e-4)
                out_img = np.uint8(out_img*255.0)
                
                filename = os.path.basename(folder)
                cv2.imwrite("%s/%s.png"%(save_path,filename),
                    out_img[:,:,::-1],[int(cv2.IMWRITE_JPEG_QUALITY),100])
                
                # calculate PSNR
                if is_test_gt == True:
                    psnr = util.calculate_psnr(np.uint8(np.clip(rlt_img,0.0,1.0)*255.0), np.uint8(np.clip(gt_img,0.0,1.0)*255.0))
                    if math.isinf(psnr) == False:
                        psnr_rlt[folder] = psnr
                        print('idx = %04d, psnr = %.4f'%(idx,psnr))
                    ssim = util.calculate_ssim(np.uint8(np.clip(rlt_img,0.0,1.0)*255.0), np.uint8(np.clip(gt_img,0.0,1.0)*255.0))
                    if math.isinf(ssim) == False:
                        ssim_rlt[folder] = ssim
            
            print('total test time %.3f seconds' % total_time, 'average test time %.3f seconds' % (total_time / len(val_loader)))
                #pbar.update('Test {}_psnr={}'.format(folder,psnr))
            if is_test_gt == True and len(psnr_rlt) > 0:
                for k, v in psnr_rlt.items():
                    psnr_total_avg += v
                for k, v in ssim_rlt.items():
                    ssim_total_avg += v
                psnr_total_avg /= len(psnr_rlt)
                ssim_total_avg /= len(ssim_rlt)
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    target=open("%s/score.txt"%(save_path),'w')
                    target.write("psnr=%.4f\n"%(psnr_total_avg))
                    target.write("ssim=%.4f\n"%(ssim_total_avg))
                    target.close()

        

    

if __name__ == '__main__':
    main()
