from eval.eval_AP import eval_AP
import os, glob, pickle
import numpy as np
from Pascal3dPlus import Pascal3dPlus

root_dir = '/home/yutong/SPMatch/vp_test_sedan/'
dir_ls = glob.glob(os.path.join(root_dir,'a*e*'))
sp_num = 39
category = 'car'
set_type = 'test'
crop = True
sp_detection = []
img_names = []
img_size = []

img_id = 0
for img_dir in dir_ls:
    if not 'a20e0' in img_dir:
        continue
        
    filelist = os.path.join(img_dir, 'file_list.txt')
    with open(filelist, 'r') as fh:
        contents = fh.readlines()
        
    img_list = [cc.strip() for cc in contents][0:-1]
    
    for img_ii in range(len(img_list)):
        sp_detection.append([])
        
        img_path = img_list[img_ii]
        img_name = img_path.split('/')[-1].split('.')[0]
        img_names.append(img_name)
        
        sp_file = os.path.join(img_dir, 'img{}VSimg{}_transSP.pickle'.format(img_ii, len(img_list)))
        sp_info = pickle.load(open(sp_file, 'rb'))
        if len(sp_info)==0:
            continue
            
        assert(sp_info[0][0] == img_name)
        for spi in range(sp_num):
            locs = np.array([ss[2] for ss in sp_info if ss[1]==spi])
            scores = np.array([ss[3] for ss in sp_info if ss[1]==spi])
            
            if len(locs)==0:
                sp_detection[-1].append(np.zeros((0,6)))
                continue
            
            c_list = locs[:,0]
            r_list = locs[:,1]
            
            bb_loc = np.column_stack((c_list - 49.5, r_list - 49.5, c_list + 49.5, r_list + 49.5, scores))
            bb_loc = np.concatenate((np.ones((bb_loc.shape[0], 1)) * img_id, bb_loc), axis=1)
            
            sp_detection[-1].append(bb_loc)
            
        img_id += 1
        
print('total number of testing images: {}'.format(len(sp_detection)))
# read in ground truth from Pascal 3D+
pascal = Pascal3dPlus(category=category, split=set_type, crop=crop, first_n_debug=9999)
detection_all = pascal.get_detection_all()

spanno = []

img_names2 = [dd[0] for dd in detection_all]

for img_name in img_names:
    img_id2 = img_names2.index(img_name)
    
    img_name, instance_id, bbox, delta_xy, delta_scale, img, ins, sp_list = detection_all[img_id2]
    
    spanno.append([[] for _ in range(sp_num)])
    for [sp_id, this] in sp_list:
        if sp_id < sp_num:
            bb_o = this + 1   # also to Matlab
            bb_o = np.array([max(np.ceil(bb_o[0]), 1), max(np.ceil(bb_o[1]), 1),
                             min(np.floor(bb_o[2]), ins.shape[1]), min(np.floor(bb_o[3]), ins.shape[0])])
            spanno[-1][sp_id].append(bb_o)

    for i in range(sp_num):
        if len(spanno[-1][i]) == 0:
            spanno[-1][i] = np.array([])
        else:
            spanno[-1][i] = np.array(spanno[-1][i])
    img_size.append(ins.shape[0:2])
    
eval_AP(sp_detection, spanno, img_size)
        