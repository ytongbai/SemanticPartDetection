import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.io as sio
import os,cv2
from Pascal3dPlus import Pascal3dPlus

category='car'
set_type='train'
dataset_root = '/mnt/1TB_SSD/dataset/'
list_dir = os.path.join(dataset_root, 'PASCAL3D+_release1.1','Image_sets')
anno_dir = os.path.join(dataset_root, 'PASCAL3D+_release1.1','Annotations', '{}_imagenet'.format(category))


split_dir='/mnt/1TB_SSD/dataset/PASCAL3D+_sp/file_list'
filelist = os.path.join(split_dir, set_type + '_list', category + '_' + set_type + '.txt')

with open(filelist, 'r') as fh:
    contents = fh.readlines()
    
file_names = [cc.strip().split()[0] for cc in contents if cc != '\n']
instance_id = [int(cc.strip().split()[1])-1 for cc in contents if cc != '\n']

pascal = Pascal3dPlus(category=category, split=set_type, crop=True, first_n_debug=9999)
detection_all = pascal.get_detection_all()
print(len(detection_all))

fname_set = [tmp[0] for tmp in detection_all]

vp_list = []
subtype_list = []
fname_index = 0
for fi, ff in enumerate(file_names):
    if ff not in fname_set:
        continue
    
    assert(ff == fname_set[fname_index])
    fname_index += 1
    
    mat_file = os.path.join(anno_dir, '{}.mat'.format(ff))
    assert(os.path.isfile(mat_file))
    mat_contents = sio.loadmat(mat_file)
    record = mat_contents['record']
    objects = record['objects']
    azi = objects[0,0]['viewpoint'][0,instance_id[fi]]['azimuth_coarse'][0,0][0,0]
    ele = objects[0,0]['viewpoint'][0,instance_id[fi]]['elevation_coarse'][0,0][0,0]
    subtype = objects[0,0]['subtype'][0,instance_id[fi]][0]
    vp_list.append((azi, ele))
    subtype_list.append(subtype)
    
print('done')

for target_vp in [0,45,90,135,180,225,270,315]:
    target_range = [target_vp-5, target_vp+5]
    image_name_list = []
    instance_info_list = []
    sp_info_list = []
    vp_set_list = []

    save_img_dir = os.path.join('/mnt/4TB_b/qing/SPMatch', 'vp_examples_less', str(target_vp))
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)

    print(save_img_dir)
    for fi, ff in enumerate(fname_set):
        if subtype_list[fi] == 'sedan':
            vp_curr = vp_list[fi]
            if len(image_name_list) < 4 and vp_curr[1]==0 and vp_curr[0]<=target_range[1] and vp_curr[0]>=target_range[0]:
                img_name, instance_id, bbox, delta_xy, delta_scale, img, ins, sp_list = detection_all[fi]
                assert(ff == img_name)
                image_name_list.append(ff)
                instance_info_list.append(instance_id)
                sp_info_list.append(sp_list)
                vp_set_list.append(vp_curr)
                cv2.imwrite(os.path.join(save_img_dir, '{}.JPEG'.format(ff)), ins[:,:,::-1])


    print(len(image_name_list))

    save_info_file = os.path.join(save_img_dir, 'info.pickle')
    with open(save_info_file, 'wb') as fh:
        pickle.dump([image_name_list, instance_info_list, sp_info_list, vp_set_list], fh)