import numpy
import os
import scipy.io as sio
import h5py
import numpy as np
from scipy import misc
import pickle
from util import *
import sys
import cv2

class Pascal3dPlus:
    def __init__(self, category='car', source='imagenet', split='train', crop=True, rescale=224.0, first_n_debug=9999,
                 base_dir='/mnt/1TB_SSD/dataset/PASCAL3D+_release1.1',
                 sp_dir='/mnt/1TB_SSD/dataset/PASCAL3D+_sp',
                 detection_split_dir='/mnt/1TB_SSD/dataset/PASCAL3D+_sp/file_list',
                 imagenet_split_dir='/mnt/1TB_SSD/dataset/PASCAL3D+_release1.1/Image_sets',
                 pascal_split_dir='',
                 occ_dir='/mnt/1TB_SSD/dataset/PASCAL3D+_occ/occ',
                 black_dir='/mnt/4TB_b/qing/dataset/PASCAL3D+_black'):
        self.base_dir = base_dir
        self.sp_dir = sp_dir
        self.detection_split_dir = detection_split_dir
        if split[0:3]=='occ':
            self.detection_split_dir = occ_dir
            
        self.imagenet_split_dir = imagenet_split_dir
        self.pascal_split_dir = pascal_split_dir

        self.category = category
        self.source = source
        self.split = split
        if split[0:3]=='occ':
            self.split='occ'
            self.occ_level = split.split('_')[1]
        elif split[0:5]=='black':
            self.split='black'
            self.black_level = split.split('_')[1]
            
        self.crop = crop
        self.rescale = rescale
        self.first_n_debug = first_n_debug  # TODO: dirty truncation for fast debugging should be deleted

        self.image_dir = os.path.join(self.base_dir, 'Images', category + '_' + source)
        if self.split=='occ':
            self.image_dir = os.path.join(occ_dir, category + 'LEVEL' + self.occ_level)
        elif self.split=='black':
            self.image_dir = os.path.join(black_dir, category, 'gray_img_30_300_loc2', 'patch_'+self.black_level)
            if self.black_level=='0':
                self.image_dir = os.path.join(black_dir, category, 'gray_img_30_300', 'patch_1')
            
        self.classification_dir = os.path.join(self.base_dir, 'Annotations', category + '_' + source)
        self.detection_dir = os.path.join(sp_dir, category + '_' + source, 'transfered')
        
        print('inited Pascal3D+:\n    category={}\n    split={}\n    crop={}'.format(self.category, split, self.crop))

    def get_classification_instances(self):
        return_list = []
        #  should be reading lists
        for file in os.listdir(self.image_dir):
            dot = file.find('.')
            img_name = file[:dot]
            return_list.append([img_name, -1])
        return return_list

    def get_classification_images(self):
        instances = self.get_classification_instances()
        return self.get_images_from_instances(instances)

    def get_classification_all(self):
        raise NotImplementedError

    def get_detection_instances(self):
        file_list = os.path.join(self.detection_split_dir, self.split + '_list', self.category + '_' + self.split + '.txt')
        if self.split == 'black':
            file_list = os.path.join(self.detection_split_dir, 'test_list', self.category + '_test.txt')
            
        return_list = []

        with open(file_list, 'r') as file:
            for line in file:
                if len(return_list) == self.first_n_debug:
                    break
                if line == '\n':
                    continue
                segments = line.split(' ')
                img_name = segments[0]
                instance_id = int(segments[1][:-1]) - 1  # matlab to python
                return_list.append([img_name, instance_id])
        return return_list

    def get_detection_all(self, target_len=None):
        return_list = []
        instances = self.get_detection_images(target_len)
        # max_spid = 0
        for (inst_i,(img_name, instance_id, bbox, delta_xy, delta_scale, img, ins)) in enumerate(instances):
            zhishuai_bbox = bound_bbox_to_int(img, bbox)
            zhishuai_shift = np.array([zhishuai_bbox[0], zhishuai_bbox[1], zhishuai_bbox[0], zhishuai_bbox[1]])
            zhishuai_scale = bbox_to_delta_scale(zhishuai_bbox, 224.0)

            this_shift = np.array([delta_xy[0], delta_xy[1], delta_xy[0], delta_xy[1]])
            this_scale = delta_scale

            mat_name = os.path.join(self.detection_dir, img_name + '.mat')
            anno = sio.loadmat(mat_name)['anno']
            sp_anno = anno[instance_id][1]
            sp_list = []
            sp_id = -1
            for sp in sp_anno:
                sp_id += 1
                    
                for sp_bbox in sp[0]:
                    sp_bbox -= 1
                    original = sp_bbox[0:4] / zhishuai_scale + zhishuai_shift
                    this = (original - this_shift) * this_scale
                    try:
                        assert(sp_bbox[8] == sp_id)
                    except:
                        # print(sp_bbox, sp_id)
                        continue
                        
                    sp_list.append([sp_id, this])
                    
            if len(sp_list) == 0:
                continue
                
            bbox = (bbox - this_shift) * this_scale
            return_list.append([img_name, instance_id, bbox, delta_xy, delta_scale, img, ins, sp_list])
        
        return return_list
    

    def get_detection_images(self, target_len):
        instances = self.get_detection_instances()
        return self.get_images_from_instances(instances, target_len)

    def get_images_from_instances(self, instances, target_len):
        return_list = []
        for (inst_i, (img_name, instance_id)) in enumerate(instances):
            if self.split == 'occ':
                img_path = os.path.join(self.image_dir, '{}_{}.mat'.format(img_name, instance_id+1))
                f = h5py.File(img_path)
                img = np.array(f['record']['img']).T # RGB
            else:
                img_path = os.path.join(self.image_dir, img_name + '.JPEG')
                try:
                    img = cv2.imread(img_path)[:,:,::-1]
                except:
                    continue
                    
                img = im2rgb(img)
            
            if self.split == 'black' and self.black_level=='0':
                RGB_mean = np.array([[[123.675, 116.28, 103.53]]])
                img = np.tile(RGB_mean, (img.shape[0], img.shape[1], 1))
                
            mat_full_name = os.path.join(self.classification_dir, img_name + '.mat')
            record = sio.loadmat(mat_full_name, )['record']
            if not instance_id == -1:  # specifying instance_id
                category_name = record[0, 0]['objects'][0, instance_id]['class'][0]
                assert(category_name == self.category)
                bbox = record[0, 0]['objects'][0, instance_id]['bbox'][0] - 1
                if target_len is None:
                    ins, delta_xy, delta_scale = bbox_crop_and_bbox_rescale(img, bbox, self.rescale, self.crop)
                    return_list.append([img_name, instance_id, bbox, delta_xy, delta_scale, img, ins])
                else:
                    bbox_ = np.array([0, 0, img.shape[1]-1, img.shape[0]-1])
                    if target_len == 'area':
                        ins, delta_xy, delta_scale = area_rescale(img, 300.0)
                    elif type(target_len) is list:
                        ins, delta_xy, delta_scale = bbox_crop_and_bbox_rescale(img, bbox_, target_len[inst_i], self.crop)
                    else:
                        ins, delta_xy, delta_scale = bbox_crop_and_bbox_rescale(img, bbox_, target_len, self.crop)
                        
                    return_list.append([img_name, instance_id, bbox, delta_xy, delta_scale, img, ins])
                    
            else:  # not specifying instance_id
                for real_instance_id in range(len(record[0, 0]['objects'][0])):
                    category_name = record[0, 0]['objects'][0, real_instance_id]['class'][0]
                    if not category_name == self.category:
                        continue
                    bbox = record[0, 0]['objects'][0, real_instance_id]['bbox'][0] - 1
                    ins, delta_xy, delta_scale = bbox_crop_and_bbox_rescale(img, bbox, self.rescale, self.crop)
                    return_list.append([img_name, real_instance_id, bbox, delta_xy, delta_scale, img, ins])
        return return_list
