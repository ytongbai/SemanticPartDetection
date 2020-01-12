import os, json
import d3, vdb
from vertex_id_picker import *
import numpy as np
import imageio
import matplotlib.pyplot as plt
import cv2


db_root = '/home/yutong/SPMatch/from_weichao/DenseMatching_CarOnly'
obj_name = 'Sedan_4Door_Sedan4Door_LOD0_11'
cam_name = 'sedan4door'

for frame_id in range(648):

    src_lit_filename, src_cam_pose, src_depth_filename, src_vertexs_3d, src_bbox  =  \
    get_frame_info(db_root, obj_name, cam_name, frame_id)

    img = cv2.imread(src_lit_filename)
    img_cropped = img[src_bbox[2]-3:src_bbox[3]+4, src_bbox[0]-3:src_bbox[1]+4]

    save_dir = os.path.join(db_root,cam_name,'lit_cropped')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_file = os.path.join(save_dir,'%08d.png' % frame_id)
    cv2.imwrite(save_file, img_cropped)