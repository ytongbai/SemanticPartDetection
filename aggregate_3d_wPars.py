import os, json, pickle
import d3, vdb
from vertex_id_picker import *
import numpy as np
import imageio
import matplotlib.pyplot as plt
import cv2
import glob
from scipy.spatial.distance import pdist,squareform

def cluster_3d(pnts, thres=1, cn=None):
    # pnts=[(id1,np.array([x1,y1,z1])),...,(idn,np.array([xn,yn,zn]))]
    groups = []
    centers = []
    
    groups.append([pnts[0]])
    centers.append(pnts[0][1])
    
    for pp in pnts[1:]:
        pp_ci = []
        for ci,cc in enumerate(centers):
            cc_dist = np.linalg.norm(pp[1]-cc)
            if cc_dist < thres:
                pp_ci.append(ci)
                
        if len(pp_ci)>0:
            new_gg = [groups[ci] for ci in range(len(centers)) if ci in pp_ci]
            new_gg = sum(new_gg, [])
            new_gg += [pp]
            new_c = np.mean(np.array([pgg[1] for pgg in new_gg]), axis=0)

            groups = [groups[ci] for ci in range(len(centers)) if ci not in pp_ci]
            centers = [centers[ci] for ci in range(len(centers)) if ci not in pp_ci]
            groups.append(new_gg)
            centers.append(new_c)
        else:
            groups.append([pp])
            centers.append(pp[1])
            
    group_size = [len(gg) for gg in groups]
    if cn is None:
        size_thres = len(pnts)/2
        selected = np.where(np.array(group_size)>size_thres)[0]
    else:
        selected = np.argsort(-np.array(group_size))[0:cn]
        
    rst_ids = []
    for ss in selected:
        ss_g = groups[ss]
        ss_c = centers[ss]
        ss_dist = [np.linalg.norm(pp[1]-ss_c) for pp in ss_g]
        ss_id = ss_g[np.argmin(ss_dist)][0]
        rst_ids.append(ss_id)
        
    return rst_ids


db_root = '/home/yutong/SPMatch/from_weichao/DenseMatching_CarOnly'
obj_name = 'Sedan_4Door_Sedan4Door_LOD0_11'
cam_name = 'sedan4door'

root_dir = '/home/yutong/SPMatch/vp_examples/'
fid_vp_ls = [(144,90),(153,45),(162,0),(171,315),(180,270),(189,225),(198,180),(207,135)]

test_dir = '/home/yutong/SPMatch/vp_test_sedan/'
dir_ls = glob.glob(os.path.join(test_dir,'a*e*'))

total_spnum=39

'''
[(0,4,10),(1,4,10),(2,4,10),(3,4,10),(4,4,10),(5,4,10),(6,4,10),(7,4,10,*20),(8,4,10),\
(9,4,20),(10,2,10),(11,4,40,*5),(12,2,15),(13,4,15,*1-10), (14,2,10,*1),(15,4,10),(16,2,10),\
(20,1,10),(23,1,10,*20),(24,1,10),(25,1,10,*???), (27,1,10,*20)\]
'''

# sp_pars = [(0,4,10),(1,4,10),(2,4,10),(3,4,10),(4,4,10),(5,4,10),(6,4,10),(7,4,10),(8,4,10),(9,4,20),(10,2,10)]

# This is some parameters for aggragation. Varys using different images.
sp_par_file = '/home/yutong/SPMatch/aggregate_3d_sp_pars_32train.txt'
with open(sp_par_file,'r') as fh:
    contents = fh.readlines()
    
sp_pars = [cc.strip().split() for cc in contents]
sp_pars = [[int(sppp) for sppp in spp[:-1]]+[spp[-1]] for spp in sp_pars]

# gather training data
input_ls = [[] for _ in range(total_spnum)]
for (frame_id,vp) in fid_vp_ls:
    src_lit_filename, src_cam_pose, src_depth_filename, src_vertexs_3d, src_bbox  =  \
    get_frame_info(db_root, obj_name, cam_name, frame_id)

    img = cv2.imread(src_lit_filename)
    img_cropped = img[src_bbox[2]-3:src_bbox[3]+4, src_bbox[0]-3:src_bbox[1]+4]

    ratio = 224/np.min(img_cropped.shape[0:2])

    img_dir = os.path.join(root_dir,str(vp))

    filelist = os.path.join(img_dir, 'file_list.txt')
    with open(filelist, 'r') as fh:
        contents = fh.readlines()

    img_list = [cc.strip() for cc in contents]
    img_idx2 = len(img_list)-1

    for img_idx1 in range(img_idx2):
        sp_transfer1 = os.path.join(img_dir, 'img{}VSimg{}_transSP.pickle'.format(img_idx1, img_idx2))
        predict_sp_info = pickle.load(open(sp_transfer1,'rb'))
        
        for tar_sp in range(total_spnum):
            sp_ls = [pp[1] for pp in predict_sp_info if pp[0]==tar_sp]
            for spp in sp_ls:
                input_vertex = np.array(spp)/ratio + np.array([src_bbox[0]-3, src_bbox[2]-3])

                vertex_id, matched_2d = get_vertex_id(input_vertex, src_vertexs_3d, src_cam_pose, src_depth_filename)
                input_ls[tar_sp].append((vertex_id, np.array(src_vertexs_3d[vertex_id])))

# run clustering algorithm
sp_3d_id = []
for sp_par in sp_pars:
    tar_sp, tar_cn, cpar = sp_par[0:3]
    print('total number of transfered sp{}: {}'.format(tar_sp, len(input_ls[tar_sp])))
    merged_id = cluster_3d(input_ls[tar_sp], cpar, tar_cn)
    sp_3d_id.append(merged_id)

# put back to each 2d images used in testing

for dd in dir_ls:        
    info = dd.split('/')[-1]
    i1 = info.index('a')
    i2 = info.index('e')
    azi_s = int(info[i1+1:i2])
    ele = int(info[i2+1:])
    
    for azi in [azi_s,azi_s+5,azi_s+10]:
        azi = azi%360
        if azi > 90:
            j1 = (90 - (azi-360))//5
        else:
            j1 = (90 - azi)//5

        j2 = ele//5+1
        frame_id = j2*72+j1

        src_lit_filename, src_cam_pose, src_depth_filename, src_vertexs_3d, src_bbox  =  \
        get_frame_info(db_root, obj_name, cam_name, frame_id)
        depth = np.load(src_depth_filename)


        img = cv2.imread(src_lit_filename)
        img_cropped = img[src_bbox[2]-3:src_bbox[3]+4, src_bbox[0]-3:src_bbox[1]+4]
        ratio = 224/np.min(img_cropped.shape[0:2])
        img_cropped = cv2.resize(img_cropped, (0,0), fx=ratio, fy=ratio) 

        sp_info = []
        plt.close()
        fig,ax = plt.subplots(1,1,figsize=(20,10))
        ax.imshow(img_cropped[:,:,::-1])

        for sp_par in sp_pars:

            tar_sp = sp_par[0]
            dep_thrh = sp_par[3]
            if len(sp_par)>4:
                vp_thrh = sp_par[4].split(',')
                vp_thrh[0] = vp_thrh[0][1:]
                vp_thrh[-1] = vp_thrh[-1][:-1]
                vp_thrh = [int(vpt) for vpt in vp_thrh ]
            else:
                vp_thrh = None
            candidates = []

            for mi in sp_3d_id[tar_sp]:
                matched_2d = src_cam_pose.project_to_cam_space(src_vertexs_3d[mi:mi+1,:])
                ds = depth[(int(matched_2d[:,1]), int(matched_2d[:,0]))]

                if abs(matched_2d[0,2] - ds) < dep_thrh:
                    if not vp_thrh is None:
                        in_range = False
                        for vpti in range(len(vp_thrh)//2):
                            if (azi >= vp_thrh[2*vpti] and azi <= vp_thrh[2*vpti+1]):
                                in_range = True

                        if not in_range:
                            continue


                    matched_2d_shift = (matched_2d[0,0:2].astype(int) - np.array([src_bbox[0]-3, src_bbox[2]-3]))*ratio
                    candidates.append(matched_2d_shift)
                    
            if len(candidates)>1:
                candidates_new = []
                c_pdist = squareform(pdist(np.array(candidates)))
                for ci in range(len(candidates)):
                    ci_f = True
                    for cj in range(len(candidates)):
                        if c_pdist[ci,cj]<30:
                            ci_f = False
                            if cj>ci:
                                candidates_new.append((candidates[ci]+candidates[cj])/2)
                            
                    if ci_f:
                        candidates_new.append(candidates[ci])          
                
                candidates = candidates_new
                
            for cc in candidates:
                sp_info.append([tar_sp, cc])
                ax.plot(int(cc[0]), int(cc[1]), 'b*')

        plt.savefig(os.path.join(dd, 'sp_anno_%08d.png' % frame_id))
        save_file = os.path.join(dd, 'sp_info_%08d.pickle' % frame_id)
        pickle.dump(sp_info, open(save_file,'wb'))