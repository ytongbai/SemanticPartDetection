import numpy as np
import os
import pickle
from scipy.spatial.distance import cdist
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.io as sio

def process_pool3_patch(lowpool_1, lowpool_2, row1, col1, row2, col2):
    pat1_core = lowpool_1[row1-1:row1+3,col1-1:col1+3].reshape(-1,256)
    pat2_core = lowpool_2[row2-1:row2+3,col2-1:col2+3].reshape(-1,256)
    cor_dist = cdist(pat1_core, pat2_core, 'cosine')
    
    sort_idx = np.argsort(cor_dist,axis=None)
    pat_dist_ls = []
    for ii in sort_idx[0:3]:
        pat1_idx_core, pat2_idx_core = np.unravel_index(ii, cor_dist.shape)
        pat1_r_core, pat1_c_core = np.unravel_index(pat1_idx_core, (4,4))
        pat2_r_core, pat2_c_core = np.unravel_index(pat2_idx_core, (4,4))
        pat1_r = pat1_r_core + (row1-1)
        pat1_c = pat1_c_core + (col1-1)
        pat2_r = pat2_r_core + (row2-1)
        pat2_c = pat2_c_core + (col2-1)
        
        rr1 = np.max((0,pat1_r-3))
        rr2 = np.max((0,pat2_r-3))
        cc1 = np.max((0,pat1_c-3))
        cc2 = np.max((0,pat2_c-3))
        
        # 6 by 6 context
        pat1 = lowpool_1[rr1:pat1_r+3, cc1:pat1_c+3]
        pat2 = lowpool_2[rr2:pat2_r+3, cc2:pat2_c+3]
        
        pat_dist = np.mean(cdist(pat1.reshape(-1,256), pat2.reshape(-1,256), 'cosine'),axis=None)
        pat_dist_ls.append(pat_dist)
        
    pat1_idx_core, pat2_idx_core = np.unravel_index(sort_idx[np.argmin(pat_dist_ls)], cor_dist.shape)
    pat1_r_core, pat1_c_core = np.unravel_index(pat1_idx_core, (4,4))
    pat2_r_core, pat2_c_core = np.unravel_index(pat2_idx_core, (4,4))
    
    pat1_r = pat1_r_core + (row1-1)
    pat1_c = pat1_c_core + (col1-1)
    pat2_r = pat2_r_core + (row2-1)
    pat2_c = pat2_c_core + (col2-1)
    
    return(pat1_r, pat1_c, pat2_r, pat2_c, 1-np.min(pat_dist_ls))
    

category = 'car'
set_type = 'train'
featDim = 256
offset = 2
dataset_root = '/home/yutong/dataset/'
list_dir = os.path.join(dataset_root, 'PASCAL3D+_release1.1','Image_sets')
proj_root ='/mnt/4TB_b/qing/VC_journal/'
cache_dir = os.path.join(proj_root, 'feat')

pool4_offset=2

root_dir = '/home/yutong/SPMatch/vp_examples_less/'
for vp in [0,45,90,135,180,225,270,315]:
# for vp in [45]:
    img_dir = os.path.join(root_dir, str(vp))
    # file list
    filelist = os.path.join(img_dir, 'file_list.txt')
    with open(filelist, 'r') as fh:
        contents = fh.readlines()
        
    img_list = [cc.strip() for cc in contents]
    img_idx2 = len(img_list)-1
    
    for img_idx1 in range(img_idx2):

        # load pool4 match result
        mat_filename = os.path.join(img_dir, 'img{}VSimg{}.mat'.format(img_idx1, img_idx2))
        mat_content = sio.loadmat(mat_filename)
        pool4_rst = mat_content['rst']

        filelist = os.path.join(img_dir, 'file_list.txt')
        with open(filelist, 'r') as fh:
            contents = fh.readlines()

        img_list = [cc.strip() for cc in contents]
        img_file1 = img_list[img_idx1]
        img_file2 = img_list[img_idx2]
        pool3_file1 = '.'.join(img_file1.split('.')[0:-1])+'_pool3.pickle'
        pool3_file2 = '.'.join(img_file2.split('.')[0:-1])+'_pool3.pickle'

        pool3_img1 = pickle.load(open(pool3_file1, 'rb'))
        pool3_img2 = pickle.load(open(pool3_file2, 'rb'))
        img1 = cv2.imread(img_file1)
        img2 = cv2.imread(img_file2)
        rratio = 224.0/np.min(img2.shape[0:2])
        img2 = cv2.resize(img2, (0,0), fx=rratio, fy=rratio) 

        save_path = mat_filename.replace('.mat','_refine/')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # to save pool4 and pool3 matched x1,y1,x1',y1'
        matched = [[],[]]

        for pi in range(pool4_rst.shape[0]):
            pool4_info = pool4_rst[pi]
            p4_score, img1_id, row1, col1, img2_id, row2, col2 = pool4_info.astype(int)

            pool3_row1 = (row1+pool4_offset-1)*2
            pool3_col1 = (col1+pool4_offset-1)*2
            pool3_row2 = (row2+pool4_offset-1)*2
            pool3_col2 = (col2+pool4_offset-1)*2

            # plot images
            plt.close()
            fig,ax = plt.subplots(1,2,figsize=(20,5))

            # Display the image
            ax[0].imshow(img1[:,:,::-1])
            ax[1].imshow(img2[:,:,::-1])

            pool4_matched = []
            # Create a Rectangle patch
            x_pool4 = 16*(col1+pool4_offset-1)-42
            y_pool4 = 16*(row1+pool4_offset-1)-42
            rect = patches.Rectangle((x_pool4,y_pool4),100,100,linewidth=1,edgecolor='r',facecolor='none')
            circ = patches.Circle((x_pool4+50,y_pool4+50),3)
            pool4_matched = pool4_matched + [x_pool4+50,y_pool4+50]
            ax[0].add_patch(rect)
            ax[0].add_patch(circ)

            x_pool4 = 16*(col2+pool4_offset-1)-42
            y_pool4 = 16*(row2+pool4_offset-1)-42
            rect = patches.Rectangle((x_pool4,y_pool4),100,100,linewidth=1,edgecolor='r',facecolor='none')
            circ = patches.Circle((x_pool4+50,y_pool4+50),3)
            pool4_matched = pool4_matched + [x_pool4+50,y_pool4+50]
            ax[1].add_patch(rect)
            ax[1].add_patch(circ)

            pool4_matched = pool4_matched + [p4_score]
            matched[0].append(pool4_matched)

            pool3_matched = []
            # refine the matches on pool3
            pat1_r, pat1_c, pat2_r, pat2_c, p3_score = process_pool3_patch(pool3_img1,pool3_img2, pool3_row1, pool3_col1, pool3_row2, pool3_col2)

            # Create a Rectangle patch -- pool3
            x_pool3 = 8*(pat1_c)-18
            y_pool3 = 8*(pat1_r)-18
            rect = patches.Rectangle((x_pool3,y_pool3),44,44,linewidth=1,edgecolor='r',facecolor='none')
            circ = patches.Circle((x_pool3+22,y_pool3+22),3, edgecolor='g',facecolor='g')
            pool3_matched = pool3_matched + [x_pool3+22,y_pool3+22]
            ax[0].add_patch(rect)
            ax[0].add_patch(circ)

            x_pool3 = 8*(pat2_c)-18
            y_pool3 = 8*(pat2_r)-18
            rect = patches.Rectangle((x_pool3,y_pool3),44,44,linewidth=1,edgecolor='r',facecolor='none')
            circ = patches.Circle((x_pool3+22,y_pool3+22),3, edgecolor='g',facecolor='g')
            pool3_matched = pool3_matched + [x_pool3+22,y_pool3+22]
            ax[1].add_patch(rect)
            ax[1].add_patch(circ)
            
            pool3_matched = pool3_matched + [p3_score]
            matched[1].append(pool3_matched)

            plt.savefig(os.path.join(save_path, '{}.png'.format(str(pi))))

        print(matched)
        save_pickle = mat_filename.replace('.mat','_matched.pickle')
        with open(save_pickle,'wb') as fh:
            pickle.dump(matched, fh)