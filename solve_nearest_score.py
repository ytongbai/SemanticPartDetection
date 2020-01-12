import pickle
import os,sys,glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d


def find_nearest(pnt1, arr, num=1):
    dist = (np.sum((pnt1.reshape(1,-1)-arr[:,0:4])**2,axis=1))**0.5
    return np.argsort(dist)[0:num]


def predict2(pnt1, matched_arr, width1, height1, width2, height2):
    vertices = np.array([[0,0,0,0],\
                [width1,0,width2,0],\
                [0,height1,0,height2],\
                [width1,height1,width2,height2]])
    matched = matched_arr
    
    nn_ls = []
    
    matched_1 = matched[:,0:2]
    field1 = np.where(np.logical_and(matched_1[:,0]<=pnt1[0], matched_1[:,1]<=pnt1[1]))[0]
    if len(field1)>0:
        nn1 = field1[find_nearest(pnt1, matched_1[field1], 1)]
        nn_ls = np.concatenate((nn_ls,nn1))
    
    field2 = np.where(np.logical_and(matched_1[:,0]>=pnt1[0], matched_1[:,1]<=pnt1[1]))[0]
    if len(field2)>0:
        nn2 = field2[find_nearest(pnt1, matched_1[field2], 1)]
        nn_ls = np.concatenate((nn_ls,nn2))
    
    
    field3 = np.where(np.logical_and(matched_1[:,0]<=pnt1[0], matched_1[:,1]>=pnt1[1]))[0]
    if len(field3)>0:
        nn3 = field3[find_nearest(pnt1, matched_1[field3], 1)]
        nn_ls = np.concatenate((nn_ls,nn3))
    
    field4 = np.where(np.logical_and(matched_1[:,0]>=pnt1[0], matched_1[:,1]>=pnt1[1]))[0]
    if len(field4)>0:
        nn4 = field4[find_nearest(pnt1, matched_1[field4], 1)]
        nn_ls = np.concatenate((nn_ls,nn4))
    
    
    matched_nearest = matched[nn_ls.astype(int)]
    weights = 1.0/((np.sum((matched_nearest[:,0:2] - pnt1)**2, axis=1))**0.5)
    
    transfer = np.average(matched_nearest[:,2:4] - matched_nearest[:,0:2], axis=0, weights = weights)
    mscore = np.average(matched_nearest[:,4], weights = weights)
    
    pnt2 = pnt1 + transfer
    
    return pnt2,mscore
    
def predict(pnt1, matched_arr, width1, height1, width2, height2):
    vertices = np.array([[0,0,0,0],\
                [width1,0,width2,0],\
                [0,height1,0,height2],\
                [width1,height1,width2,height2]])
    matched = matched_arr
    
    nn_ls = []
    
    matched_1 = matched[:,0:2]
    matched_1_dist = np.sum((matched_1-pnt1.reshape(1,-1))**2,axis=1)
    nn_ls = np.argsort(matched_1_dist)[0:4]
    
    
    matched_nearest = matched[nn_ls.astype(int)]
    weights = 1.0/((np.sum((matched_nearest[:,0:2] - pnt1)**2, axis=1))**0.5)
    
    transfer = np.average(matched_nearest[:,2:4] - matched_nearest[:,0:2], axis=0, weights = weights)
    mscore = np.average(matched_nearest[:,4], weights = weights)
    
    pnt2 = pnt1 + transfer
    
    return pnt2,mscore

    
def predict_inter(pnt1, matched_arr, width1, height1, width2, height2):
    # pnt1 = [x1, y1]
    vertices = np.array([[0,0,0,0],\
                [width1,0,width2,0],\
                [0,height1,0,height2],\
                [width1,height1,width2,height2]])
    
    matched = matched_arr
    
    nn_ls = []
    
    matched_1 = matched[:,0:2]
    matched_1_dist = np.sum((matched_1-pnt1.reshape(1,-1))**2,axis=1)
    nn_ls = np.argsort(matched_1_dist)[0:10]
    
    matched_nearest = matched[nn_ls.astype(int)]
    sidx = np.argsort(matched_nearest[:,0])
    try:
        fx = interp1d(matched_nearest[sidx,0], matched_nearest[sidx,2], kind='linear', fill_value="extrapolate")
    except:
        print(matched_nearest[sidx,0], matched_nearest[sidx,2])
    sidx = np.argsort(matched_nearest[:,1])
    fy = interp1d(matched_nearest[sidx,1], matched_nearest[sidx,3], kind='linear', fill_value="extrapolate")
    pnt2 = np.array([fx(pnt1[0]), fy(pnt1[1])])
    
    weights = 1.0/((np.sum((matched_nearest[:,0:2] - pnt1)**2, axis=1))**0.5)
    mscore = np.average(matched_nearest[:,4], weights = weights)
    
    return pnt2,mscore

    
def draw_img(img1,img2,pnt_img1,pnt_img2,savefile,edge=0):
    plt.close()
    fig,ax = plt.subplots(1,2,figsize=(20,5))
    ax[0].imshow(img1[:,:,::-1])
    ax[1].imshow(img2[:,:,::-1])
    
    for pp in pnt_img1:
        circ = patches.Circle(pp,3)
        ax[0].add_patch(circ)
        if edge > 0:
            edge_half = int(edge/2)
            rect = patches.Rectangle((pp[0]-edge_half,pp[1]-edge_half),edge,edge,linewidth=1,edgecolor='r',facecolor='none')
            ax[0].add_patch(rect)
            
    for pp in pnt_img2:
        circ = patches.Circle(pp,3)
        ax[1].add_patch(circ)
        if edge > 0:
            edge_half = int(edge/2)
            rect = patches.Rectangle((pp[0]-edge_half,pp[1]-edge_half),edge,edge,linewidth=1,edgecolor='r',facecolor='none')
            ax[1].add_patch(rect)
            
    plt.savefig(savefile)
    

root_dir = '/home/yutong/SPMatch/vp_test_sedan/'
dir_ls = glob.glob(os.path.join(root_dir,'a*e*'))
total_spnum=39
draw_p=True
for img_dir in dir_ls:
    if not 'a20e0' in img_dir:
        continue
        
    draw=False
        
    print(img_dir)
    # file list
    filelist = os.path.join(img_dir, 'file_list.txt')
    with open(filelist, 'r') as fh:
        contents = fh.readlines()
        
    img_list = [cc.strip() for cc in contents]
    syn_n = 3
    
    for img_idx2 in range(len(img_list)-syn_n, len(img_list)):
        img_file2 = img_list[img_idx2]
        img2 = cv2.imread(img_file2)
        rratio = 224.0/np.min(img2.shape[0:2])
        img2 = cv2.resize(img2, (0,0), fx=rratio, fy=rratio)
        
        img2_name = img_file2.split('/')[-1].split('.')[0]

        info_file = os.path.join(img_dir, 'sp_info_{}.pickle'.format(img2_name))
        sp_info = pickle.load(open(info_file, 'rb'))

        img2_pool4_file = '.'.join(img_file2.split('.')[0:-1])+'_pool4.pickle'
        img2_pool4 = pickle.load(open(img2_pool4_file, 'rb'))
        sp_features = []
        for isp in range(total_spnum):
            sp_features.append([])

            isp_ls = [spp[1] for spp in sp_info if spp[0]==isp]
            if len(isp_ls)==0:
                continue

            for sppos in isp_ls:
                pnt2 = np.array([min(img2.shape[1],max(0,sppos[0])),\
                                 min(img2.shape[0],max(0,sppos[1]))])
                pnt2_pool4 = (pnt2//16).astype(int)
                sp_features[-1].append(img2_pool4[min(img2_pool4.shape[0]-1,pnt2_pool4[1]), \
                                                  min(img2_pool4.shape[1]-1,pnt2_pool4[0])])


        for img_idx1 in range(len(img_list)-syn_n):
            img_file1 = img_list[img_idx1]
            img_file_name = img_file1.split('/')[-1]
            img_file_name = img_file_name.split('.')[0]

            img1_pool4_file = '.'.join(img_file1.split('.')[0:-1])+'_pool4.pickle'
            img1_pool4 = pickle.load(open(img1_pool4_file, 'rb'))

            # read in images
            img1 = cv2.imread(img_file1)
            matched_file = os.path.join(img_dir, 'img{}VSimg{}_matched.pickle'.format(img_idx1, img_idx2))
            with open(matched_file, 'rb') as fh:
                matched = pickle.load(fh)

            matched_arr = np.array(matched[1])
            # print(matched_arr.shape)

            matched_arr_rv = np.zeros_like(matched_arr)
            matched_arr_rv[:,0:2] = matched_arr[:,2:4]
            matched_arr_rv[:,2:4] = matched_arr[:,0:2]
            matched_arr_rv[:,4] = matched_arr[:,4]


            save_dir = os.path.join(img_dir, 'img{}VSimg{}_transferSP_nearest'.format(img_idx1, img_idx2))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)


            predict_sp_info = []
            for isp in range(total_spnum):
                isp_ls = [spp[1] for spp in sp_info if spp[0]==isp]

                if len(isp_ls)==0:
                    continue

                save_file = os.path.join(save_dir, 'transfer_sp{}.png'.format(isp))
                pnt_ls1 = []
                pnt_ls2 = []
                for spii,sppos in enumerate(isp_ls):
                    pnt2 = np.array([min(img2.shape[1],max(0,sppos[0])),\
                                     min(img2.shape[0],max(0,sppos[1]))])

                    # pnt1 = np.array([(sppos[0]+sppos[2])/2, (sppos[1]+sppos[3])/2])
                    pnt1,_ = predict(pnt2,matched_arr_rv,img2.shape[1],img2.shape[0],img1.shape[1],img1.shape[0] )
                    pnt1 = np.array([min(img1.shape[1],max(0,pnt1[0])),\
                                     min(img1.shape[0],max(0,pnt1[1]))])

                    pnt1_pool4 = (pnt1//16).astype(int)
                    pnt1_pool4_f = img1_pool4[min(img1_pool4.shape[0]-1,pnt1_pool4[1]), \
                                              min(img1_pool4.shape[1]-1,pnt1_pool4[0])]
                    mscore = 1-cdist(pnt1_pool4_f.reshape(-1,512), np.array(sp_features[isp][spii:spii+1]), 'cosine')[0,0]

                    img1_pool4_h, img1_pool4_w = img1_pool4.shape[0:2]
                    img1_pool4_dist = cdist(img1_pool4.reshape(-1,512), np.array(sp_features[isp][spii:spii+1]), 'cosine')

                    img1_pool4_dist = img1_pool4_dist.reshape(img1_pool4_h*img1_pool4_w,)

                    to_select = int(img1_pool4_h*img1_pool4_w*0.03) # top 3% close features
                    img1_pool4_cans = np.argsort(img1_pool4_dist)[0:to_select]

                    img1_cans = []
                    for p4c in img1_pool4_cans:
                        prow,pcol = np.unravel_index(p4c, (img1_pool4_h, img1_pool4_w))
                        img1_cans.append(np.array([pcol,prow])*16-42+50)

                    img1_nearest_idx = np.argmin(cdist(np.array(img1_cans), pnt1.reshape(1,2)).reshape(-1,))
                    img1_nearest = img1_cans[img1_nearest_idx]
                    img1_nearest_score = 1-img1_pool4_dist[img1_pool4_cans[img1_nearest_idx]]

                    pnt_ls1.append(pnt1)
                    pnt_ls2.append(pnt2)
                    predict_sp_info.append([img_file_name,isp,pnt1,mscore])

                if draw:
                    draw_img(img1,img2,pnt_ls1,pnt_ls2,save_file,100)

            sp_file = os.path.join(img_dir, 'img{}VSimg{}_transSP.pickle'.format(img_idx1, img_idx2))
            pickle.dump(predict_sp_info, open(sp_file, 'wb'))