import pickle
import os,sys
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def predict(pnt1, matched_arr, width1, height1, width2, height2):
    # pnt1 = [x1, y1]
    matched = matched_arr
    
    matched_1 = matched[:,0:2]
    matched_1_dist = np.sum((matched_1-pnt1.reshape(1,-1))**2,axis=1)
    nn_ls = np.argsort(matched_1_dist)[0:4]
    
    matched_nearest = matched[nn_ls.astype(int)]
    weights = 1.0/((np.sum((matched_nearest[:,0:2] - pnt1)**2, axis=1))**0.5)
    
    transfer = np.average(matched_nearest[:,2:4]/np.array([[width2, height2]]) - \
                          matched_nearest[:,0:2]/np.array([[width1, height1]]), axis=0, weights = weights)
    mscore = np.average(matched_nearest[:,4], weights = weights)
    
    pnt2 = (pnt1/np.array([width1, height1]) + transfer)*np.array([width2, height2])
    
    return pnt2,mscore


def find_nearest(pnt1, arr, num=1):
    # arr = [[x1,y1],...,[xn,yn]]
    dist = (np.sum((pnt1.reshape(1,-1)-arr[:,0:4])**2,axis=1))**0.5
    return np.argsort(dist)[0:num]


def predict2(pnt1, matched_arr, width1, height1, width2, height2):
    # pnt1 = [x1, y1]
    vertices = np.array([[0,0,0,0],\
                [width1,0,width2,0],\
                [0,height1,0,height2],\
                [width1,height1,width2,height2]])
    
    # matched = np.concatenate((matched_arr, vertices), axis=0)
    matched = matched_arr
    
    nn_ls = []
    
    matched_1 = matched[:,0:2]
    field1 = np.where(np.logical_and(matched_1[:,0]<=pnt1[0], matched_1[:,1]<=pnt1[1]))[0]
    if len(field1)>0:
        nn1 = field1[find_nearest(pnt1, matched_1[field1])]
        nn_ls = np.concatenate((nn_ls,nn1))
    
    field2 = np.where(np.logical_and(matched_1[:,0]>=pnt1[0], matched_1[:,1]<=pnt1[1]))[0]
    if len(field2)>0:
        nn2 = field2[find_nearest(pnt1, matched_1[field2])]
        nn_ls = np.concatenate((nn_ls,nn2))
    
    
    field3 = np.where(np.logical_and(matched_1[:,0]<=pnt1[0], matched_1[:,1]>=pnt1[1]))[0]
    if len(field3)>0:
        nn3 = field3[find_nearest(pnt1, matched_1[field3])]
        nn_ls = np.concatenate((nn_ls,nn3))
    
    field4 = np.where(np.logical_and(matched_1[:,0]>=pnt1[0], matched_1[:,1]>=pnt1[1]))[0]
    if len(field4)>0:
        nn4 = field4[find_nearest(pnt1, matched_1[field4])]
        nn_ls = np.concatenate((nn_ls,nn4))
    
    matched_nearest = matched[nn_ls]
    transfer = np.mean(matched_nearest[:,2:4] - matched_nearest[:,0:2], axis=0)
    mscore = np.mean(matched_nearest[:,4])
    
    pnt2 = pnt1 + transfer
    
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
    

root_dir = '/mnt/4TB_b/qing/SPMatch/vp_examples_less/'
for vp in [0,45,90,135,180,225,270,315]:
    img_dir = os.path.join(root_dir, str(vp))
    # file list
    filelist = os.path.join(img_dir, 'file_list.txt')
    with open(filelist, 'r') as fh:
        contents = fh.readlines()
        
    img_list = [cc.strip() for cc in contents]
    img_idx2 = len(img_list)-1
    img_file2 = img_list[img_idx2]
    img2 = cv2.imread(img_file2)
    rratio = 224.0/np.min(img2.shape[0:2])
    img2 = cv2.resize(img2, (0,0), fx=rratio, fy=rratio) 

    info_file = os.path.join(img_dir, 'info.pickle')
    image_name_list, instance_info_list, sp_info_list, vp_set_list = pickle.load(open(info_file, 'rb'))

    for img_idx1 in range(img_idx2):
        img_file1 = img_list[img_idx1]
        
        # read in images
        img1 = cv2.imread(img_file1)
        # rratio = 224.0/np.min(img1.shape[0:2])
        # img1 = cv2.resize(img1, (0,0), fx=rratio, fy=rratio) 


        # load pool4 match result
        matched_file = os.path.join(img_dir, 'img{}VSimg{}_matched.pickle'.format(img_idx1, img_idx2))
        with open(matched_file, 'rb') as fh:
            matched = pickle.load(fh)

        N = len(matched[0]) + len(matched[1])
        matched_arr = np.zeros((N, 5))

        for nn in range(N):
            if nn < len(matched[0]):
                matched_arr[nn] = np.array(matched[0][nn])
            else:
                matched_arr[nn] = np.array(matched[1][nn-len(matched[0])])


        # print(matched_arr.shape)
        
        for ispinfo,iname in zip(sp_info_list, image_name_list):
            if iname in img_list[img_idx1]:
                sp_info = ispinfo

        save_dir = os.path.join(img_dir, 'img{}VSimg{}_transferSP_nearest'.format(img_idx1, img_idx2))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    

        predict_sp_info = []
        for isp in range(39):
            isp_ls = [spp[1] for spp in sp_info if spp[0]==isp]

            if len(isp_ls)==0:
                continue

            save_file = os.path.join(save_dir, 'transfer_sp{}.png'.format(isp))
            pnt_ls1 = []
            pnt_ls2 = []
            for sppos in isp_ls:
                pnt1 = np.array([min(img1.shape[1],max(0,(sppos[0]+sppos[2])/2)),\
                                 min(img1.shape[0],max(0,(sppos[1]+sppos[3])/2))])
                # pnt1 = np.array([(sppos[0]+sppos[2])/2, (sppos[1]+sppos[3])/2])
                pnt2,_ = predict(pnt1,matched_arr,img1.shape[1],img1.shape[0],img2.shape[1],img2.shape[0] )
                pnt2 = np.array([min(img2.shape[1],max(0,pnt2[0])),\
                                 min(img2.shape[0],max(0,pnt2[1]))])

                pnt_ls1.append(pnt1)
                pnt_ls2.append(pnt2)
                predict_sp_info.append([isp,pnt2])

            draw_img(img1,img2,pnt_ls1,pnt_ls2,save_file,100)

        sp_file = os.path.join(img_dir, 'img{}VSimg{}_transSP.pickle'.format(img_idx1, img_idx2))
        pickle.dump(predict_sp_info, open(sp_file, 'wb'))
