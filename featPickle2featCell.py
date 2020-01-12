import numpy as np
import os
import pickle
import scipy.io as sio
import glob

featDim = 512
offset = 3
layer_n = 'pool4'

for vpi in [0, 45, 90, 135, 180, 225, 270, 315]:
    img_dir = '/mnt/4TB_b/qing/SPMatch/vp_examples/{}/'.format(vpi)
    pickle_list = glob.glob(os.path.join(img_dir, '*_{}.pickle'.format(layer_n)))
    N = len(pickle_list)
    print('Transfer features for {} images...'.format(N))
    
    for file_cache_feat in pickle_list:
        with open(file_cache_feat, 'rb') as fh:
            feat_org = pickle.load(fh)
            
        feat_r, feat_c = feat_org.shape[0:2]
        feat = feat_org[offset:feat_r-offset, offset:feat_c-offset, :]

        feat_r, feat_c = feat.shape[0:2]
        lff = feat.reshape(-1, featDim)
        lff_norm = lff/np.sqrt(np.sum(lff**2, 1)).reshape(-1,1)
        feat = lff_norm.reshape(feat_r,feat_c,-1)

        assert(feat.shape[2]==featDim)

        info_r = np.tile(np.arange(feat_r).reshape(feat_r,1, 1), [1, feat_c, 1])+1
        info_c = np.tile(np.arange(feat_c).reshape(1,feat_c, 1), [feat_r, 1, 1])+1
        info_nn = np.ones((feat_r, feat_c, 1))*1

        feat_info = np.concatenate([feat, info_nn, info_r, info_c], axis=2)
        feat_cell_locinfo = np.array([feat_info], dtype=np.object)
        
    save_file = os.path.join(img_dir, 'feat_cell_locinfo.mat')
    sio.savemat(save_file, {'feat_cell_locinfo': feat_cell_locinfo})
    