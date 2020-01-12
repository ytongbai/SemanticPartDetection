from scipy.spatial.distance import cdist
from FeatureExtractor_torch import FeatureExtractor_torch
import os, glob
import numpy as np
import math
import pickle
from copy import *

def extractLayerFeat(img_dir, extractor, scale_size=224):
    img_list = glob.glob(os.path.join(img_dir, '*.JPEG'))
    N = len(img_list)
    print('Extracting features from {} images...'.format(N))
    
    
    feat_set = [None for nn in range(N)]
    for nn,impath in enumerate(img_list):
        layer_feature = extractor.extract_feature_image_from_path(impath)[0]
        layer_feature = layer_feature.transpose([1,2,0])
        feat_set[nn] = layer_feature
        
    print('extracted feature shape: {}'.format(feat_set[0].shape))
    
    return feat_set


def extractLayerFeat_one(img_path, extractor, scale_size=224):
    layer_feature = extractor.extract_feature_image_from_path(img_path)[0]
    layer_feature = layer_feature.transpose([1,2,0])
    
    # print('extracted feature shape: {}'.format(feat_set[0].shape))
    return layer_feature
        
            
if __name__=='__main__':
    layer_n = 'pool3'
    extractor = FeatureExtractor_torch(layer=layer_n)
    
    for vpi in [0, 45, 90, 135, 180, 225, 270, 315]:
        img_dir = '/mnt/4TB_b/qing/SPMatch/vp_examples/{}/'.format(vpi)
        img_list = glob.glob(os.path.join(img_dir, '*.JPEG'))
        N = len(img_list)
        print('Extracting features from {} images...'.format(N))
        for nn,impath in enumerate(img_list):
            layer_feature = extractLayerFeat_one(impath, extractor)
            file_cache_feat = '.'.join(impath.split('.')[0:-1])+'_{}.pickle'.format(layer_n)
            with open(file_cache_feat, 'wb') as fh:
                pickle.dump(layer_feature, fh)