import datetime
from joblib import Parallel, delayed
import numpy as np

SP = dict()
SP['criteria'] = 'iou'
SP['iou_thresh'] = 0.5

def VOCap(rec, prec):
    mrec = np.append(0, rec)
    mrec = np.append(mrec, 1)
    
    mpre = np.append(0, prec)
    mpre = np.append(mpre, 0)
    
    for ii in range(len(mpre)-2,-1,-1):
        mpre[ii] = max(mpre[ii], mpre[ii+1])
        
    msk = [i!=j for i,j in zip(mrec[1:], mrec[0:-1])]
    ap = np.sum((mrec[1:][msk]-mrec[0:-1][msk])*mpre[1:][msk])
    return ap

def eval_AP_inner(inp):  
    sp_detection, spanno, img_size = inp
    N = len(spanno) 
    kp_pos = np.sum([spanno[nn].shape[0] for nn in range(N)])

    tot = sp_detection.shape[0]
    sort_idx = np.argsort(-sp_detection[:, 5])
    id_list = sp_detection[sort_idx, 0]
    col_list = (sp_detection[sort_idx, 1] + sp_detection[sort_idx, 3]) / 2
    row_list = (sp_detection[sort_idx, 2] + sp_detection[sort_idx, 4]) / 2
    bbox_list = sp_detection[sort_idx, 1:5].astype(int)

    tp = np.zeros(tot)
    fp = np.zeros(tot)
    flag = np.zeros((N, 20))
    for dd in range(tot):
        if np.sum(flag) == kp_pos:
            fp[dd:] = 1
            break

        img_id = int(id_list[dd])
        col_c = col_list[dd]
        row_c = row_list[dd]
        if SP['criteria'] == 'dist':
            min_dist = np.inf
            inst = spanno[img_id]
            for ii in range(inst.shape[0]):
                xx = (inst[ii, 0] + inst[ii, 2]) / 2
                yy = (inst[ii, 1] + inst[ii, 3]) / 2

                if np.sqrt((xx - col_c) ** 2 + (yy - row_c) ** 2) < min_dist:
                    min_dist = np.sqrt((xx - col_c) ** 2 + (yy - row_c) ** 2)
                    min_idx = ii

            if min_dist < SP['dist_thresh'] and flag[img_id, min_idx] == 0:
                tp[dd] = 1
                flag[img_id, min_idx] = 1
            else:
                fp[dd] = 1

        elif SP['criteria'] == 'iou':
            max_iou = -np.inf
            inst = spanno[img_id]
            for ii in range(inst.shape[0]):
                bbgt = inst[ii]
                bb = bbox_list[dd]
                bb = np.array([max(np.ceil(bb[0]), 1), max(np.ceil(bb[1]), 1),
                               min(np.floor(bb[2]), img_size[img_id][1]), min(np.floor(bb[3]), img_size[img_id][0])])

                bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                iw = bi[2] - bi[0] + 1
                ih = bi[3] - bi[1] + 1

                if iw > 0 and ih > 0:
                    ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + \
                         (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - \
                         iw * ih
                    ov = iw * ih / ua
                    if ov > max_iou:
                        max_iou = ov
                        max_idx = ii

            if max_iou > SP['iou_thresh'] and flag[img_id, max_idx] == 0:
                tp[dd] = 1
                flag[img_id, max_idx] = 1
            else:
                fp[dd] = 1

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / kp_pos
    prec = tp / (tp + fp)
    ap = VOCap(rec, prec)
    return ap


def eval_AP(sp_detection, spanno, img_size):
    paral_num = 6
    inp_ls = [(np.concatenate(tuple([sp_detection[img_id][sp_id] for img_id in range(len(spanno))]), axis=0),
              [spanno[img_id][sp_id] for img_id in range(len(spanno))],
              img_size) for sp_id in range(len(sp_detection[0]))]

    ap_ls = np.array(Parallel(n_jobs=paral_num)(delayed(eval_AP_inner)(i) for i in inp_ls))
    for sp_id in range(len(sp_detection[0])):
        print('{:3.1f}'.format(ap_ls[sp_id]*100))

    print(np.nanmean(ap_ls) * 100)
    








