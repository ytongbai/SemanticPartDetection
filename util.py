import numpy
import os
import scipy.io as sio
import h5py
import numpy as np
from scipy import misc
import pickle
import torch
import math

def img_rescale(img, delta_scale):
    new_shape0 = int(img.shape[0] * delta_scale)
    new_shape1 = int(img.shape[1] * delta_scale)
    return misc.imresize(img, (new_shape0, new_shape1))


def bbox_rescale(img, bbox, rescale):
    # calculate min_edge and calculate scale only based on bbox
    delta_scale = bbox_to_delta_scale(bbox, rescale)
    return img_rescale(img, delta_scale), bbox * delta_scale, delta_scale


def bbox_to_delta_scale(bbox, rescale):
    min_edge = min(bbox[3] - bbox[1], bbox[2] - bbox[0])
    assert (min_edge > 0.5)
    if rescale > 0:
        delta_scale = rescale / float(min_edge)
    else:
        delta_scale = -rescale
    return delta_scale


def bbox_crop(img, bbox):
    """
    :param img:
    :param bbox:
    :return: cropped image, pixel delta at [x, y]
    """
    bbox = bound_bbox_to_int(img, bbox)
    cropped = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    return cropped, np.array([bbox[0], bbox[1]])


def bound_bbox_to_int(img, bbox):
    """
    :param img: np.array(RGB_img) shape = [y, x, 3]
    :param bbox: [x0, y0, x1, y1]
    :return: bounded bbox [x0:x1), [y0:y1) => [x0, y0, x1, y1]
    """
    if bbox[1] < -0.5:
        # print(str(bbox[1]) + ' < -0.5')
        bbox[1] = -0.5
    if bbox[0] < -0.5:
        # print(str(bbox[0]) + ' < -0.5')
        bbox[0] = -0.5
    if bbox[3] > img.shape[0] - 0.5:
        # print(str(bbox[3]) + ' > ' + str(img.shape[0] - 0.5))
        bbox[3] = img.shape[0] - 0.5
    if bbox[2] > img.shape[1] - 0.5:
        # print(str(bbox[2]) + ' > ' + str(img.shape[1] - 0.5))
        bbox[2] = img.shape[1] - 0.5
    if bbox[1] >= bbox[3] or bbox[0] >= bbox[2]:
        print('invalid bbox')
    return np.round((bbox + 0.5)).astype(np.int32)


def bbox_rescale_and_bbox_crop(img, bbox, rescale, crop=True):
    """
    :param img: RGB images
    :param bbox: rescale and crop based one bbox[x0, y0, x1, y1], set to img size if do not
    :param rescale: negative means delta_scale, positive means pixel scale of short edge
    :param crop: crop the bbox or not
    :return: img, new = old * delta_scale - delta_xy
    """
    delta_xy = np.array([0.0, 0.0])
    img, bbox, delta_scale = bbox_rescale(img, bbox, rescale)
    if crop:
        img, delta_xy = bbox_crop(img, bbox)
    return img, delta_scale, delta_xy


def area_rescale(img, sqrt_area=300.0):
    """
    :param img: RGB images
    :param sqrt_area: desired sqrt of area
    :return: img, new = (old - delta_xy) * delta_scale
    """
    delta_xy = np.array([0.0, 0.0])
    img_sqrt_area = math.sqrt(img.shape[0] * img.shape[1])
    delta_scale = sqrt_area / img_sqrt_area
    return img_rescale(img, delta_scale), delta_xy, delta_scale



def bbox_crop_and_bbox_rescale(img, bbox, rescale, crop=True):
    """
    :param img: RGB images
    :param bbox: rescale and crop based one bbox[x0, y0, x1, y1], set to img size if do not
    :param rescale: negative means delta_scale, positive means pixel scale of short edge
    :param crop: crop the bbox or not
    :return: img, new = (old - delta_xy) * delta_scale
    """
    delta_xy = np.array([0.0, 0.0])
    
    if crop:
        img, delta_xy = bbox_crop(img, bbox)
        bbox = np.array([-0.5, -0.5, img.shape[1] - 0.5, img.shape[0] - 0.5])
    else:
        bbox = bound_bbox_to_int(img, bbox)
        
    img, bbox, delta_scale = bbox_rescale(img, bbox, rescale)
    return img, delta_xy, delta_scale


def im2rgb(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        return img
    if len(img.shape) == 3 and img.shape[2] == 4:
        return img[:, :, :3]
    if len(img.shape) == 2:
        return np.expand_dims(img, axis=2).repeat(3, axis=2)
    if len(img.shape) == 3 and img.shape[2] == 1:
        return img.repeat(3, axis=2)
    print('im2rgb error')
    return img


# def np_rgb2img_var(img):
#     numpy_array = np.float32(img)  # H, W, RGB
#     numpy_array -= np.float32([[[123.68, 116.779, 103.939]]])  # H, W, RGB
#     numpy_array = numpy_array[:, :, ::-1]  # H, W, BGR
#     numpy_array = numpy_array.transpose([2, 0, 1])  # BGR, H, W
#     numpy_array = np.expand_dims(numpy_array, axis=0)  # 1, BGR, H, W
#
#     input_tensor = torch.from_numpy(numpy_array.copy())
#     return torch.autograd.Variable(input_tensor)

def get_coverage_firing_stats(encoding, mask):
    # encoding: C,H,W. mask: H,W
    firecnt_pixel = np.sum(encoding, axis=0)
    fire_pixel = firecnt_pixel!=0
    
    firing = np.mean(firecnt_pixel, axis=None)
    firing_b = np.mean(firecnt_pixel[np.where(mask==1)])
    firing_ob = np.mean(firecnt_pixel[np.where(mask==0)])
    
    coverage = np.mean(fire_pixel, axis=None)
    coverage_b = np.mean(fire_pixel[np.where(mask==1)])
    coverage_ob = np.mean(fire_pixel[np.where(mask==0)])
    return (firing, firing_b, firing_ob, coverage, coverage_b, coverage_ob)

def predict_bbox(vc_encoding, short_edge_len = 14, coverage_thres = 0.3):
    height = vc_encoding.shape[2]
    width = vc_encoding.shape[3]
    
    mask = np.zeros((height, width))
    center = (height//2, width//2)
    mask[center[0]-1:center[0]+2, center[1]-1:center[1]+2] = 1 # start with 3 by 3
    rlist, clist = np.where(mask==1)
    short_edge = np.min((np.max(rlist)-np.min(rlist), np.max(clist)-np.min(clist)))+1
    while True:
        rmin = np.min(rlist)
        rmax = np.max(rlist)
        cmin = np.min(clist)
        cmax = np.max(clist)

        mask_ls = []
        fb_ls = []
        cb_ls = []
        if rmin>0:
            mask1 = np.copy(mask)
            mask1[rmin-1, cmin:cmax+1] = 1
            _, fb1, _, _, cb1, _ = get_coverage_firing_stats(vc_encoding[0], mask1)
            mask_ls.append(mask1)
            fb_ls.append(fb1)
            cb_ls.append(cb1)

        if rmax<height-1:
            mask2 = np.copy(mask)
            mask2[rmax+1, cmin:cmax+1] = 1
            _, fb2, _, _, cb2, _ = get_coverage_firing_stats(vc_encoding[0], mask2)
            mask_ls.append(mask2)
            fb_ls.append(fb2)
            cb_ls.append(cb2)

        if cmin>0:
            mask3 = np.copy(mask)
            mask3[rmin:rmax+1, cmin-1] = 1
            _, fb3, _, _, cb3, _ = get_coverage_firing_stats(vc_encoding[0], mask3)
            mask_ls.append(mask3)
            fb_ls.append(fb3)
            cb_ls.append(cb3)

        if cmax<width-1:
            mask4 = np.copy(mask)
            mask4[rmin:rmax+1, cmax+1] = 1
            _, fb4, _, _, cb4, _ = get_coverage_firing_stats(vc_encoding[0], mask4)
            mask_ls.append(mask4)
            fb_ls.append(fb4)
            cb_ls.append(cb4)
            
        if len(cb_ls)==0:
            break

        mask = mask_ls[np.argmax(cb_ls)]
        cb_curr = np.max(cb_ls)

        rlist, clist = np.where(mask==1)
        short_edge = np.min((np.max(rlist)-np.min(rlist), np.max(clist)-np.min(clist)))+1

        if short_edge > short_edge_len or cb_curr < coverage_thres:
            break
            
    for mm in mask_ls:
        mask = np.logical_or(mask, mm)
            
    return mask