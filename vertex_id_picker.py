import os, json

import d3, vdb

import numpy as np

import imageio

import matplotlib.pyplot as plt

import matplotlib.patches as patches



def read_vertex_list(obj_filename):

    with open(obj_filename) as f:

        lines = f.readlines()



    lines = [v.split(' ') for v in lines] 

    lines = [(float(v[1]),float(v[2]),float(v[3])) for v in lines]

    vertexs = np.array(lines)



    return vertexs



def get_vertex_id(pixel_coords, vertexs_3d, cam_pose, depth_filename):

    ''' Get vertex id

    pixel_coords: (x, y)

    obj_filename: *.obj for car or bike

    cam_pose: camera_pose, define the viewpoint

    depth_filename: depth for this view, to filter invisible points

    '''

    vertexs_2d = cam_pose.project_to_cam_space(vertexs_3d)

    # plt.plot(vertexs_2d[:,0], vertexs_2d[:,1], '*')

    depth = np.load(depth_filename)

    xs = vertexs_2d[:,0].astype('int'); ys = vertexs_2d[:,1].astype('int'); 

    zs = vertexs_2d[:,2]

    ds = depth[(ys, xs)]

    # print(zs)

    # print(ds)

    # print(abs(zs - ds))

    visible = abs(zs - ds) < 10

    invisible = abs(zs - ds) > 10

    # print(visible.sum())

    # print(invisible.sum())



    # visible_index = np.where(visible)[0]

    # print(visible_index)

    # visible_vertexs = vertexs_2d[visible_index, :]

    # print(visible_vertexs.shape)

    vertexs_2d[invisible, :] = 10e10

    dist = np.linalg.norm(vertexs_2d[:,[0,1]] - np.array(pixel_coords), axis=1)

    vertex_index = np.argmin(dist)

    assert dist[vertex_index] == min(dist)

    # print('Match a match for coord %s, min distance is %.2f, vertex id is %d' % (pixel_coords, min(dist), vertex_index))


    return vertex_index, vertexs_2d[vertex_index, :]



def project_vertex_id(vertex_id, vertexs_3d, cam_pose):

    vertexs_2d = cam_pose.project_to_cam_space(vertexs_3d)

    point_2d = vertexs_2d[vertex_id, :]

    return point_2d



def draw_bb(ax, bbox):

    x0, x1, y0, y1 = bbox

    cx = (x0 + x1) / 2; cy = (y0 + y1) / 2 

    w = x1 - x0; h = y1 - y0

    rect = patches.Rectangle((x0, y0), w, h, linewidth=1, edgecolor='r', facecolor='none')

    ax.add_patch(rect)



def get_frame_info(db_root, obj_name, cam_name, frame_id):

    obj_filename = os.path.join(db_root, obj_name + '.obj')

    lit_filename = os.path.join(db_root, cam_name, 'lit', '%08d.png' % frame_id)

    depth_filename = os.path.join(db_root, cam_name, 'depth', '%08d.npy' % frame_id)

    cam_filename = os.path.join(db_root, cam_name, 'caminfo', '%08d.json' % frame_id)

    scene_filename = os.path.join(db_root, 'scene', '%08d.json' % frame_id)

    seg_filename = os.path.join(db_root, cam_name, 'seg', '%08d.png' % frame_id)

    with open(scene_filename) as f:

        data = json.load(f)



    obj_info = data[obj_name]

    obj_location = np.array([

        obj_info['Location']['X'],

        obj_info['Location']['Y'],

        obj_info['Location']['Z']

    ])



    with open(cam_filename) as f:

        data = json.load(f)



    loc = data['Location']; rot = data['Rotation']

    assert data['Fov'] == 90

    cam_pose = d3.CameraPose(loc['X'], loc['Y'], loc['Z'], 

        rot['Pitch'], rot['Yaw'], rot['Roll'], 

        data['FilmWidth'], data['FilmHeight'], data['FilmWidth'] / 2)



    vertexs_3d = read_vertex_list(obj_filename)

    vertexs_3d = vertexs_3d + obj_location



    seg_im = imageio.imread(seg_filename)

    obj_mask = vdb.get_obj_mask(seg_im, obj_info['AnnotationColor'])

    bbox = vdb.seg2bb(obj_mask)

    

    return lit_filename, cam_pose, depth_filename, vertexs_3d, bbox



def main():

    db_root = '/mnt/c/data/temp/DenseMatching/20181026_1324'

    obj_name = 'Sedan2Door_Vehicle_Sedan2Door_LOD0_8'

    cam_name = 'sedan2door'

    src_frame_id = 0

    tgt_frame_id = 1



    src_lit_filename, src_cam_pose, src_depth_filename, src_vertexs_3d, src_bbox  =  get_frame_info(db_root, obj_name, cam_name, src_frame_id)

    tgt_lit_filename, tgt_cam_pose, d, tgt_vertexs_3d, tgt_bbox  =  get_frame_info(db_root, obj_name, cam_name, tgt_frame_id)



    ax1 = plt.subplot(121); 

    ax2 = plt.subplot(122); 

    ax1.imshow(plt.imread(src_lit_filename))

    ax2.imshow(plt.imread(tgt_lit_filename))

    for input_vertex in [(10,10), (320, 190), (353, 105), (345, 220), (410, 210)]:

        vertex_id, matched_2d = get_vertex_id(input_vertex, src_vertexs_3d, src_cam_pose, src_depth_filename)

        predicted_2d = project_vertex_id(vertex_id, tgt_vertexs_3d, tgt_cam_pose)



        ax1.plot(input_vertex[0], input_vertex[1], 'ro')

        draw_bb(ax1, src_bbox)

        ax1.plot(matched_2d[0], matched_2d[1], 'b*')

        ax2.plot(predicted_2d[0], predicted_2d[1], 'b*')

        draw_bb(ax2, tgt_bbox)

    plt.show()



if __name__ == '__main__':

    main()

    