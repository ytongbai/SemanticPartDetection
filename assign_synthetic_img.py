import os
import glob
from shutil import copyfile

root_dir = '/mnt/4TB_b/qing/SPMatch'
source_dir = os.path.join(root_dir, 'from_weichao','DenseMatching_CarOnly','sedan4door','lit_cropped')
target_dir = os.path.join(root_dir, 'vp_test_sedan', 'a*e*')
dir_list = glob.glob(target_dir)
for dd in dir_list:
    info = dd.split('/')[-1]
    i1 = info.index('a')
    i2 = info.index('e')
    azi = int(info[i1+1:i2])+10
    if azi == 360:
        azi=0
    ele = int(info[i2+1:])
    
    if azi > 90:
        j1 = (90 - (azi-360))//5
    else:
        j1 = (90 - azi)//5
        
    j2 = ele//5+1
    frame_id = j2*72+j1
    
    s_file = os.path.join(source_dir, '%08d.png' % frame_id)
    t_file = os.path.join(dd, '%08d.png' % frame_id)
    copyfile(s_file,t_file)