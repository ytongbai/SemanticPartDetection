xlist="0 10 80 90 100 170 180 190 260 270 280 350"
for azi in $(seq 0 10 350)
do
if [[ ${xlist} =~ (^|[[:space:]])${azi}($|[[:space:]]) ]] && true || false
then
    continue
fi
for ele in $(seq -5 5 35)
do
if [ -d "/home/yutong/SPMatch/vp_test_sedan/a${azi}e${ele}/" ]
then
    /usr/local/MATLAB/R2017a/bin/matlab -nodisplay -nosplash -nodesktop -r "dir1='/mnt/4TB_b/qing/SPMatch/vp_test_sedan/a${azi}e${ele}/';pairImgs_split;exit"
else
    echo "not exist: /home/yutong/SPMatch/vp_test_sedan/a${azi}e${ele}/"
fi
done
done
