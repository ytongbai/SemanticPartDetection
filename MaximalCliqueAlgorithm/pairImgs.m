%% VGG parameter ( We fix pooling 4 here)
% clear all;
% close all;
Apad_set = [2, 6, 18, 42, 90]; % padding size = 42
Astride_set = [2, 4, 8, 16, 32]; % stride size = 16
featDim_set = [64, 128, 256, 512, 512]; % feature dimension = 512

Arf_set = [6, 16, 44, 100, 212]; % Arf_set = 100
offset_set = round(Apad_set./Astride_set)-1;   %%Round to nearest decimal or integer. offset = 3
layer_n = 3;  %% this is python's index of pooling layer 4.
Apad = Apad_set(layer_n+1);
Astride = Astride_set(layer_n+1);
featDim = featDim_set(layer_n+1);
Arf = Arf_set(layer_n+1); %% add 1 here 
offset = offset_set(layer_n+1);



%% They are corresponding between the matrix and img_idx normally it is 512(extracted from pooling-4 layer), but add 3 demensions into that.
%% The other 3 demensions are:
%% feat_cell_locinfo.mat 是指每个feat后三位加上了(img_index,i,j)的feat来源loc信息
% dir1='/mnt/4TB_b/qing/SPMatch/vp_test_sedan/a40e0/';

load(strcat(dir1, 'feat_cell_locinfo_3syn.mat'));
for img_idx2 = drange(size(feat_cell_locinfo,2)-2:size(feat_cell_locinfo,2))
temp_feat_tensor_info2=feat_cell_locinfo{1,img_idx2};
feat_num2=size(temp_feat_tensor_info2,1)*size(temp_feat_tensor_info2,2);

for img_idx1 = drange(1:size(feat_cell_locinfo,2)-3)
% for img_idx1 = drange(1:1)
img_idx1
temp_feat_tensor_info1=feat_cell_locinfo{1,img_idx1};
feat_num1=size(temp_feat_tensor_info1,1)*size(temp_feat_tensor_info1,2);


%% Initial the compare vector:

compare_vec_len = feat_num1*feat_num2;  
compare_vec=zeros(compare_vec_len,1+3+3); 

%% Compute the compare_vec
for i=1:compare_vec_len
    [idx_r,idx_c]=vec2Mat(i,feat_num1,feat_num2);
    [i1,j1]=vec2Mat(idx_r, size(temp_feat_tensor_info1,1), size(temp_feat_tensor_info1,2));
    [i2,j2]=vec2Mat(idx_c, size(temp_feat_tensor_info2,1), size(temp_feat_tensor_info2,2));
    
    % temp_feat_tensor_info = feat + info
    % what is the info? Need to ask
    feat1=reshape(temp_feat_tensor_info1(i1,j1,1:512),1,512);  %size(feat1) = 1 512
    feat2=reshape(temp_feat_tensor_info2(i2,j2,1:512),1,512);  %size(feat2) = 1 512
    
    info1=reshape(temp_feat_tensor_info1(i1,j1,512+1:512+3),1,3);
    info2=reshape(temp_feat_tensor_info2(i2,j2,512+1:512+3),1,3);

    % The composition of the compare_vec, including 7 dimensions:
    % Is compare_vec equals to the adjcent matrix? Needs confirmation.
    compare_vec(i,1)=dot(feat1,feat2); 
    compare_vec(i,2:4)=info1;
    compare_vec(i,5:7)=info2;

end
%% Get the compare_vec_sort. 
[sortval, sortpos]=sort(compare_vec(:,1),'descend'); 
compare_vec_sort=compare_vec(sortpos,:);


%% Find the best matches between two images
% Process image1 first
posinimg1=compare_vec_sort(:,3)*1E3+compare_vec_sort(:,4);
[~,unique_pos1]=unique(posinimg1); % When there are only 2 parameters, the unique_pos1 is the element that appears firstly.
compare_vec_sort_tmp1=compare_vec_sort(unique_pos1,:);

posinimg2=compare_vec_sort_tmp1(:,6)*1E3+compare_vec_sort_tmp1(:,7);
[~,unique_pos2]=unique(posinimg2);
to_select1 = unique_pos1(unique_pos2);

compare_vec_sort1 = compare_vec_sort(to_select1,:);
%% Again, find the best matches between two images in the remaining features
% posinimg1_1=compare_vec_sort1(:,3)*1E3+compare_vec_sort1(:,4);
% posinimg2_1=compare_vec_sort1(:,6)*1E3+compare_vec_sort1(:,7);
% 
% posinimg1=compare_vec_sort(:,3)*1E3+compare_vec_sort(:,4);
% posinimg2=compare_vec_sort(:,6)*1E3+compare_vec_sort(:,7);

% compare_vec_sort_r = compare_vec_sort(~(ismember(posinimg1, posinimg1_1) | ismember(posinimg2, posinimg2_1)),:);

% compare_vec_sort_r = compare_vec_sort(~ismember(1:compare_vec_len, to_select1),:);
% posinimg1_r=compare_vec_sort_r(:,3)*1E3+compare_vec_sort_r(:,4);
% [~,unique_pos1_r]=unique(posinimg1_r); % When there are only 2 parameters, the unique_pos1 is the element that appears firstly.
% compare_vec_sort_tmp1_r=compare_vec_sort_r(unique_pos1_r,:);
% 
% posinimg2_r=compare_vec_sort_tmp1_r(:,6)*1E3+compare_vec_sort_tmp1_r(:,7);
% [~,unique_pos2_r]=unique(posinimg2_r);
% to_select2 = unique_pos1_r(unique_pos2_r);
% 
% compare_vec_sort2 = compare_vec_sort_r(to_select2, :);
%% Cut the two compare_vec_sort matrices
% the program will freeze if compare more than 200 feature pairs
% compare_vec_sort1 = vertcat(compare_vec_sort1,compare_vec_sort2);

max_num_to_compare1 = 100;
% max_num_to_compare2 = 100;

[sortval2, sortpos2]=sort(compare_vec_sort1(:,1),'descend'); 
compare_vec_sort1=compare_vec_sort1(sortpos2,:);
if (size(compare_vec_sort1, 1) > max_num_to_compare1)
    compare_vec_sort1 = compare_vec_sort1(1:max_num_to_compare1, :);
end

% [sortval2, sortpos2]=sort(compare_vec_sort2(:,1),'descend'); 
% compare_vec_sort2=compare_vec_sort2(sortpos2,:);
% if (size(compare_vec_sort2, 1) > max_num_to_compare2)
%     compare_vec_sort2 = compare_vec_sort2(1:max_num_to_compare2, :);
% end

% compare_vec_sort1 = vertcat(compare_vec_sort1,compare_vec_sort2);
%% MCP
maximal_clique_q1=MCP(compare_vec_sort1);

% maximal_clique_q=find(MC(:,2)==1);
% temp_test=rela_adj_matrix(:,maximal_clique_q);
% temp_test=temp_test(maximal_clique_q,:);

make_filepath=[dir1,'img',num2str(img_idx1-1),'VSimg',num2str(img_idx2-1)];
% mkdir(make_filepath);

% compare_vec_sort2 = compare_vec_sort1(~ismember(1:size(compare_vec_sort1, 1), ... 
%     maximal_clique_q1),:);
% maximal_clique_q2=MCP(compare_vec_sort2);

rst = compare_vec_sort1(maximal_clique_q1,:);
save_filename = strcat(make_filepath, '.mat');
save(save_filename, 'rst');


%% Show the result,change the path of the image here:
% img_filepath=['/mnt/1TB_SSD/dataset/PASCAL3D+_cropped/car_imagenet/'];
% img_list_file=['/mnt/1TB_SSD/dataset/PASCAL3D+_release1.1/Image_sets/car_imagenet_train.txt'];
% file_list=textread(img_list_file, '%s', 'delimiter', '\n', 'whitespace', '');

% img_list_file1=strcat(dir1, 'file_list.txt');
% file_list=textread(img_list_file1, '%s', 'delimiter', '\n', 'whitespace', '');
% img1_o=imread([file_list{img_idx1}]);
% % scl1 = 224/min([size(img1_o,1),size(img1_o,2)]);
% % img1 = imresize(img1_o,scl1);
% img1 = img1_o;
% img2_o=imread([file_list{img_idx2}]);
% scl2 = 224/min([size(img2_o,1),size(img2_o,2)]);
% img2 = imresize(img2_o,scl2);


% 
% two_img_screen=zeros( max(size(img1,1),size(img2,1)) , size(img1,2)+size(img2,2)+10 , 3 );
% two_img_screen(1:size(img1,1),1:size(img1,2),:)=img1;
% two_img_screen(1:size(img2,1),...
% (size(img1,2)+10+1):(size(img1,2)+10+size(img2,2)), : )=img2;
% two_img_screen=uint8(two_img_screen);
% hold on;
% 
% for i=maximal_clique_q'
%     imshow(two_img_screen);
%     hold on;
%     %% parameters of loc_set: [ii, hi, wi, hi+Arf, wi+Arf]
%     ihi1=compare_vec_sort(i,3)-1;
%     iwi1=compare_vec_sort(i,4)-1;
%     hi1 = Astride * (ihi1 + offset) - Apad; %same
%     wi1 = Astride * (iwi1 + offset) - Apad; %same
%     x1=[wi1,wi1+Arf,wi1+Arf,wi1]; 
%     y1=[hi1,hi1,hi1+Arf,hi1+Arf];
%     patch(x1,y1,'r','FaceAlpha',0.35,'edgealpha',0);
%     hold on;
%     
%     ihi2=compare_vec_sort(i,6)-1;
%     iwi2=compare_vec_sort(i,7)-1;
%     hi2 = Astride * (ihi2 + offset) - Apad;  %same
%     wi2 = Astride * (iwi2 + offset) - Apad; %same
%     x2=[wi2,wi2+Arf,wi2+Arf,wi2]+size(img1,2)+10;
%     y2=[hi2,hi2,hi2+Arf,hi2+Arf];
%     patch(x2,y2,'r','FaceAlpha',0.35,'edgealpha',0);
% %   hold on;
% saveas(gcf,[make_filepath,'\img1_',num2str(img_idx1-1),'VSimg1_',num2str(img_idx2-1),...
%                                           '_order',num2str(i),'of',num2str(MCP_test_range),...
%                                           '_cos',num2str(compare_vec_sort(i,1)),...
%                                           '_h1',num2str(compare_vec_sort(i,3)),...
%                                           '_w1',num2str(compare_vec_sort(i,4)),...
%                                           '_h2',num2str(compare_vec_sort(i,6)),...
%                                           '_w2',num2str(compare_vec_sort(i,7)),...
%                                           '.jpg']);
% end

end
end
