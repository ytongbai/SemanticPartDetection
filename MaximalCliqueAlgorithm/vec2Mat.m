function [idx_r,idx_c]=vec2Mat(idx_in_vec,mat_h,mat_w)
idx_c=ceil(idx_in_vec/mat_h);
idx_r=mod(idx_in_vec-1,mat_h)+1;

%% Tips:
% Y = ceil(X) rounds each element of X to the nearest integer greater than or equal to that element.
% It is a small tool to change vector back to matrix, using the default method of MATLAB.

