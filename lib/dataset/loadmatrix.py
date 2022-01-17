from scipy.io import loadmat, savemat
import json_tricks as json
import os
# file = os.path.join('C:\Users\msi\Downloads', 'gt_valid.mat')
x = loadmat(r'C:\Users\msi\Downloads\gt_valid.mat')
y = loadmat(r'C:\Users\msi\Downloads\mpii_human_pose_v1_u12_2\mpii_human_pose_v1_u12_2\mpii_human_pose_v1_u12_1.mat')
with open(r'C:\Users\msi\Downloads\test.json', 'r') as f:
    all_boxes = json.load(f)
print('hodA')