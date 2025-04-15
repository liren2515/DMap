import os, sys
import trimesh
import numpy as np
import torch
from trimesh.transformations import rotation_matrix

def apply_rotation(alpha, mesh, which_axis):
    if which_axis == 'x':
        axis = [1, 0, 0]
    elif which_axis == 'y':
        axis = [0, 1, 0]
    elif which_axis == 'z':
        axis = [0, 0, 1]
    else:
        print('Wrong input for the rotation axis')
        raise NotImplementedError

    R = rotation_matrix(alpha, axis)
    mesh.apply_transform(R)
    return mesh

def rotate_pose(pose, angle, which_axis='x'):
    # pose: (n, 72)
    
    from scipy.spatial.transform import Rotation as R
    is_tensor = torch.is_tensor(pose)
    if is_tensor:
        pose = pose.cpu().detach().numpy()
    
    swap_rotation = R.from_euler(which_axis, [angle/np.pi*180], degrees=True)
    root_rot = R.from_rotvec(pose[:, :3])
    pose[:, :3] = (swap_rotation * root_rot).as_rotvec()

    if is_tensor:
        pose = torch.FloatTensor(pose).cuda()

    return pose