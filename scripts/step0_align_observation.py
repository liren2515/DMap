import os, sys
import cv2
import numpy as np
import trimesh
import torch
from typing import Optional
from smplx.body_models import SMPLLayer
import argparse

sys.path.append('../')
from utils.mesh import apply_rotation
from utils.render import get_render

def perspective_projection(points: torch.Tensor,
                           translation: torch.Tensor,
                           focal_length: torch.Tensor,
                           camera_center: Optional[torch.Tensor] = None,
                           rotation: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the perspective projection of a set of 3D points.
    Args:
        points (torch.Tensor): Tensor of shape (B, N, 3) containing the input 3D points.
        translation (torch.Tensor): Tensor of shape (B, 3) containing the 3D camera translation.
        focal_length (torch.Tensor): Tensor of shape (B, 2) containing the focal length in pixels.
        camera_center (torch.Tensor): Tensor of shape (B, 2) containing the camera center in pixels.
        rotation (torch.Tensor): Tensor of shape (B, 3, 3) containing the camera rotation.
    Returns:
        torch.Tensor: Tensor of shape (B, N, 2) containing the projection of the input points.
    """
    batch_size = points.shape[0]
    if rotation is None:
        rotation = torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0).expand(batch_size, -1, -1)
    if camera_center is None:
        camera_center = torch.zeros(batch_size, 2, device=points.device, dtype=points.dtype)
    # Populate intrinsic camera matrix K.
    K = torch.zeros([batch_size, 3, 3], device=points.device, dtype=points.dtype)
    K[:,0,0] = focal_length[:,0]
    K[:,1,1] = focal_length[:,1]
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]

def get_scale_trans(points_src, points_tgt):
    points_src_stack = points_src.reshape(-1)
    stack0 = np.zeros((len(points_src_stack)))
    stack1 = np.zeros((len(points_src_stack)))
    stack0[::2] = 1
    stack1[1::2] = 1
    points_src_stack = np.stack((points_src_stack, stack0, stack1), axis=-1)
    points_tgt_stack = points_tgt.reshape(-1, 1)

    x = np.linalg.inv(points_src_stack.T@points_src_stack)@(points_src_stack.T)@points_tgt_stack
    warp_mat = np.float32([[x[0],0,x[1]],[0,x[0],x[2]]])
    return warp_mat

def align_image(image, warp_mat, size=512):
    image_new = np.zeros((size, size, 3)).astype(np.uint8)
    rows, cols, ch = image_new.shape
    image_new = cv2.warpAffine(image, warp_mat, (cols, rows), flags=cv2.INTER_NEAREST)
    return image_new


def camera_proj(jts, trans, render_res, P):

    camera_center = [render_res[0] / 2., render_res[1] / 2.]
    width = float(render_res[0])
    height = float(render_res[1])

    jts_new = (jts[0] + trans).cpu().detach().numpy()
    jts_new[:,1:] *= -1
    jts_new = np.concatenate((jts_new, np.ones_like(jts_new)[:,:1]), axis=-1)
    jts_new_p = P@jts_new.T
    jts_new_p=jts_new_p/jts_new_p[3]
    jts_new_p[0]=width/2*jts_new_p[0]+width/2         # transformation from [-1,1] -> [0,width]
    jts_new_p[1]=height - (height/2*jts_new_p[1]+height/2)
    jts_new_p = jts_new_p.T
    return jts_new_p[:,:2]



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_root', type=str, default='../observations/skirt')
    parser.add_argument('--seg_root', type=str, default='../observations/mask-skirt')
    parser.add_argument('--normal_root', type=str, default='../observations/normal-skirt')
    parser.add_argument('--smpl_root', type=str, default='../observations/smpl-skirt')
    parser.add_argument('--save_root', type=str, default='../data')
    parser.add_argument('--garment', type=str, default='Skirt')

    args = parser.parse_args()

    img_dir = args.img_dir
    seg_dir = args.seg_dir
    normal_dir = args.normal_dir
    smpl_dir = args.smpl_dir
    save_root = args.save_root
    garment = args.garment

    save_dir = os.path.join(save_root, garment)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    images_list = sorted(list(set([i.split('_')[0] for i in sorted(os.listdir(seg_dir))])))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    scale = 0.8 # 1
    IMAGE_SIZE = 256
    FL = 5000.

    smpl_model = SMPLLayer( model_path='../checkpoints/smpl/',
                            gender='neutral', 
                            use_face_contour=False,
                            num_betas=10,
                            num_expression_coeffs=10,
                            ext='npz',
                            num_pca_comps=12).cuda()

    _, _, transform = get_render(render_res=512, return_transform=True)

    for i in range(len(images_list)):

        img_name = images_list[i]

        crop = cv2.imread(os.path.join(image_dir, '%s.png'%img_name))
        mask_full = cv2.imread(os.path.join(seg_dir, '%s_mask_full.png'%img_name))
        w, h = crop.shape[:2]

        data = torch.load(os.path.join(smpl_dir, '%s_all.pt'%img_name))
        output = data['output']
        pred_cam_t_full = data['pred_cam_t_full'][:1]
        pred_cam_t_full = torch.FloatTensor(pred_cam_t_full).cuda()

        pred_smpl_params = output['pred_smpl_params']
        pred_keypoints_3d = output['pred_keypoints_3d'][:1]
        pred_keypoints_2d = output['pred_keypoints_2d'][:1]

        focal_length = FL * torch.ones(1, 2, device=pred_cam_t_full.device, dtype=pred_cam_t_full.dtype)
        focal_length = focal_length.reshape(-1, 2)
        scaled_focal_length = focal_length / IMAGE_SIZE * max(w, h)
        pred_cam_t_full = pred_cam_t_full.reshape(-1, 3)
        pred_keypoints_2d_full = perspective_projection(pred_keypoints_3d, 
                                                    translation=pred_cam_t_full,
                                                    focal_length=focal_length / IMAGE_SIZE)

        P = data['P']
        pred_keypoints_2d_full = camera_proj(pred_keypoints_3d, pred_cam_t_full, crop.shape[:2][::-1], P)
        pred_keypoints_2d_full = pred_keypoints_2d_full[:22].astype(int)

        global_orient = pred_smpl_params['global_orient'][:1]
        body_pose = pred_smpl_params['body_pose'][:1]
        betas = pred_smpl_params['betas'][:1]
        smpl_output = smpl_model(betas=betas,
                    global_orient=global_orient,
                    body_pose=body_pose,
                    pose2rot=False,
                    )

        verts = smpl_output['vertices']*scale
        joints = smpl_output['joints'][:,:22]*scale
        verts = verts - joints[:,[0]]
        joints = joints - joints[:,[0]]

        body = trimesh.Trimesh(verts[0].cpu().numpy(), smpl_model.faces)
        body = apply_rotation(np.pi, body, 'x')
        body.export(os.path.join(save_dir, '%s_body.ply'%img_name))
        joints[:,:,1:] *=-1


        joints_2d = transform.transform_points(joints)
        joints_2d = (-joints_2d[0,:,:2].cpu().numpy() + 1)/2*511
        
        warp_mat = get_scale_trans(pred_keypoints_2d_full, joints_2d)

        size = crop.shape[1]
        normal = cv2.imread(os.path.join(normal_dir, '%s.png'%img_name))[:,size:]
        seg = cv2.imread(os.path.join(seg_dir, '%s_segmentation.png'%img_name))
        
        crop_align = align_image(crop.copy(), warp_mat)
        normal_align = align_image(normal.copy(), warp_mat)
        seg_align = align_image(seg.copy(), warp_mat)
        mask_full_align = align_image(mask_full.copy(), warp_mat)
        cv2.imwrite(os.path.join(save_dir, '%s_crop_align.png'%img_name), crop_align)
        cv2.imwrite(os.path.join(save_dir, '%s_normal_align.png'%img_name), normal_align)
        cv2.imwrite(os.path.join(save_dir, '%s_seg_align.png'%img_name), seg_align)
        cv2.imwrite(os.path.join(save_dir, '%s_mask_full_align.png'%img_name), mask_full_align)
        #sys.exit()
