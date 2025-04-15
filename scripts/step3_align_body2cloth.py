import os, sys
import cv2
import numpy as np
import trimesh
import torch
import torch.nn.functional as F
from torchgeometry import rotation_matrix_to_angle_axis
import argparse

from smplx.body_models import SMPLLayer
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

sys.path.append('../')
from utils.process import get_mask_label, fill_background_with_nearest_foreground
from utils.render import get_render
from utils.mesh import apply_rotation
from utils.rasterize import get_pix_to_face_v2

from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
from snug.snug_helper import collision_penalty


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def flip_verts(verts):
    sign = torch.ones_like(verts)
    sign[:,:,1:] *=-1
    verts = verts*sign
    return verts

def fit_body(body_params, mask_full, mask_cloth, depth_cloth, joints_2d_reg, pt_cloth, raster, transform, renderer, smpl_model):

    idx_x, idx_y = np.where(mask_full>0.5)
    idx_mask = np.stack((idx_x, idx_y), axis=-1).astype(float)
    idx_mask = torch.FloatTensor(idx_mask).cuda()
    idx_mask_pend = torch.cat((idx_mask, torch.zeros(idx_mask.shape[0], 1).cuda()), dim=-1)

    depth_cloth_f, depth_cloth_b = depth_cloth
    depth_cloth_f = torch.FloatTensor(depth_cloth_f).cuda()
    depth_cloth_b = torch.FloatTensor(depth_cloth_b).cuda()
    mask_cloth = torch.FloatTensor(mask_cloth).cuda()
    joints_2d_reg = torch.FloatTensor(joints_2d_reg).cuda()

    verts_cloth = torch.FloatTensor(pt_cloth.vertices).cuda()
    
    faces_body = torch.LongTensor(smpl_model.faces.astype(int)).cuda()

    trans = torch.zeros(3).cuda()
    pose, betas = body_params
    pose.requires_grad = True
    betas.requires_grad = True
    trans.requires_grad = True
    lr = 5e-3
    eps = 2e-3
    optimizer = torch.optim.Adam([{'params': pose, 'lr': lr*0.05},
                                  {'params': betas, 'lr': lr},
                                  {'params': trans, 'lr': lr},
                                  ])

    with torch.no_grad():
        verts_zero = torch.zeros(6890, 3).cuda()
        smpl_rgb = torch.zeros(6890, 3) + 255 # (1, V, 3)
        verts_rgb = smpl_rgb[None]
        textures = TexturesVertex(verts_features=verts_rgb.cuda())
        mask_full = torch.FloatTensor(mask_full).cuda()

    iters = 300
    for i in range(iters):

        rotmat_hom = torch.cat([pose.reshape(-1, 3, 3), torch.FloatTensor([0,0,1]).cuda().reshape(1, 3, 1).expand(24, -1, -1)], dim=-1)
        pose_euler = rotation_matrix_to_angle_axis(rotmat_hom).reshape(1, -1)

        body_poZ = vp.encode(pose_euler[:,3:66]).mean

        global_orient = pose[:,:1]
        body_pose = pose[:,1:]

        smpl_output = smpl_model(betas=betas,
                    global_orient=global_orient,
                    body_pose=body_pose,
                    pose2rot=False,
                    )

        verts = smpl_output['vertices']*scale_depth
        joints = smpl_output['joints']*scale_depth
        verts = verts - joints[:,[0]]
        joints = joints - joints[:,[0]]

        verts = flip_verts(verts).squeeze()
        joints = flip_verts(joints).squeeze()

        verts = verts + trans[None, :]
        joints = joints + trans[None, :]
        
        joints_2d = transform.transform_points(joints)
        joints_2d = (-joints_2d[:,:2] + 1)/2
        joints_2d = joints_2d[keep_jt]

        loss_jt_2d = ((joints_2d - joints_2d_reg)**2).sum(dim=-1).mean()

        
        with torch.no_grad():
            idx_faces_f, idx_vertices_f = get_pix_to_face_v2(verts, faces_body, raster)
            verts_back = verts.clone().detach()
            verts_back[:, -1] *= -1
            idx_faces_b, idx_vertices_b = get_pix_to_face_v2(verts_back, faces_body, raster)
            faces_f = faces_body[idx_faces_f]
            faces_b = faces_body[idx_faces_b]
            is_torsor_f = faces_id_torsor[idx_faces_f]
            is_torsor_b = faces_id_torsor[idx_faces_b]
        tri_f = verts[faces_f.reshape(-1)].reshape(-1,3,3)
        tri_b = verts[faces_b.reshape(-1)].reshape(-1,3,3)
        tri_center_f = tri_f.mean(dim=1)
        tri_center_b = tri_b.mean(dim=1)
        tri_center_f_z = tri_center_f[:, -1]
        tri_center_b_z = tri_center_b[:, -1]*(-1)

        with torch.no_grad():
            verts_2D_f = (transform.transform_points(tri_center_f.unsqueeze(0))[:,:,[1,0]]*(-255.5)) + 255.5 # x,y
            verts_2D_b = (transform.transform_points(tri_center_b.unsqueeze(0))[:,:,[1,0]]*(-255.5)) + 255.5 # x,y
            verts_2D_f = torch.clamp(verts_2D_f, 0, 511).long().squeeze()
            verts_2D_b = torch.clamp(verts_2D_b, 0, 511).long().squeeze()

            z_constrain_f = depth_cloth_f[verts_2D_f[:,0],verts_2D_f[:,1]]
            z_constrain_b = depth_cloth_b[verts_2D_b[:,0],verts_2D_b[:,1]]

        loss_depth_f = F.relu(tri_center_f_z - z_constrain_f + eps).mean()
        loss_depth_b = F.relu(tri_center_b_z - z_constrain_b + eps).mean()
        loss_depth = (loss_depth_f + loss_depth_b)*10

        is_torsor_f = torch.logical_and(is_torsor_f, z_constrain_f != 1) 
        is_torsor_b = torch.logical_and(is_torsor_b, z_constrain_b != 1) 
        loss_close_f = (tri_center_f_z - z_constrain_f)[is_torsor_f].abs().mean()
        loss_close_b = (tri_center_b_z - z_constrain_b)[is_torsor_b].abs().mean()
        loss_close = (loss_close_f + loss_close_b)
        
        
        mesh = Meshes(
            verts=[verts_zero],   
            faces=[faces_body],
            textures=textures
        )
        new_src_mesh = mesh.offset_verts(verts)
        images_predicted = renderer(new_src_mesh)
        images_pred = images_predicted[0, :, :, 3]
        images_pred = torch.clamp(images_pred + mask_cloth, 0, 1)
        
        intersection = (images_pred*mask_full).sum()
        union = images_pred.sum() + mask_full.sum() - intersection
        loss_mask = (1 - intersection/union)

        tri = verts[faces_body.reshape(-1)].reshape(-1,3,3)
        vectors = tri[:,1:] - tri[:,:2]
        nb = torch.cross(vectors[:, 0], vectors[:, 1], dim=-1)
        nb = nb/nb.norm(p=2, dim=-1, keepdim=True)
        tri = tri.mean(dim=1)
        loss_collision = collision_penalty(verts_cloth.unsqueeze(0), tri.unsqueeze(0), nb.unsqueeze(0), eps=eps)/10/5
        loss_beta = torch.pow(betas, 2).sum()*0.001
        loss_z = torch.pow(body_poZ, 2).sum()

        loss = loss_jt_2d + loss_mask + loss_depth + loss_collision + loss_beta + loss_z*0.001 + loss_close
        print('iter: %3d, loss: %0.4f, loss_jt_2d: %0.4f, loss_mask: %0.4f, loss_depth: %0.4f, loss_collision: %0.4f, loss_beta: %0.4f, loss_z: %0.4f, loss_close: %0.4f'%(i, loss.item(), loss_jt_2d.item(), loss_mask.item(), loss_depth.item(), loss_collision.item(), loss_beta.item(), loss_z.item(), loss_close.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    with torch.no_grad():
        global_orient = pose[:,:1]
        body_pose = pose[:,1:]

        smpl_output = smpl_model(betas=betas,
                    global_orient=global_orient,
                    body_pose=body_pose,
                    pose2rot=False,
                    )

        verts = smpl_output['vertices']
        joints = smpl_output['joints']
        verts = verts - joints[:,[0]]

        body = trimesh.Trimesh(verts[0].cpu().numpy(), smpl_model.faces)
        body = apply_rotation(np.pi, body, 'x')
        body.vertices += trans[None, :].cpu().detach().numpy()

    return body, pose.detach(), betas.detach(), trans.detach()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_root', type=str, default='../checkpoints')
    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--save_root', type=str, default='../fitting-results')
    parser.add_argument('--garment', type=str, default='Skirt')

    args = parser.parse_args()

    ckpt_root = args.ckpt_root
    data_root = args.data_root
    save_root = args.save_root
    garment = args.garment

    ckpt_path = os.path.join(ckpt_root, 'mapper-%s'%garment)
    data_path = os.path.join(data_root, garment)
    data_smpl_path = os.path.join(data_root, 'smpl-%s'%garment)
    save_folder = os.path.join(save_root, garment)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    scale_depth = 0.8 if garment == 'Skirt' else 1

    images_list = sorted(list(set([f.split('_')[0] for f in os.listdir(data_path)])))


    expr_dir = os.path.join('../human_body_prior/', 'V02_05')
    vp, ps = load_model(expr_dir, model_code=VPoser,
                                remove_words_in_model_weights='vp_model.',
                                disable_grad=True)
    vp = vp.to('cuda')

    smpl_model = SMPLLayer( model_path='../checkpoints/smpl/',
                            gender='neutral', 
                            use_face_contour=False,
                            num_betas=10,
                            num_expression_coeffs=10,
                            ext='npz',
                            num_pca_comps=12).cuda()

    raster, renderer_textured_soft, transform = get_render(render_res=512, is_soft=True, return_transform=True)


    color_smpl = np.load('../extra-data/color_smpl_faces.npy')
    faces_id_torsor = (color_smpl[:, 0] == 1) + (color_smpl[:, 0] == 2)
    faces_id_torsor = torch.BoolTensor(faces_id_torsor).cuda()
    target_label = get_mask_label(garment)

    keep_jt = [i for i in range(22)] + [i for i in range(24, 29)]


    for i in range(len(images_list)):
        img_name = images_list[i]

        step1_path = os.path.join(save_folder, img_name)
        step2_path = os.path.join(save_folder, img_name+'-FB')

        mask_full = cv2.imread(os.path.join(data_path, '%s_mask_full_align.png'%img_name))[:,:,0]/255

        seg = cv2.imread(os.path.join(data_path, '%s_seg_align.png'%img_name))[:,:,0]
        mask_cloth = (((seg == 60) + (seg == 120) + (seg == 180) + (seg == 240)).astype(np.uint8))

        pt_cloth = trimesh.load(os.path.join(step2_path, 'xyz_%s.ply'%(img_name)))
        pt_cloth.vertices /= scale_depth

        images_uv = np.load(os.path.join(step2_path, 'uv_transfer_%s.npz'%(img_name)))['uv_transfer']
        images_uv_f = images_uv[:,:,:4]
        images_uv_b = images_uv[:,:,4:]
        depth_cloth_f = images_uv_f[:,:,-1]/scale_depth
        depth_cloth_b = images_uv_b[:,:,-1]/scale_depth
        depth_mask_f = (cv2.imread(os.path.join(step2_path, 'images_normal_front_%s.png'%(img_name))) == 0).sum(axis=-1) != 3
        depth_mask_b = cv2.imread(os.path.join(step1_path, 'images_mask_back_%s.png'%(img_name)))[:,:,0] == 255
        depth_mask_b = np.logical_and(depth_mask_b, depth_cloth_b<0)

        depth_cloth_f = fill_background_with_nearest_foreground(depth_cloth_f, depth_mask_f)
        depth_cloth_b = fill_background_with_nearest_foreground(depth_cloth_b, depth_mask_b)*(-1)

        if scale_depth == 0.8:
            depth_cloth_f = cv2.resize(depth_cloth_f, (640, 640))[64:-64, 64:-64]#, interpolation=cv2.INTER_NEAREST)
            depth_cloth_b = cv2.resize(depth_cloth_b, (640, 640))[64:-64, 64:-64]#, interpolation=cv2.INTER_NEAREST)
            depth_mask_f = cv2.resize(depth_mask_f.astype(np.uint8), (640, 640))[64:-64, 64:-64] == 1 
            depth_mask_b = cv2.resize(depth_mask_b.astype(np.uint8), (640, 640))[64:-64, 64:-64] == 1 
        elif scale_depth == 1:
            depth_cloth_f = cv2.resize(depth_cloth_f, (512, 512))#, interpolation=cv2.INTER_NEAREST)
            depth_cloth_b = cv2.resize(depth_cloth_b, (512, 512))#, interpolation=cv2.INTER_NEAREST)
            depth_mask_f = cv2.resize(depth_mask_f.astype(np.uint8), (512, 512)) == 1 
            depth_mask_b = cv2.resize(depth_mask_b.astype(np.uint8), (512, 512)) == 1

        depth_cloth_f[~depth_mask_f] = 1
        depth_cloth_b[~depth_mask_b] = 1

        depth_cloth_f = depth_cloth_f.copy()
        depth_cloth_b = depth_cloth_b.copy()


        data_smpl = torch.load(os.path.join(data_smpl_path, '%s_all.pt'%img_name))
        output = data_smpl['output']
        pred_smpl_params = output['pred_smpl_params']
        global_orient = pred_smpl_params['global_orient'][:1]
        body_pose = pred_smpl_params['body_pose'][:1]
        betas = pred_smpl_params['betas'].cuda()[:1]
        pose = torch.cat((global_orient, body_pose), dim=1).cuda()

        smpl_output = smpl_model(betas=betas,
                    global_orient=global_orient,
                    body_pose=body_pose,
                    pose2rot=False,
                    )

        verts = smpl_output['vertices']*scale_depth
        joints = smpl_output['joints']*scale_depth
        verts = verts - joints[:,[0]]
        joints = joints - joints[:,[0]]
        joints[:,:,1:] *=-1

        joints_2d = transform.transform_points(joints)
        joints_2d_reg = (-joints_2d[0,:,:2].cpu().numpy() + 1)/2
        joints_2d_reg = joints_2d_reg[keep_jt]

        depths = [depth_cloth_f, depth_cloth_b]

        body, pose, beta, trans = fit_body([pose, betas], mask_full, mask_cloth, depths, joints_2d_reg, pt_cloth, raster, transform, renderer_textured_soft, smpl_model)

        body.vertices *= scale_depth
        body.export(os.path.join(step2_path, '%s_body_opt.ply'%(img_name)))

