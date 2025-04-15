import os, sys
import cv2
import numpy as np
import trimesh
import torch
import argparse

sys.path.append('../')
from utils.isp import create_uv_mesh, get_barycentric
from utils.cutting import get_connected_paths_skirt
from utils.mesh import apply_rotation
from utils.render import get_render
from utils.process import dilate_indicator, _to_xyz, _to_uv_FB, fill_background_with_nearest_foreground, get_mask_label
from utils.process import remove_arm, project_waist
from utils.process import rescale, clean_pt
from snug.snug_class import Cloth_from_NP, Material
from utils.optimization_cloth import align_observation_uv, align_observation_pt_verts, remesh


def mask_to_coord(mask):
    x, y = np.where(mask)
    coord = np.stack((x,y), axis=-1)
    return coord


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

    ckpt_path = os.path.join(ckpt_root, 'shape-%s'%garment)
    data_path = os.path.join(data_root, garment)
    save_folder = os.path.join(save_root, garment)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    target_label = get_mask_label(garment)

    use_double_cd = False

    _, renderer_textured_soft, transform = get_render(render_res=512, is_soft=True, return_transform=True)
    raster, _ = get_render(render_res=512, is_soft=False)

    renders = [renderer_textured_soft, transform, raster]

    color_smpl = np.load('../extra-data/color_smpl_faces.npy')
    faces_id_no_arm = remove_arm(color_smpl.astype(int))

    x_res = y_res = 128
    uv_vertices, uv_faces = create_uv_mesh(x_res, y_res, debug=False)

    images_list = sorted(list(set([f.split('-')[0] for f in os.listdir(save_folder)])))
    for i in range(len(images_list)):

        img_name = images_list[i]

        step1_path = os.path.join(save_folder, img_name)
        step2_path = os.path.join(save_folder, img_name+'-FB')
        step3_path = os.path.join(save_folder, img_name+'-isp')
        step4_path = os.path.join(save_folder, img_name+'-inpainting')

        save_path = os.path.join(save_folder, img_name+'-post')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        normal_front = cv2.imread(os.path.join(data_path, '%s_normal_align.png'%img_name))[:,:,::-1].copy()
        seg = cv2.imread(os.path.join(data_path, '%s_seg_align.png'%img_name))[:,:,0]
        mask_target = ((seg == target_label).astype(np.uint8))
        mask_other = (((seg == 60) + (seg == 120) + (seg == 180) + (seg == 240)).astype(np.uint8)) - mask_target
        
        body_mesh = trimesh.load(os.path.join(step2_path, '%s_body_opt.ply'%(img_name)))
        clothed_mesh = body_mesh

        mask_back = cv2.imread(os.path.join(step1_path, 'images_mask_back_%s.png'%(img_name)))[:,:,0]
        normal_back = cv2.imread(os.path.join(step1_path, 'images_normal_back_%s.png'%(img_name)))[:,:,::-1].copy()
        normal_back = cv2.resize(normal_back, (512, 512))
        mask_back = (cv2.resize(mask_back, (512, 512)) > 122).astype(np.uint8)

        masks = [mask_target, mask_back, mask_other]
        normals = [normal_front, normal_back]

        prediction = np.load(os.path.join(step2_path, 'uv_transfer_%s.npz'%(img_name)))
        data_transfer = prediction['uv_transfer'] # [-1, 1]
        depth_front = data_transfer[:,:,3]
        depth_back = data_transfer[:,:,7]
        mask_depth_front = cv2.imread(os.path.join(step2_path, 'images_mask_front_%s.png'%img_name))[:,:,0]/255
        mask_depth_back = cv2.imread(os.path.join(step1_path, 'images_mask_back_%s.png'%(img_name)))[:,:,0]/255
        depth_front = cv2.resize(depth_front, (512, 512))
        depth_back = cv2.resize(depth_back, (512, 512))
        mask_depth_front = (cv2.resize(mask_depth_front*255, (512, 512)) == 255).astype(np.uint8)
        mask_depth_back = (cv2.resize(mask_depth_back*255, (512, 512)) == 255).astype(np.uint8)
        coord_img_f = mask_to_coord(mask_depth_front)
        coord_img_b = mask_to_coord(mask_depth_back)
        
        xyz = trimesh.load(os.path.join(step2_path, 'xyz_%s.ply'%(img_name)))
        z_f = depth_front[coord_img_f[:,0], coord_img_f[:,1]].reshape(-1)
        z_b = depth_back[coord_img_b[:,0], coord_img_b[:,1]].reshape(-1)
        xyz_f = _to_xyz(coord_img_f, z_f, img_size=511.).astype(np.float32)
        xyz = np.concatenate((xyz_f, xyz.vertices), axis=0)
        xyz = trimesh.PointCloud(xyz)
        xyz, idx = clean_pt(xyz, nb_neighbors=10, std_ratio=2)

        xyz = xyz.vertices
        xyz = torch.FloatTensor(xyz).cuda().unsqueeze(0)
        

        barycentric = np.load(os.path.join(step4_path, 'barycentric-%s.npz'%(img_name)))
        v_barycentric_f = barycentric['v_barycentric_f']
        v_barycentric_b = barycentric['v_barycentric_b']
        closest_face_idx_f = barycentric['closest_face_idx_f']
        closest_face_idx_b = barycentric['closest_face_idx_b']
        faces_f = barycentric['faces_f']
        faces_b = barycentric['faces_b']


        cloth_rest = trimesh.load(os.path.join(step3_path, 'sewing-%s.ply'%(img_name)), validate=False, process=False)
        altas_f = trimesh.load(os.path.join(step3_path, 'atlas-f-%s.ply'%(img_name)), validate=False, process=False)
        altas_b = trimesh.load(os.path.join(step3_path, 'atlas-b-%s.ply'%(img_name)), validate=False, process=False)
        pattern_f = trimesh.load(os.path.join(step3_path, 'pattern-f-%s.ply'%(img_name)), validate=False, process=False)
        pattern_b = trimesh.load(os.path.join(step3_path, 'pattern-b-%s.ply'%(img_name)), validate=False, process=False)
        num_v_f = len(pattern_f.vertices)

        cloth_rest_z_up = apply_rotation(np.pi/2, cloth_rest.copy(), 'x')
        waist_v_id = get_connected_paths_skirt(cloth_rest_z_up)[0]

        barycentric_front, idx_faces_front = get_barycentric(pattern_f, uv_vertices)
        barycentric_back, idx_faces_back = get_barycentric(pattern_b, uv_vertices)
        faces_front = pattern_f.faces
        faces_back = pattern_b.faces + num_v_f
        
        triangles_front = cloth_rest.vertices[faces_front[idx_faces_front]]
        triangles_back = cloth_rest.vertices[faces_back[idx_faces_back]]
        bary_f = (triangles_front * barycentric_front[:, :, None]).sum(axis=-2)
        bary_b = (triangles_back * barycentric_back[:, :, None]).sum(axis=-2)
        uv_size = 128
        bary_f = bary_f.reshape(uv_size, uv_size, 3)
        bary_b = bary_b.reshape(uv_size, uv_size, 3)
        image_rest = np.concatenate((bary_f, bary_b), axis=1)
        image_rest = torch.FloatTensor(image_rest).unsqueeze(0).permute(0,3,1,2).cuda()

        
        ############### inpaint uv ###############
        uv_inpaint = np.load(os.path.join(step4_path, 'uv-inpaint-filling_%s.npy'%(img_name)))
        image = np.transpose(uv_inpaint, (2,0,1))
        image = torch.FloatTensor(image).cuda().unsqueeze(0)

        ############### sparse uv ###############
        prediction = np.load(os.path.join(step2_path, 'uv_transfer_%s.npz'%(img_name)))
        data_transfer = prediction['uv_transfer'] # [-1, 1]
        uv_transfer_f = data_transfer[:,:,:3]
        uv_transfer_b = data_transfer[:,:,4:4+3]
        depth_transfer_f = data_transfer[:,:,3]
        depth_transfer_b = data_transfer[:,:,7]

        mask_cloth_f = cv2.imread(os.path.join(step2_path, 'images_mask_front_%s.png'%img_name))[:,:,0]
        mask_cloth_b = cv2.imread(os.path.join(step1_path, 'images_mask_back_%s.png'%(img_name)))[:,:,0]
        coord_img_f = mask_to_coord(mask_cloth_f)
        coord_img_b = mask_to_coord(mask_cloth_b)

        z_f = depth_transfer_f[coord_img_f[:,0], coord_img_f[:,1]].reshape(-1)
        z_b = depth_transfer_b[coord_img_b[:,0], coord_img_b[:,1]].reshape(-1)
        xyz_f = _to_xyz(coord_img_f, z_f, img_size=191.).astype(np.float32)
        xyz_b = _to_xyz(coord_img_b, z_b, img_size=191.).astype(np.float32)

        sparse_uv, sparse_mask = _to_uv_FB(garment, uv_transfer_f, uv_transfer_b, coord_img_f, coord_img_b, xyz_f, xyz_b, size_uv=128)
        sparse_uv = torch.FloatTensor(sparse_uv).cuda().unsqueeze(0)
        sparse_mask = torch.BoolTensor(sparse_mask).cuda().unsqueeze(0)

        cloth_pose = trimesh.load(os.path.join(step4_path, 'deform-inpaint-%s.ply'%(img_name)), validate=False, process=False)
        cloth_pose_f = trimesh.load(os.path.join(step4_path, 'deform-f-inpaint-%s.ply'%(img_name)), validate=False, process=False)
        cloth_pose_b = trimesh.load(os.path.join(step4_path, 'deform-b-inpaint-%s.ply'%(img_name)), validate=False, process=False)

        scale = rescale(cloth_pose_f, cloth_pose_b, altas_f, altas_b)
        
        material = Material()
        cloth_state = Cloth_from_NP(cloth_rest.vertices*scale, cloth_rest.faces, material)
        np.savez(os.path.join(save_path, 'cloth_state_%s'%(img_name)), vertices=cloth_rest.vertices*scale, faces=cloth_rest.faces)

        mapping_related = [faces_f, faces_b, v_barycentric_f, v_barycentric_b, closest_face_idx_f, closest_face_idx_b, sparse_uv, sparse_mask, xyz]
        
        
        cloth_pose_uv, img_mask = align_observation_uv(renders, image, cloth_pose, cloth_state, mapping_related, image_rest, masks, normals, body_mesh, clothed_mesh, waist_v_id, use_double_cd=use_double_cd)
        cloth_pose_uv.export(os.path.join(save_path, 'mesh-uv-%s.ply'%(img_name)))
        
        
        cloth_pose_pt = trimesh.load(os.path.join(save_path, 'mesh-uv-%s.ply'%(img_name)), process=False, validate=False)

        body_mesh_no_arm = body_mesh.copy()
        body_mesh_no_arm.faces = body_mesh_no_arm.faces[faces_id_no_arm]
        barycentric_waist, idx_f_waist = get_barycentric(body_mesh_no_arm, cloth_pose_pt.vertices[waist_v_id])
        vertices_waist = project_waist(body_mesh_no_arm, barycentric_waist, idx_f_waist)

        cloth_pose_pt_new, img_mask = align_observation_pt_verts(renders, cloth_pose_pt, cloth_state, body_mesh, clothed_mesh, masks, normals, vertices_waist, waist_v_id, xyz, use_double_cd=use_double_cd)
        cloth_pose_pt_new.export(os.path.join(save_path, 'mesh_pt_verts_%s.ply'%(img_name)))

        cloth_pose_pt = trimesh.load(os.path.join(save_path, 'mesh_pt_verts_%s.ply'%(img_name)), process=False, validate=False)
        cloth_pose_remseh = remesh(cloth_pose_pt, cloth_state, body_mesh)
        cloth_pose_remseh.export(os.path.join(save_path, 'mesh_remesh_%s.ply'%(img_name)))
        
    
