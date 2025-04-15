import os, sys
import cv2
import numpy as np
import trimesh
import torch
import torch.nn.functional as F
import argparse

import diffusers
from diffusers import DDPMScheduler

sys.path.append('../')
from utils.process import _to_xyz, _to_uv_FB, clean_uv, filter_faces, dilate_indicator
from utils.process import get_mask_label, fill_background_with_nearest_foreground
from utils.isp import create_uv_mesh, uv_to_3D, barycentric_faces

from diffusion.pipeline_ddpm_guided import DDPMPipeline

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def measure_func(model_output, observation, t=None):
    isp_mask_bool, isp_mask, sparse_mask, sparse_uv = observation

    output_uv = model_output[:,:3].permute(0,2,3,1)
    output_mask = model_output[:,3]
    loss_sparse_uv = torch.linalg.norm((output_uv[sparse_mask] - sparse_uv[sparse_mask]), dim=-1).mean()

    loss_mask = ((output_mask - isp_mask).abs()).mean()#*0
    #print(loss_sparse_uv.item(), loss_mask.item())

    loss = loss_sparse_uv + loss_mask
    return loss


def mask_to_coord(mask):
    x, y = np.where(mask)
    coord = np.stack((x,y), axis=-1)
    return coord
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_root', type=str, default='../checkpoints')
    parser.add_argument('--save_root', type=str, default='../fitting-results')
    parser.add_argument('--garment', type=str, default='Skirt')

    args = parser.parse_args()

    ckpt_root = args.ckpt_root
    save_root = args.save_root
    garment = args.garment

    ckpt_path = os.path.join(ckpt_root, 'shape-%s'%garment)
    save_folder = os.path.join(save_root, garment)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    ddpm_num_inference_steps = 1000
    pipeline = DDPMPipeline.from_pretrained(ckpt_path, use_safetensors=True).to(device)
    
    vertices_uv, faces_uv = create_uv_mesh(128, 128)

    images_list = sorted(list(set([f.split('-')[0] for f in os.listdir(save_folder)])))

    for i in range(0, len(images_list)):
        img_name = images_list[i]

        step1_path = os.path.join(save_folder, img_name)
        step2_path = os.path.join(save_folder, img_name+'-FB')
        step3_path = os.path.join(save_folder, img_name+'-isp')
        save_path = os.path.join(save_folder, img_name+'-inpainting')
        if not os.path.exists(save_path):
            os.makedirs(save_path)


        isp_mask = cv2.imread(os.path.join(step3_path, 'isp-fitting-%s.png'%(img_name)))[:, :, 0]/255
        isp_mask = cv2.resize(isp_mask.astype(float), (256,128), interpolation = cv2.INTER_AREA) >= 0.5
        isp_mask = dilate_indicator(isp_mask.astype(np.uint8), size=5)
        cv2.imwrite(os.path.join(save_path, 'isp_mask_resize_%s.png'%(img_name)), isp_mask*255)
        isp_mask_bool = torch.BoolTensor(isp_mask).cuda().unsqueeze(0)
        isp_mask = torch.FloatTensor((isp_mask.astype(float) - 0.5)/0.5).cuda().unsqueeze(0)

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
        cv2.imwrite(os.path.join(save_path, 'sparse_uv_%s.png'%(img_name)), ((sparse_uv+1)/2*255).astype(np.uint8))

        sparse_mask = torch.BoolTensor(sparse_mask).cuda().unsqueeze(0)
        sparse_uv = torch.FloatTensor(sparse_uv).cuda().unsqueeze(0)
        sparse_mask = torch.logical_and(sparse_mask, isp_mask_bool)

        
        # run pipeline in inference (sample random noise and denoise)
        observation = [isp_mask_bool, isp_mask, sparse_mask, sparse_uv]
        images = pipeline(
            measure_func=measure_func,
            observation=observation,
            guide_scale=40.0,#20.0,#40.0,
            generator=torch.Generator(device=pipeline.device).manual_seed(0),
            batch_size=1,
            num_inference_steps=ddpm_num_inference_steps,
            output_type="numpy",
        ).images

        np.save(os.path.join(save_path, 'uv-inpaint_%s.npy'%(img_name)), images[0])

        images_processed_uv = images[:,:,:,:3].copy()
        images_processed_mask = images[:,:,:,-1].copy()
        images_processed_mask[images_processed_mask>0.5] = 1
        images_processed_mask[images_processed_mask<0.5] = 0


        uv_f = images_processed_uv[0,:,:128]
        uv_b = images_processed_uv[0,:,128:]
        mask_f = images_processed_mask[0,:,:128]
        mask_b = images_processed_mask[0,:,128:]

        uv_f = fill_background_with_nearest_foreground(uv_f, mask_f)
        uv_b = fill_background_with_nearest_foreground(uv_b, mask_b)
        images_processed_uv = np.concatenate((uv_f, uv_b), axis=1).reshape(1, 128,256, 3)
        images[:,:,:,:3] = images_processed_uv
        np.save(os.path.join(save_path, 'uv-inpaint-filling_%s.npy'%(img_name)), images[0])
        faces_f = faces_uv
        faces_b = faces_uv

        uv_f = images_processed_uv[0, :,:128].reshape(-1, 3)
        uv_b = images_processed_uv[0, :,128:].reshape(-1, 3)

        images_processed_uv = (images_processed_uv / 2 + 0.5)
        images_processed_uv = (images_processed_uv[0] * 255).round().astype("uint8")
        cv2.imwrite(os.path.join(save_path, 'uv-inpaint_%s.png'%(img_name)), images_processed_uv)
        cv2.imwrite(os.path.join(save_path, 'uv-inpaint-mask_%s.png'%(img_name)), (images_processed_mask[0]* 255).astype("uint8"))


        ##########
        # baricentric mapping
        pattern_f = trimesh.load(os.path.join(step3_path, 'pattern-f-%s.ply'%(img_name)), validate=False, process=False)
        pattern_b = trimesh.load(os.path.join(step3_path, 'pattern-b-%s.ply'%(img_name)), validate=False, process=False)
        sewing = trimesh.load(os.path.join(step3_path, 'sewing-%s.ply'%(img_name)), validate=False, process=False)

        pattern_f_128 = trimesh.Trimesh(vertices_uv, faces_uv, valid=False, process=False)
        pattern_b_128 = trimesh.Trimesh(vertices_uv, faces_uv, valid=False, process=False)
        v_barycentric_f, closest_face_idx_f = barycentric_faces(pattern_f, pattern_f_128)
        v_barycentric_b, closest_face_idx_b = barycentric_faces(pattern_b, pattern_b_128)

        verts_f = uv_to_3D(uv_f, faces_f, v_barycentric_f, closest_face_idx_f)
        verts_b = uv_to_3D(uv_b, faces_b, v_barycentric_b, closest_face_idx_b)
        verts = np.concatenate((verts_f, verts_b), axis=0)

        deform = trimesh.Trimesh(verts, sewing.faces, validate=False, process=False)
        deform_f = trimesh.Trimesh(verts_f, pattern_f.faces, validate=False, process=False)
        deform_b = trimesh.Trimesh(verts_b, pattern_b.faces, validate=False, process=False)
        deform.export(os.path.join(save_path, 'deform-inpaint-%s.ply'%(img_name)))
        deform_f.export(os.path.join(save_path, 'deform-f-inpaint-%s.ply'%(img_name)))
        deform_b.export(os.path.join(save_path, 'deform-b-inpaint-%s.ply'%(img_name)))

        np.savez(os.path.join(save_path, 'barycentric-%s'%(img_name)), v_barycentric_f=v_barycentric_f, v_barycentric_b=v_barycentric_b, closest_face_idx_f=closest_face_idx_f, closest_face_idx_b=closest_face_idx_b, faces_f=faces_f, faces_b=faces_b)
            