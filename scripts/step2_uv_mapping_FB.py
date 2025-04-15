import os, sys
import cv2
import numpy as np
import trimesh
import torch
import torch.nn.functional as F
import argparse

sys.path.append('../')
from diffusion.pipeline_ddpm_condition import DDPMPipeline
from utils.process import _process_depth, _process_seg, _process_image, get_mask_label
from utils.process import draw_uv_mapping, mask_to_coord, _to_xyz, remove_arm
from utils.render import get_render, render_segmentation, render_depth, render_torsor, render_depth_discrete


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def _measure_func_close_body(model_output, observation, t, eps=0):
        
    mask_body_f, mask_body_b, depth_f, depth_b = observation
    output_depth_f = model_output[:,3]
    output_depth_b = model_output[:,7]
    
    if t < 200:
        loss_f = F.relu(depth_f[mask_body_f] - output_depth_f[mask_body_f] + eps).mean()
        loss_b = F.relu(output_depth_b[mask_body_b] - depth_b[mask_body_b] + eps).mean()
        loss_depth = loss_f + loss_b
        loss_body_f = (depth_f[mask_body_f] - output_depth_f[mask_body_f]).abs().mean()
        loss_body_b = (depth_b[mask_body_b] - output_depth_b[mask_body_b]).abs().mean()
        loss_body = (loss_body_f + loss_body_b)/100*2*1.5
    else:
        loss_f = F.relu(depth_f[mask_body_f] - output_depth_f[mask_body_f] + eps).mean()
        loss_b = F.relu(output_depth_b[mask_body_b] - depth_b[mask_body_b] + eps).mean()
        loss_depth = (loss_f + loss_b)*0
        loss_body_f = (depth_f[mask_body_f] - output_depth_f[mask_body_f]).abs().mean()
        loss_body_b = (depth_b[mask_body_b] - output_depth_b[mask_body_b]).abs().mean()
        loss_body = (loss_body_f + loss_body_b)*0
    
    loss = (loss_depth + loss_body)
    #print('loss_f: %0.4f, loss_b: %0.4f, loss_body_f: %0.4f, loss_body_b: %0.4f '%(loss_f.item(), loss_b.item(), loss_body_f.item(), loss_body_b.item()))
    return loss

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
    save_folder = os.path.join(save_root, garment)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    use_guidance = True
    ddpm_num_inference_steps = 1000
    measure_func = _measure_func_close_body

    target_label = get_mask_label(garment)
    pipeline = DDPMPipeline.from_pretrained(ckpt_path, use_safetensors=True).to(device)
    raster, renderer_textured_hard = get_render()
    raster_back, renderer_textured_hard_back = get_render(is_back=True)
    color_smpl_raw = np.load('../extra-data/color_smpl_faces.npy')
    color_smpl = np.load('../extra-data/color_smpl_faces.npy')/15

    faces_noArm = remove_arm(color_smpl_raw)

    images_list = sorted(list(set([f.split('_')[0] for f in os.listdir(data_path)])))

    for i in range(len(images_list)):
        img_name = images_list[i]

        load_path = os.path.join(save_folder, img_name)
        save_path = os.path.join(save_folder, img_name+'-FB')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        normal = cv2.imread(os.path.join(data_path, '%s_normal_align.png'%img_name))
        seg = cv2.imread(os.path.join(data_path, '%s_seg_align.png'%img_name))[:,:,0]
        mask = ((seg == target_label).astype(np.uint8))*255
        normal[seg != target_label] = 0
        cv2.imwrite(os.path.join(save_path, 'mask_%s.png'%img_name), mask.astype(np.uint8))

        mask_back = cv2.imread(os.path.join(load_path, 'images_mask_back_%s.png'%(img_name)))[:,:,0]/255
        normal_back = cv2.imread(os.path.join(load_path, 'images_normal_back_%s.png'%(img_name)))
        normal_back = normal_back.astype(np.float32)/255
        normal_back = (normal_back - 0.5)/0.5
        #normal_back = normal_back/np.linalg.norm(normal_back, keepdims=True, axis=-1)
        normal_back[mask_back==0] = -1
        
        body_smpl = trimesh.load(os.path.join(data_path, '%s_body.ply'%img_name))
        body_smpl_noArm = trimesh.Trimesh(body_smpl.vertices, body_smpl.faces[faces_noArm])
        body_smpl.export(os.path.join(save_path, 'body_%s.ply'%img_name))

        body_seg = render_segmentation(body_smpl, renderer_textured_hard, raster, color_smpl)
        body_seg_back = render_segmentation(body_smpl, renderer_textured_hard_back, raster_back, color_smpl)
        body_seg_back = np.fliplr(body_seg_back)
        cv2.imwrite(os.path.join(save_path, 'body_seg_%s.png'%img_name), body_seg)
        cv2.imwrite(os.path.join(save_path, 'body_seg_back_%s.png'%img_name), body_seg_back)

        body_depth = render_depth(body_smpl, renderer_textured_hard)
        body_depth_back = render_depth(body_smpl, renderer_textured_hard_back, flip_bg=False)
        body_depth_back = np.fliplr(body_depth_back)

        if garment == 'Tshirt' or garment == 'Jacket':
            body_depth_raw = render_depth_discrete(body_smpl, renderer_textured_hard, raster)
            body_depth_back_raw = render_depth_discrete(body_smpl, renderer_textured_hard_back, raster_back)
        else:
            body_depth_raw = render_depth_discrete(body_smpl_noArm, renderer_textured_hard, raster)
            body_depth_back_raw = render_depth_discrete(body_smpl_noArm, renderer_textured_hard_back, raster_back)
        body_depth_back_raw = np.fliplr(body_depth_back_raw)

        
        mask_torsor = render_torsor(body_smpl, renderer_textured_hard, raster, color_smpl_raw).astype(np.uint8)[:,:,0]
        mask_torsor_back = render_torsor(body_smpl, renderer_textured_hard_back, raster_back, color_smpl_raw).astype(np.uint8)[:,:,0]
        mask_torsor_back = np.fliplr(mask_torsor_back)
        mask_torsor = cv2.resize(mask_torsor, (192, 192), interpolation=cv2.INTER_NEAREST)
        mask_torsor_back = cv2.resize(mask_torsor_back, (192, 192), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(save_path, 'mask_torsor_%s.png'%img_name), mask_torsor*255)
        cv2.imwrite(os.path.join(save_path, 'mask_torsor_back_%s.png'%img_name), mask_torsor_back*255)
        mask_torsor = torch.BoolTensor(mask_torsor).unsqueeze(0).cuda()
        mask_torsor_back = torch.BoolTensor(mask_torsor_back).unsqueeze(0).cuda()

        body_seg = _process_seg(body_seg, resize=True)
        body_seg_back = _process_seg(body_seg_back, resize=True)

        body_depth = _process_depth(body_depth, resize=True)
        body_depth_back = _process_depth(body_depth_back, resize=True)
        
        body_depth_raw = _process_depth(body_depth_raw, resize=True)
        body_depth_back_raw = _process_depth(body_depth_back_raw, resize=True)

        cond_img, normal_front = _process_image(normal, resize=True)
        cv2.imwrite(os.path.join(save_path, 'images_normal_front_%s.png'%img_name), normal_front)

        _, mask_front = _process_image(mask, resize=True)
        mask_front = mask_front == 255
        cv2.imwrite(os.path.join(save_path, 'images_mask_front_%s.png'%img_name), mask_front.astype(np.uint8)*255)


        cond_img = np.concatenate((cond_img, body_seg[:,:, [0]], body_depth), axis=-1)
        cond_img_back = np.concatenate((normal_back, body_seg_back[:,:, [0]], body_depth_back), axis=-1)
        cond_img_fb = np.concatenate((cond_img, cond_img_back), axis=-1)
        conditions = torch.FloatTensor(cond_img_fb).cuda().permute(2,0,1).unsqueeze(0)
        

        mask_bool = torch.BoolTensor(mask_front).unsqueeze(0).cuda()
        mask_bool_back = torch.BoolTensor(mask_back).unsqueeze(0).cuda()
        body_depth = torch.FloatTensor(body_depth[:,:,0]).unsqueeze(0).cuda()
        body_depth_back = torch.FloatTensor(body_depth_back[:,:,0]).unsqueeze(0).cuda()
        body_depth_raw = torch.FloatTensor(body_depth_raw[:,:,0]).unsqueeze(0).cuda()
        body_depth_back_raw = torch.FloatTensor(body_depth_back_raw[:,:,0]).unsqueeze(0).cuda()


        mask_body_f = body_depth_raw != -1
        mask_body_b = body_depth_back_raw != -1
        mask_body_f = torch.logical_and(mask_body_f, mask_bool)
        mask_body_b = torch.logical_and(mask_body_b, mask_bool_back)

        body_depth_back[body_depth_back==-1] = 1
        body_depth_back_raw[body_depth_back_raw==-1] = 1

        if garment == 'Tshirt' or garment == 'Trousers' or garment == 'Jacket':
            observation = [mask_body_f, mask_body_b, body_depth_raw, body_depth_back_raw]
        elif garment == 'Skirt':
            mask_torsor = torch.logical_and(mask_bool, mask_torsor)
            mask_torsor = torch.logical_and(mask_body_f, mask_torsor)
            mask_torsor_back = torch.logical_and(mask_bool_back, mask_torsor_back)
            mask_torsor_back = torch.logical_and(mask_body_b, mask_torsor_back)
            observation = [mask_torsor, mask_torsor_back, body_depth_raw, body_depth_back_raw] 
            
        images_uv = pipeline(
            conditions=conditions,
            generator=torch.Generator(device=pipeline.device).manual_seed(0),
            batch_size=1,
            num_inference_steps=ddpm_num_inference_steps,
            output_type="numpy",
            use_guidance=use_guidance,
            measure_func=measure_func,
            observation=observation,
            guide_scale=20.0,
        ).images
        

        images_uv_f = images_uv[:,:,:,:4]
        images_uv_b = images_uv[:,:,:,4:]
        coord_img_f = mask_to_coord(mask_front.reshape(1, mask_front.shape[0], mask_front.shape[1]))
        coord_img_b = mask_to_coord(mask_back.reshape(1, mask_back.shape[0], mask_back.shape[1]))

        np.savez(os.path.join(save_path, 'uv_transfer_%s'%(img_name)), uv_transfer=images_uv[0]*2-1)


        images_depth_f = (images_uv_f[:,:,:,[-1]] * 255).round().astype("uint8")
        images_depth_b = (images_uv_b[:,:,:,[-1]] * 255).round().astype("uint8")

        uv_mapping_pred_f, uv_mapping_pred_mask_f = draw_uv_mapping(images_uv_f[:,:,:,:3], coord_img_f)
        uv_mapping_pred_b, uv_mapping_pred_mask_b = draw_uv_mapping(images_uv_b[:,:,:,:3], coord_img_b)

        cv2.imwrite(os.path.join(save_path, 'img_pred_depth_transfer_f_%s.png'%(img_name)), images_depth_f[0])
        cv2.imwrite(os.path.join(save_path, 'img_pred_depth_transfer_b_%s.png'%(img_name)), images_depth_b[0])

        cv2.imwrite(os.path.join(save_path, 'img_pred_uv_transfer_f_%s.png'%(img_name)), (images_uv_f[0,:,:,:3] * 255).round().astype("uint8"))
        cv2.imwrite(os.path.join(save_path, 'img_pred_uv_transfer_b_%s.png'%(img_name)), (images_uv_b[0,:,:,:3] * 255).round().astype("uint8"))
        cv2.imwrite(os.path.join(save_path, 'img_pred_uv_f_%s.png'%(img_name)), uv_mapping_pred_f[0])
        cv2.imwrite(os.path.join(save_path, 'img_pred_uv_b_%s.png'%(img_name)), uv_mapping_pred_b[0])
        cv2.imwrite(os.path.join(save_path, 'img_pred_uv_mask_f_%s.png'%(img_name)), uv_mapping_pred_mask_f[0])
        cv2.imwrite(os.path.join(save_path, 'img_pred_uv_mask_b_%s.png'%(img_name)), uv_mapping_pred_mask_b[0])

        coord_img_f = coord_img_f[0]
        coord_img_b = coord_img_b[0]
        depth_img_f = images_uv_f[0][:,:,[-1]]*2-1
        depth_img_b = images_uv_b[0][:,:,[-1]]*2-1
        z_f = depth_img_f[coord_img_f[:,0], coord_img_f[:,1]].reshape(-1)
        z_b = depth_img_b[coord_img_b[:,0], coord_img_b[:,1]].reshape(-1)
        xyz_f = _to_xyz(coord_img_f, z_f).astype(np.float32)
        xyz_b = _to_xyz(coord_img_b, z_b).astype(np.float32)
        xyz = np.concatenate((xyz_f, xyz_b), axis=0)
        pt = trimesh.PointCloud(xyz)
        pt.export(os.path.join(save_path, 'xyz_%s.ply'%(img_name)))