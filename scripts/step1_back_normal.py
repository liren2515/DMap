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
from utils.render import get_render, render_segmentation, render_depth

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")



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

    ddpm_num_inference_steps = 1000

    ckpt_path = os.path.join(ckpt_root, 'normal-back-%s'%garment)
    data_path = os.path.join(data_root, garment)
    save_folder = os.path.join(save_root, garment)


    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    target_label = get_mask_label(garment)
    pipeline = DDPMPipeline.from_pretrained(ckpt_path, use_safetensors=True).to(device)
    raster, renderer_textured_hard = get_render()
    raster_back, renderer_textured_hard_back = get_render(is_back=True)
    color_smpl = np.load('../extra-data/color_smpl_faces.npy')/15


    images_list = sorted(list(set([f.split('_')[0] for f in os.listdir(data_path)])))

    for i in range(len(images_list)):
        img_name = images_list[i]

        save_path = os.path.join(save_folder, img_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        normal = cv2.imread(os.path.join(data_path, '%s_normal_align.png'%img_name))
        seg = cv2.imread(os.path.join(data_path, '%s_seg_align.png'%img_name))[:,:,0]
        mask = ((seg == target_label).astype(np.uint8))*255
        normal[seg != target_label] = 0
        
        body_smpl = trimesh.load(os.path.join(data_path, '%s_body.ply'%img_name))
        body_seg = render_segmentation(body_smpl, renderer_textured_hard, raster, color_smpl)
        body_seg_back = render_segmentation(body_smpl, renderer_textured_hard_back, raster_back, color_smpl)
        body_seg_back = np.fliplr(body_seg_back)
        body_depth = render_depth(body_smpl, renderer_textured_hard)
        body_depth_back = render_depth(body_smpl, renderer_textured_hard_back)
        body_depth_back = np.fliplr(body_depth_back)
        cv2.imwrite(os.path.join(save_path, 'body_seg_%s.png'%img_name), body_seg)
        cv2.imwrite(os.path.join(save_path, 'body_seg_back_%s.png'%img_name), body_seg_back)
        cv2.imwrite(os.path.join(save_path, 'body_depth_%s.png'%img_name), ((body_depth+1)/2*255).astype(np.uint8))
        cv2.imwrite(os.path.join(save_path, 'body_depth_back_%s.png'%img_name), ((body_depth_back+1)/2*255).astype(np.uint8))
        body_seg = _process_seg(body_seg, resize=True)
        body_seg_back = _process_seg(body_seg_back, resize=True)
        body_depth = _process_depth(body_depth, resize=True)
        body_depth_back = _process_depth(body_depth_back, resize=True)

        cond_img, normal_resize = _process_image(normal, resize=True)
        cv2.imwrite(os.path.join(save_path, 'normal_resize_%s.png'%img_name), normal_resize)

        cond_img = np.concatenate((cond_img, body_seg[:,:, [0]], body_depth, body_seg_back[:,:, [0]], body_depth_back), axis=-1)
        conditions = torch.FloatTensor(cond_img).cuda().permute(2,0,1).unsqueeze(0)

        # run pipeline in inference (sample random noise and denoise)
        images_uv = pipeline(
            conditions=conditions,
            generator=torch.Generator(device=pipeline.device).manual_seed(0),
            batch_size=1,
            num_inference_steps=ddpm_num_inference_steps,
            output_type="numpy",
            use_guidance=False,
            measure_func=None,
            observation=[],
            guide_scale=50.0,
        ).images # (0, 1)

        images_mask = images_uv[:,:,:,[-1]]
        images_normal = images_uv[:,:,:,:3]

        images_mask = images_mask > 0.5
        images_mask = images_mask.astype(np.uint8)*255
        images_normal = (images_normal*255).astype(np.uint8)

        cv2.imwrite(os.path.join(save_path, 'images_normal_back_%s.png'%(img_name)), images_normal[0])
        cv2.imwrite(os.path.join(save_path, 'images_mask_back_%s.png'%(img_name)), images_mask[0])