import os,sys
import numpy as np 
import torch
import trimesh
import cv2
import argparse

sys.path.append('../')
from networks import SDF
from utils.isp import create_uv_mesh, repair_pattern
from utils.isp import reconstruct_pattern_with_label, sewing_front_back
from utils.optimization_isp import optimize_lat_code_anchors, vis_diff, cat_images



def load_isp(garment_type, ckpt_path):
    if 'Skirt' == garment_type or 'Tshirt' == garment_type:
        numG = 100 
    elif  garment_type == 'Trousers':
        numG = 239 
    elif  garment_type == 'Jacket':
        numG = 146 
    num_edges = 3 if 'Skirt' == garment_type else 5
    num_seams = num_edges - 1

    rep_size = 32
    model_sdf_f = SDF.SDF2branch_deepSDF(d_in=2+rep_size, d_out=1+num_edges, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()
    model_sdf_b = SDF.SDF2branch_deepSDF(d_in=2+rep_size, d_out=1+num_edges, dims=[256, 256, 256, 256, 256, 256], skip_in=[3]).cuda()
    model_rep = SDF.learnt_representations(rep_size=rep_size, samples=numG).cuda()
    model_atlas_f = SDF.SDF(d_in=2+rep_size, d_out=3, dims=[256, 256, 256, 256, 256, 256], skip_in=[3], geometric_init=False).cuda()
    model_atlas_b = SDF.SDF(d_in=2+rep_size, d_out=3, dims=[256, 256, 256, 256, 256, 256], skip_in=[3], geometric_init=False).cuda()


    model_sdf_f.load_state_dict(torch.load(os.path.join(ckpt_path, 'sdf_f_%s.pth'%garment_type)))
    model_sdf_b.load_state_dict(torch.load(os.path.join(ckpt_path, 'sdf_b_%s.pth'%garment_type)))
    model_rep.load_state_dict(torch.load(os.path.join(ckpt_path, 'rep_%s.pth'%garment_type)))
    model_atlas_f.load_state_dict(torch.load(os.path.join(ckpt_path, 'atlas_f_%s.pth'%garment_type)))
    model_atlas_b.load_state_dict(torch.load(os.path.join(ckpt_path, 'atlas_b_%s.pth'%garment_type)))

    latent_codes = model_rep.weights.detach()
    
    return [model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b], latent_codes, num_seams



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_root', type=str, default='../checkpoints')
    parser.add_argument('--save_root', type=str, default='../fitting-results')
    parser.add_argument('--garment', type=str, default='Skirt')

    args = parser.parse_args()

    ckpt_root = args.ckpt_root
    save_root = args.save_root
    garment = args.garment

    ckpt_path = os.path.join(ckpt_root, 'ISP')
    save_folder = os.path.join(save_root, garment)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    images_list = sorted(list(set([f.split('-')[0] for f in os.listdir(save_folder) if 'FB' not in f])))

    x_res = y_res = 256
    uv_vertices, uv_faces = create_uv_mesh(x_res, y_res, debug=False)
    mesh_uv = trimesh.Trimesh(uv_vertices, uv_faces, process=False, validate=False)
    edges = torch.LongTensor(mesh_uv.edges).cuda()
    uv_vertices = torch.FloatTensor(uv_vertices).cuda()

    ISP, latent_codes, num_seams = load_isp(garment, ckpt_path)

    for i in range(len(images_list)):
        img_name = images_list[i]
        
        load_path = os.path.join(save_folder, img_name+'-FB')
        save_path = os.path.join(save_folder, img_name+'-isp')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        mask_f = cv2.imread(os.path.join(load_path, 'img_pred_uv_mask_f_%s.png'%(img_name)))[:,:,0]
        mask_b = cv2.imread(os.path.join(load_path, 'img_pred_uv_mask_b_%s.png'%(img_name)))[:,:,0]
        mask = ((mask_f == 255) + (mask_b == 255)).astype(int)*255
        W = mask.shape[1]
        img_f = mask[:,:W//2]
        img_b = mask[:,W//2:]

        anchor_codes = latent_codes
        weight_rep = 0.1/2 if garment == 'Skirt' else 0.02
        weight_area = 0.5 
        latent_code, img_f_new, img_b_new, label_f, label_b = optimize_lat_code_anchors(ISP, anchor_codes, [img_f, img_b], uv_vertices, iters=500, weight_rep=weight_rep, weight_area=weight_area)

        img_f_diff = vis_diff(img_f_new.copy(), img_f.copy())
        img_b_diff = vis_diff(img_b_new.copy(), img_b.copy())

        img_new = np.concatenate((img_f_new, img_b_new), axis=1)
        img_diff = np.concatenate((img_f_diff, img_b_diff), axis=1)

        img_cat = cat_images(mask, img_new, img_diff)
        cv2.imwrite(os.path.join(save_path, 'isp-fitting-difference-%s.png'%(img_name)), img_cat)
        cv2.imwrite(os.path.join(save_path, 'isp-fitting-%s.png'%(img_name)), img_new)

        mesh_atlas_f, mesh_atlas_b, mesh_pattern_f, mesh_pattern_b, label_f, label_b = reconstruct_pattern_with_label(ISP, latent_code, uv_vertices, uv_faces, edges, resolution=x_res)

        mesh_atlas_sewing, labels = sewing_front_back(garment, mesh_pattern_f, mesh_pattern_b, mesh_atlas_f, mesh_atlas_b, label_f, label_b, num_seams=num_seams)

        mesh_pattern_f.export(os.path.join(save_path, 'pattern-f-%s.ply'%(img_name)))
        mesh_pattern_b.export(os.path.join(save_path, 'pattern-b-%s.ply'%(img_name)))
        mesh_atlas_f.export(os.path.join(save_path, 'atlas-f-%s.ply'%(img_name)))
        mesh_atlas_b.export(os.path.join(save_path, 'atlas-b-%s.ply'%(img_name)))
        mesh_atlas_sewing.export(os.path.join(save_path, 'sewing-%s.ply'%(img_name)))
