import os, sys
import torch
import torch.nn.functional as F
import cv2
import numpy as np

def process_sparse(img):
    img = img/255
    img = img + np.fliplr(img)
    img = (img > 0).reshape(-1)
    return img

def comput_iou(x1, x2):
    intersection = (x1 * x2).sum()
    union = x1.sum() + x2.sum() - intersection
    return intersection*1.0/union

def get_best_anchor(model_isp, anchor_codes, imgs_sparse, uv_vertices):
    model_sdf_f, model_sdf_b, _, _ = model_isp
    img_sparse_f, img_sparse_b = imgs_sparse
    
    with torch.no_grad():
        uv_vertices_f_input = uv_vertices[:,:2].clone()*10
        uv_vertices_b_input = uv_vertices[:,:2].clone()*10

        max_IoU = 0
        max_i = -1
        for i in range(len(anchor_codes)):
            lat_code = anchor_codes[i]
        
            latent_code_f = lat_code.repeat(len(uv_vertices_f_input), 1)
            latent_code_b = lat_code.repeat(len(uv_vertices_b_input), 1)

            pred_f = model_sdf_f(uv_vertices_f_input, latent_code_f)
            pred_b = model_sdf_b(uv_vertices_b_input, latent_code_b)

            sdf_pred_f = pred_f[:, 0].squeeze() < 0
            sdf_pred_b = pred_b[:, 0].squeeze() < 0

            iou_f = comput_iou(sdf_pred_f, img_sparse_f)
            iou_b = comput_iou(sdf_pred_b, img_sparse_b)
            iou = (iou_f + iou_b)/2

            if max_IoU < iou:
                max_i = i
                max_IoU = iou

    return anchor_codes[max_i].clone()

def optimize_lat_code_anchors(model_isp, anchor_codes, imgs_sparse, uv_vertices, iters=1000, weight_rep=0.02, weight_area=2):
    model_sdf_f, model_sdf_b, _, _ = model_isp
    img_sparse_f, img_sparse_b = imgs_sparse
    
    for param in model_sdf_f.parameters():
        param.requires_grad = False
    for param in model_sdf_b.parameters():
        param.requires_grad = False

    v_indicator_f = process_sparse(img_sparse_f)
    v_indicator_b = process_sparse(img_sparse_b)
    v_indicator_f = torch.BoolTensor(v_indicator_f).cuda()
    v_indicator_b = torch.BoolTensor(v_indicator_b).cuda()


    lat_code = get_best_anchor(model_isp, anchor_codes, [v_indicator_f.int(), v_indicator_b.int()], uv_vertices).unsqueeze(0)
    lat_code_offset = torch.zeros_like(lat_code)
    lat_code_offset.requires_grad = True

    
    lr = 1e-3
    eps = 0#-1e-3
    optimizer = torch.optim.Adam([{'params': lat_code_offset, 'lr': lr},])
    
    uv_vertices_f_input = uv_vertices[:,:2].clone()*10
    uv_vertices_b_input = uv_vertices[:,:2].clone()*10
    uv_vertices_f_input.requires_grad = False
    uv_vertices_b_input.requires_grad = False

    loss_pre = 1e10
    for i in range(iters):
        lat_code_new = lat_code+lat_code_offset
        latent_code_f = lat_code_new.repeat(len(uv_vertices_f_input), 1)
        latent_code_b = lat_code_new.repeat(len(uv_vertices_b_input), 1)

        pred_f = model_sdf_f(uv_vertices_f_input, latent_code_f)
        pred_b = model_sdf_b(uv_vertices_b_input, latent_code_b)

        sdf_pred_f = pred_f[:, 0].squeeze()
        sdf_pred_b = pred_b[:, 0].squeeze()

        loss_sdf = 0
        if v_indicator_f.sum() > 0:
            loss_sdf += F.relu(eps + sdf_pred_f[v_indicator_f]).mean() 
        if v_indicator_b.sum() > 0:
            loss_sdf += F.relu(eps + sdf_pred_b[v_indicator_b]).mean()
        loss_rep = lat_code_offset.norm(dim=-1).mean()
        loss_area = -sdf_pred_f.mean() - sdf_pred_b.mean()
        loss = loss_sdf/4 + loss_rep*weight_rep + loss_area/100*weight_area
        print('iter: %3d, loss: %0.5f, loss_sdf: %0.5f, loss_rep: %0.5f, loss_area: %0.5f'%(i, loss.item(), loss_sdf.item(), loss_rep.item(), loss_area.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if abs(loss_pre-loss.item()) < 1e-6:
            break
        else:
            loss_pre = loss.item()

    with torch.no_grad():
        lat_code_new = lat_code+lat_code_offset
        latent_code_f = lat_code_new.repeat(len(uv_vertices_f_input), 1)
        latent_code_b = lat_code_new.repeat(len(uv_vertices_b_input), 1)

        pred_f = model_sdf_f(uv_vertices_f_input, latent_code_f)
        pred_b = model_sdf_b(uv_vertices_b_input, latent_code_b)

        v_indicator_f_new = pred_f[:, 0].squeeze() < 0
        v_indicator_b_new = pred_b[:, 0].squeeze() < 0

        label_f = pred_f[:, 1:]
        label_b = pred_b[:, 1:]
        label_f = torch.argmax(label_f, dim=-1)
        label_b = torch.argmax(label_b, dim=-1)

        res = img_sparse_f.shape[0]
        v_indicator_f_new = v_indicator_f_new.reshape(res, res).detach().cpu().numpy().astype(int)*255
        v_indicator_b_new = v_indicator_b_new.reshape(res, res).detach().cpu().numpy().astype(int)*255

        label_f = label_f.reshape(res, res).detach().cpu().numpy().astype(int)
        label_b = label_b.reshape(res, res).detach().cpu().numpy().astype(int)

    return lat_code_new.detach(), v_indicator_f_new, v_indicator_b_new, label_f, label_b


def vis_diff(img_opt, img):
    if img.max() == 255:
        img = img/255
    if img_opt.max() == 255:
        img_opt = img_opt/255

    img_opt = img_opt.astype(int)
    img = img.astype(int)

    img_diff = ((img_opt - img)+1)*127
    img_diff = img_diff.astype(np.uint8)

    return img_diff

def cat_images(img1, img2, img3):

    insert = (np.zeros((10, img1.shape[1]))+255).astype(np.uint8)
    img = np.concatenate((img1, insert, img2, insert, img3), axis=0)

    return img