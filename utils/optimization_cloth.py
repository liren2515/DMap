import os, sys
import cv2
import numpy as np
import trimesh
import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

from .cutting import select_boundary
from .rasterize import get_pix_to_face_with_body, get_pix_to_face_v2
from .chamfer import chamfer_distance_single, chamfer_distance

sys.path.append('../')
from snug.snug_helper import stretching_energy, bending_energy, gravitational_energy, collision_penalty
from networks.SDF import SDF

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def uv_to_3D_torch(pattern_deform, uv_faces, barycentric_uv, closest_face_idx_uv):
    uv_faces_id = uv_faces[closest_face_idx_uv]
    uv_faces_id = uv_faces_id.reshape(-1)

    pattern_deform_triangles = pattern_deform[uv_faces_id].reshape(-1, 3, 3)
    pattern_deform_bary = (pattern_deform_triangles * barycentric_uv[:, :, None]).sum(dim=-2)
    return pattern_deform_bary

def align_observation_uv(renders, img_init, cloth_pose, cloth_state, mapping_related, image_rest, masks, normals, body_mesh, clothed_mesh, waist_v_id, use_double_cd=False):

    renderer_textured_soft, transform, raster = renders
    vertices_waist = torch.FloatTensor(cloth_pose.vertices[waist_v_id]).cuda()

    faces_f, faces_b, v_barycentric_f, v_barycentric_b, closest_face_idx_f, closest_face_idx_b, sparse_uv, sparse_mask, xyz = mapping_related

    faces_f = torch.LongTensor(faces_f).cuda()
    faces_b = torch.LongTensor(faces_b).cuda()
    closest_face_idx_f = torch.LongTensor(closest_face_idx_f).cuda()
    closest_face_idx_b = torch.LongTensor(closest_face_idx_b).cuda()
    v_barycentric_f = torch.FloatTensor(v_barycentric_f).cuda()
    v_barycentric_b = torch.FloatTensor(v_barycentric_b).cuda()

    vb = torch.FloatTensor(body_mesh.vertices).cuda()
    vb_flip = vb.clone()
    vb_flip[:, -1] *= -1
    nb = torch.FloatTensor(body_mesh.vertex_normals).cuda()
    fb = torch.LongTensor(body_mesh.faces).cuda()

    vb_clothed = torch.FloatTensor(clothed_mesh.vertices).cuda()
    vb_clothed_flip = vb_clothed.clone()
    vb_clothed_flip[:, -1] *= -1
    fb_clothed = torch.LongTensor(clothed_mesh.faces).cuda()

    mask_front, mask_back, mask_other = masks
    normal_front, normal_back = normals
    mask_front = torch.FloatTensor(mask_front).cuda()
    mask_back = torch.FloatTensor(mask_back).cuda()
    mask_other = torch.FloatTensor(mask_other).cuda()
    

    with torch.no_grad():
        faces_cloth = torch.LongTensor(cloth_pose.faces).cuda()
        verts_cloth_zero = torch.FloatTensor(cloth_pose.vertices).cuda()*0
        cloth_rgb = torch.zeros(len(verts_cloth_zero), 3) + 255 # (1, V, 3)
        verts_rgb = cloth_rgb[None]
        textures = TexturesVertex(verts_features=verts_rgb.cuda())
        
        idx_x, idx_y = np.where(mask_front.cpu().numpy()>0.5)
        idx_mask = np.stack((idx_x, idx_y), axis=-1).astype(float)
        idx_mask = torch.FloatTensor(idx_mask).cuda()
        idx_mask_pend_f = torch.cat((idx_mask, torch.zeros(idx_mask.shape[0], 1).cuda()), dim=-1).unsqueeze(0)

        normal = (normal_front[idx_x, idx_y].astype(float)/255*2) - 1
        normal = torch.FloatTensor(normal).cuda()
        normal_img_f = normal/normal.norm(p=2, dim=-1, keepdim=True)
        normal_img_f = normal_img_f.unsqueeze(0)
        
        idx_x, idx_y = np.where(mask_back.cpu().numpy()>0.5)
        idx_mask = np.stack((idx_x, idx_y), axis=-1).astype(float)
        idx_mask = torch.FloatTensor(idx_mask).cuda()
        idx_mask_pend_b = torch.cat((idx_mask, torch.zeros(idx_mask.shape[0], 1).cuda()), dim=-1).unsqueeze(0)
        
        normal = (normal_back[idx_x, idx_y].astype(float)/255*2) - 1
        normal = torch.FloatTensor(normal).cuda()
        normal_img_b = normal/normal.norm(p=2, dim=-1, keepdim=True)
        normal_img_b = normal_img_b.unsqueeze(0)

        idx_x, idx_y = np.where(((mask_front+mask_other).cpu().numpy())>0.5)
        idx_mask = np.stack((idx_x, idx_y), axis=-1).astype(float)
        idx_mask = torch.FloatTensor(idx_mask).cuda()
        idx_mask_full_pend = torch.cat((idx_mask, torch.zeros(idx_mask.shape[0], 1).cuda()), dim=-1).unsqueeze(0)

    
    nn = SDF(d_in=6, d_out=3, dims=[256, 256, 256], skip_in=[]).cuda()
    lr = 1e-3
    eps = 2e-3
    optimizer = torch.optim.Adam(list(nn.parameters()), lr=lr)
    condition = torch.cat((image_rest, img_init[:,:3]), dim=1)*10

    iters = 1500+500
    for i in range(iters):
        condition_reshape = condition.permute(0,2,3,1).reshape(-1, 6)
        offset = nn(condition_reshape, None)/100
        offset = offset.reshape(1, 128, 256, 3).permute(0,3,1,2)

        image_est = img_init[:,:3] + offset

        uv_est = image_est[:,:3].squeeze().permute(1,2,0)
        uv_f = uv_est[:,:128].reshape(-1,3)
        uv_b = uv_est[:,128:].reshape(-1,3)

        output_uv = image_est[:,:3].permute(0,2,3,1)
        loss_sparse_uv = torch.linalg.norm((output_uv[sparse_mask] - sparse_uv[sparse_mask]), dim=-1).mean()*100*5*2
        
        if i >= 500 and i < 1500:
            loss_sparse_uv /= 10
        elif i >= 1500:
            loss_sparse_uv *= 0

        verts_f = uv_to_3D_torch(uv_f, faces_f, v_barycentric_f, closest_face_idx_f)
        verts_b = uv_to_3D_torch(uv_b, faces_b, v_barycentric_b, closest_face_idx_b)
        verts_cloth_new = torch.cat((verts_f, verts_b), axis=0)      

        loss_strain = stretching_energy(verts_cloth_new.unsqueeze(0), cloth_state)
        loss_bending = bending_energy(verts_cloth_new.unsqueeze(0), cloth_state)*5
        loss_gravity = gravitational_energy(verts_cloth_new.unsqueeze(0), cloth_state.v_mass)

        if i == iters-1:
            mesh = Meshes(
                verts=[verts_cloth_zero],   
                faces=[faces_cloth],
                textures=textures
            )
            new_src_mesh = mesh.offset_verts(verts_cloth_new)
            images_predicted = renderer_textured_soft(new_src_mesh)
            images_pred = images_predicted[0, :, :, 3]
            img_mask = (images_pred.detach().cpu().numpy()*255).astype(np.uint8)
        

        with torch.no_grad():
            idx_faces_f, _ = get_pix_to_face_with_body(verts_cloth_new, faces_cloth, vb_clothed, fb_clothed, raster)
            verts_cloth_new_flip = verts_cloth_new.clone()
            verts_cloth_new_flip[:,-1] *=-1
            idx_faces_b, _ = get_pix_to_face_with_body(verts_cloth_new_flip, faces_cloth, vb_clothed_flip, fb_clothed, raster)
            faces_cloth_f = faces_cloth[idx_faces_f]
            faces_cloth_b = faces_cloth[idx_faces_b]
        tri_f = verts_cloth_new[faces_cloth_f.reshape(-1)].reshape(-1,3,3)
        tri_b = verts_cloth_new[faces_cloth_b.reshape(-1)].reshape(-1,3,3)
        tri_center_f = tri_f.mean(dim=1)
        tri_center_b = tri_b.mean(dim=1)
        vectors_f = tri_f[:,1:] - tri_f[:,:2]
        vectors_b = tri_b[:,1:] - tri_b[:,:2]
        normal_f = torch.cross(vectors_f[:, 0], vectors_f[:, 1], dim=-1)
        normal_b = torch.cross(vectors_b[:, 0], vectors_b[:, 1], dim=-1)
        normal_f = normal_f/normal_f.norm(p=2, dim=-1, keepdim=True)
        normal_b = normal_b/normal_b.norm(p=2, dim=-1, keepdim=True)
        normal_f = normal_f.unsqueeze(0)
        normal_b = normal_b.unsqueeze(0)

        
        verts_cloth_2D_f = (transform.transform_points(tri_center_f.unsqueeze(0))[:,:,[1,0]]*(-255.5)) + 255.5 # x,y
        verts_cloth_2D_b = (transform.transform_points(tri_center_b.unsqueeze(0))[:,:,[1,0]]*(-255.5)) + 255.5 # x,y
        verts_cloth_2D_pend_f = torch.cat((verts_cloth_2D_f, torch.zeros(verts_cloth_2D_f.shape[0], verts_cloth_2D_f.shape[1], 1).cuda()), dim=-1)
        verts_cloth_2D_pend_b = torch.cat((verts_cloth_2D_b, torch.zeros(verts_cloth_2D_b.shape[0], verts_cloth_2D_b.shape[1], 1).cuda()), dim=-1)
        
        _, loss_normal_f = chamfer_distance(verts_cloth_2D_pend_f, idx_mask_pend_f, x_normals=normal_f, y_normals=normal_img_f)
        _, loss_normal_b = chamfer_distance(idx_mask_pend_b, verts_cloth_2D_pend_b, x_normals=normal_img_b, y_normals=normal_b, abs_normal=True)
        loss_cd_2d_f_0, _ = chamfer_distance_single(idx_mask_pend_f, verts_cloth_2D_pend_f)
        loss_cd_2d_f_1, _ = chamfer_distance_single(verts_cloth_2D_pend_f, idx_mask_full_pend)
        loss_cd_2d_b_0, _ = chamfer_distance_single(idx_mask_pend_b, verts_cloth_2D_pend_b)
        loss_cd_2d_b_1, _ = chamfer_distance_single(verts_cloth_2D_pend_b, idx_mask_full_pend)
        loss_mask = (loss_cd_2d_f_0 + loss_cd_2d_f_1 + loss_cd_2d_b_0 + loss_cd_2d_b_1)/5
        loss_normal = (loss_normal_f + loss_normal_b)*2


        if use_double_cd:
            loss_cd_3d, _ = chamfer_distance(xyz, verts_cloth_new.unsqueeze(0))
        else:
            loss_cd_3d, _ = chamfer_distance_single(xyz, verts_cloth_new.unsqueeze(0))
            loss_cd_3d = loss_cd_3d*2
        if i < 500:
            loss_cd_3d *= 0
        else:
            loss_cd_3d *= 10000

        loss_collision = collision_penalty(verts_cloth_new.unsqueeze(0), vb.unsqueeze(0), nb.unsqueeze(0), eps=eps)
        loss_edge = torch.sqrt(((verts_cloth_new[waist_v_id] - vertices_waist)**2).sum(dim=-1)).mean()*10


        loss = (loss_bending + loss_strain/4 + loss_gravity) + loss_sparse_uv + loss_cd_3d + loss_mask + loss_normal + loss_collision + loss_edge
        print('iter: %3d, loss: %0.4f, loss_strain: %0.4f, loss_bending: %0.4f, loss_gravity: %0.4f, loss_sparse_uv: %0.4f , loss_mask: %0.4f , loss_cd_3d: %0.4f , loss_normal: %0.4f , loss_collision: %0.4f , loss_edge: %0.4f '%(i, loss.item(), loss_strain.item(), loss_bending.item(), loss_gravity.item(), loss_sparse_uv.item(), loss_mask.item(), loss_cd_3d.item(), loss_normal.item(), loss_collision.item(), loss_edge.item()))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    verts_cloth_new = verts_cloth_new.detach().cpu().numpy()
    cloth_pose_uv = cloth_pose.copy()
    cloth_pose_uv.vertices = verts_cloth_new

    return cloth_pose_uv, img_mask


def align_observation_pt_verts(renders, cloth_pose, cloth_state, body_mesh, clothed_mesh, masks, normals, vertices_waist, waist_v_id, xyz, use_double_cd=False):

    renderer_textured_soft, transform, raster = renders
    vertices_waist = torch.FloatTensor(vertices_waist).cuda()

    vb = torch.FloatTensor(body_mesh.vertices).cuda()
    vb_flip = vb.clone()
    vb_flip[:, -1] *= -1
    nb = torch.FloatTensor(body_mesh.vertex_normals).cuda()
    fb = torch.LongTensor(body_mesh.faces).cuda()

    vb_clothed = torch.FloatTensor(clothed_mesh.vertices).cuda()
    vb_clothed_flip = vb_clothed.clone()
    vb_clothed_flip[:, -1] *= -1
    fb_clothed = torch.LongTensor(clothed_mesh.faces).cuda()

    mask_front, mask_back, mask_other = masks
    normal_front, normal_back = normals
    mask_front = torch.FloatTensor(mask_front).cuda()
    mask_back = torch.FloatTensor(mask_back).cuda()
    mask_other = torch.FloatTensor(mask_other).cuda()

    with torch.no_grad():
        faces_cloth = torch.LongTensor(cloth_pose.faces).cuda()
        verts_cloth_zero = torch.FloatTensor(cloth_pose.vertices).cuda()*0
        cloth_rgb = torch.zeros(len(verts_cloth_zero), 3) + 255 # (1, V, 3)
        verts_rgb = cloth_rgb[None]
        textures = TexturesVertex(verts_features=verts_rgb.cuda())
        
        idx_x, idx_y = np.where(mask_front.cpu().numpy()>0.5)
        idx_mask = np.stack((idx_x, idx_y), axis=-1).astype(float)
        idx_mask = torch.FloatTensor(idx_mask).cuda()
        idx_mask_pend_f = torch.cat((idx_mask, torch.zeros(idx_mask.shape[0], 1).cuda()), dim=-1).unsqueeze(0)

        normal = (normal_front[idx_x, idx_y].astype(float)/255*2) - 1
        normal = torch.FloatTensor(normal).cuda()
        normal_img_f = normal/normal.norm(p=2, dim=-1, keepdim=True)
        normal_img_f = normal_img_f.unsqueeze(0)
        
        idx_x, idx_y = np.where(mask_back.cpu().numpy()>0.5)
        idx_mask = np.stack((idx_x, idx_y), axis=-1).astype(float)
        idx_mask = torch.FloatTensor(idx_mask).cuda()
        idx_mask_pend_b = torch.cat((idx_mask, torch.zeros(idx_mask.shape[0], 1).cuda()), dim=-1).unsqueeze(0)
        
        normal = (normal_back[idx_x, idx_y].astype(float)/255*2) - 1
        normal = torch.FloatTensor(normal).cuda()
        normal_img_b = normal/normal.norm(p=2, dim=-1, keepdim=True)
        normal_img_b = normal_img_b.unsqueeze(0)
        
        idx_x, idx_y = np.where(((mask_front+mask_other).cpu().numpy())>0.5)
        idx_mask = np.stack((idx_x, idx_y), axis=-1).astype(float)
        idx_mask = torch.FloatTensor(idx_mask).cuda()
        idx_mask_full_pend = torch.cat((idx_mask, torch.zeros(idx_mask.shape[0], 1).cuda()), dim=-1).unsqueeze(0)


    verts_zero_clothed = torch.zeros(len(body_mesh.vertices)+len(cloth_pose.vertices), 3).cuda()
    faces_clothed = torch.LongTensor(np.concatenate((body_mesh.faces, cloth_pose.faces + len(body_mesh.vertices)))).cuda()
    smpl_rgb = torch.zeros(len(body_mesh.vertices), 3)
    smpl_rgb[:,0] += 255
    gar_rgb = torch.zeros(len(cloth_pose.vertices), 3)
    gar_rgb[:,1] += 255
    verts_rgb = torch.cat((smpl_rgb, gar_rgb))[None]
    textures_clothed = TexturesVertex(verts_features=verts_rgb.to(device))

    verts_pose = torch.FloatTensor(cloth_pose.vertices).cuda()
    
    offset = torch.zeros_like(verts_pose)
    offset.requires_grad = True
    lr = 1e-4
    eps = 2e-3
    optimizer = torch.optim.Adam([{'params': offset, 'lr': lr},])

    iters = 200
    for i in range(iters):
        verts_cloth_new = verts_pose + offset

        loss_strain = stretching_energy(verts_cloth_new.unsqueeze(0), cloth_state)
        loss_bending = bending_energy(verts_cloth_new.unsqueeze(0), cloth_state)*5
        loss_gravity = gravitational_energy(verts_cloth_new.unsqueeze(0), cloth_state.v_mass)

        if i == iters-1:
            mesh = Meshes(
                verts=[verts_cloth_zero],   
                faces=[faces_cloth],
                textures=textures
            )
            new_src_mesh = mesh.offset_verts(verts_cloth_new)
            images_predicted = renderer_textured_soft(new_src_mesh)
            images_pred = images_predicted[0, :, :, 3]
            img_mask = (images_pred.detach().cpu().numpy()*255).astype(np.uint8)
        

        with torch.no_grad():
            idx_faces_f, _ = get_pix_to_face_with_body(verts_cloth_new, faces_cloth, vb_clothed, fb_clothed, raster)
            verts_cloth_new_flip = verts_cloth_new.clone()
            verts_cloth_new_flip[:,-1] *=-1
            idx_faces_b, _ = get_pix_to_face_with_body(verts_cloth_new_flip, faces_cloth, vb_clothed_flip, fb_clothed, raster)
            faces_cloth_f = faces_cloth[idx_faces_f]
            faces_cloth_b = faces_cloth[idx_faces_b]
        tri_f = verts_cloth_new[faces_cloth_f.reshape(-1)].reshape(-1,3,3)
        tri_b = verts_cloth_new[faces_cloth_b.reshape(-1)].reshape(-1,3,3)
        tri_center_f = tri_f.mean(dim=1)
        tri_center_b = tri_b.mean(dim=1)
        vectors_f = tri_f[:,1:] - tri_f[:,:2]
        vectors_b = tri_b[:,1:] - tri_b[:,:2]
        normal_f = torch.cross(vectors_f[:, 0], vectors_f[:, 1], dim=-1)
        normal_b = torch.cross(vectors_b[:, 0], vectors_b[:, 1], dim=-1)
        normal_f = normal_f/normal_f.norm(p=2, dim=-1, keepdim=True)
        normal_b = normal_b/normal_b.norm(p=2, dim=-1, keepdim=True)
        normal_f = normal_f.unsqueeze(0)
        normal_b = normal_b.unsqueeze(0)


        verts_cloth_2D_f = (transform.transform_points(tri_center_f.unsqueeze(0))[:,:,[1,0]]*(-255.5)) + 255.5 # x,y
        verts_cloth_2D_b = (transform.transform_points(tri_center_b.unsqueeze(0))[:,:,[1,0]]*(-255.5)) + 255.5 # x,y
        verts_cloth_2D_pend_f = torch.cat((verts_cloth_2D_f, torch.zeros(verts_cloth_2D_f.shape[0], verts_cloth_2D_f.shape[1], 1).cuda()), dim=-1)
        verts_cloth_2D_pend_b = torch.cat((verts_cloth_2D_b, torch.zeros(verts_cloth_2D_b.shape[0], verts_cloth_2D_b.shape[1], 1).cuda()), dim=-1)
        
        _, loss_normal_f = chamfer_distance(verts_cloth_2D_pend_f, idx_mask_pend_f, x_normals=normal_f, y_normals=normal_img_f)
        _, loss_normal_b = chamfer_distance(idx_mask_pend_b, verts_cloth_2D_pend_b, x_normals=normal_img_b, y_normals=normal_b, abs_normal=True)
        loss_cd_2d_f_0, _ = chamfer_distance_single(idx_mask_pend_f, verts_cloth_2D_pend_f)
        loss_cd_2d_f_1, _ = chamfer_distance_single(verts_cloth_2D_pend_f, idx_mask_full_pend)
        loss_cd_2d_b_0, _ = chamfer_distance_single(idx_mask_pend_b, verts_cloth_2D_pend_b)
        loss_cd_2d_b_1, _ = chamfer_distance_single(verts_cloth_2D_pend_b, idx_mask_full_pend)
        loss_mask = (loss_cd_2d_f_0 + loss_cd_2d_f_1 + loss_cd_2d_b_0 + loss_cd_2d_b_1)/5/10
        loss_normal = (loss_normal_f + loss_normal_b)*2

        flip_z = torch.FloatTensor([1,1,-1]).cuda()
        verts_clothed = torch.cat((vb, verts_cloth_new), dim=0)
        verts_clothed_flip = verts_clothed*flip_z[None]
        mesh_f = Meshes(
                verts=[verts_zero_clothed],   
                faces=[faces_clothed],
                textures=textures_clothed
        )
        mesh_b = Meshes(
                verts=[verts_zero_clothed],   
                faces=[faces_clothed],
                textures=textures_clothed
        )
        new_src_mesh_f = mesh_f.offset_verts(verts_clothed)
        new_src_mesh_b = mesh_b.offset_verts(verts_clothed_flip)
        images_predicted_f = renderer_textured_soft(new_src_mesh_f)
        images_predicted_b = renderer_textured_soft(new_src_mesh_b)
        images_pred_f = images_predicted_f[0, :, :, 1]/255
        images_pred_b = images_predicted_b[0, :, :, 1]/255

        intersection_f = (images_pred_f*mask_front).sum()
        intersection_b = (images_pred_b*mask_back).sum()
        union_f = images_pred_f.sum() + mask_front.sum() - intersection_f
        union_b = images_pred_b.sum() + mask_back.sum() - intersection_b
        loss_mask = ((1 - intersection_f/union_f) + (1 - intersection_b/union_b)) #*224
        
        if use_double_cd:
            loss_cd_3d, _ = chamfer_distance(xyz, verts_cloth_new.unsqueeze(0))
        else:
            loss_cd_3d, _ = chamfer_distance_single(xyz, verts_cloth_new.unsqueeze(0))
            loss_cd_3d = loss_cd_3d*2
        loss_cd_3d *= 10000

        loss_collision = collision_penalty(verts_cloth_new.unsqueeze(0), vb.unsqueeze(0), nb.unsqueeze(0), eps=eps)
        loss_waist = torch.sqrt(((verts_cloth_new[waist_v_id] - vertices_waist)**2).sum(dim=-1)).mean()*10

        loss = (loss_bending + loss_strain/2 + loss_gravity) + loss_cd_3d + loss_mask + loss_normal + loss_collision + loss_waist
        print('iter: %3d, loss: %0.4f, loss_strain: %0.4f, loss_bending: %0.4f, loss_gravity: %0.4f, loss_mask: %0.4f , loss_cd_3d: %0.4f , loss_normal: %0.4f , loss_collision: %0.4f , loss_waist: %0.4f '%(i, loss.item(), loss_strain.item(), loss_bending.item(), loss_gravity.item(), loss_mask.item(), loss_cd_3d.item(), loss_normal.item(), loss_collision.item(), loss_waist.item()))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    cloth_pose_new = cloth_pose.copy()
    cloth_pose_new.vertices = verts_cloth_new.detach().cpu().numpy()

    return cloth_pose_new, img_mask

def remesh(cloth_pose, cloth_state, body_mesh):
    
    idx_boundary_v, _ = select_boundary(cloth_pose)
    
    vb = torch.FloatTensor(body_mesh.vertices).cuda()
    nb = torch.FloatTensor(body_mesh.vertex_normals).cuda()

    verts_cloth = torch.FloatTensor(cloth_pose.vertices).cuda()
    tri_center_cloth = torch.FloatTensor(cloth_pose.triangles_center).cuda()
    faces_cloth = torch.LongTensor(cloth_pose.faces).cuda()
    normals_cloth = torch.FloatTensor(cloth_pose.face_normals).cuda()
    valid_fn = torch.isnan(normals_cloth).sum(dim=-1) == 0
    
    verts_boundary = torch.FloatTensor(cloth_pose.vertices[idx_boundary_v]).cuda()
    idx_boundary = torch.LongTensor(idx_boundary_v).cuda()
    
    offset = torch.randn(verts_cloth.shape).cuda()*0.001*0
    offset.requires_grad = True
    lr = 1e-3
    eps = 1e-3
    optimizer = torch.optim.Adam([{'params': offset, 'lr': lr},])
    
    iters = 1000
    for i in range(iters):
        
        verts_cloth_new = verts_cloth + offset
        loss_waist, _ = chamfer_distance(verts_boundary.unsqueeze(0), verts_cloth_new[idx_boundary].unsqueeze(0))
        if i < 300:
            loss_waist *= 100
        else:
            loss_waist *= 10000

        loss_strain = stretching_energy(verts_cloth_new.unsqueeze(0), cloth_state)
        loss_bending = bending_energy(verts_cloth_new.unsqueeze(0), cloth_state)*5
        loss_gravity = gravitational_energy(verts_cloth_new.unsqueeze(0), cloth_state.v_mass)

        tri_full = verts_cloth_new[faces_cloth.reshape(-1)].reshape(-1,3,3)
        tri_center = tri_full.mean(dim=1)
        vec1 = tri_full[:,1] - tri_full[:,0]
        vec2 = tri_full[:,2] - tri_full[:,0]
        normal_full = torch.cross(vec1, vec2, dim=-1)
        normal_full = normal_full/normal_full.norm(p=2, dim=-1, keepdim=True)

        valid_fn_pred = torch.isnan(normal_full).sum(dim=-1) == 0
        _valid_fn = torch.logical_and(valid_fn_pred, valid_fn)

        loss_cd_3d, _ = chamfer_distance(verts_cloth.unsqueeze(0), verts_cloth_new.unsqueeze(0))
        if i < 300:
            loss_cd_3d *= 100
        else:
            loss_cd_3d *= 10000

        _, loss_normal_3d = chamfer_distance(tri_center[_valid_fn].unsqueeze(0), tri_center_cloth[_valid_fn].unsqueeze(0), x_normals=normal_full[_valid_fn].unsqueeze(0), y_normals=normals_cloth[_valid_fn].unsqueeze(0), abs_normal=True)
        if i < 300:
            loss_normal_3d /= 100

        loss_collision = collision_penalty(verts_cloth_new.unsqueeze(0), vb.unsqueeze(0), nb.unsqueeze(0), eps=eps)

        loss = (loss_bending + loss_strain/2 + loss_gravity) + loss_cd_3d + loss_waist + loss_normal_3d + loss_collision
        print('iter: %3d, loss: %0.4f, loss_strain: %0.4f, loss_bending: %0.4f, loss_gravity: %0.4f, loss_cd_3d: %0.4f , loss_waist: %0.4f  , loss_normal_3d: %0.4f , loss_collision: %0.4f '%(i, loss.item(), loss_strain.item(), loss_bending.item(), loss_gravity.item(), loss_cd_3d.item(), loss_waist.item(), loss_normal_3d.item(), loss_collision.item()))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    cloth_pose_remesh = cloth_pose.copy()
    cloth_pose_remesh.vertices = verts_cloth_new.detach().cpu().numpy()

    return cloth_pose_remesh