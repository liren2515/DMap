import os, sys
import cv2
import numpy as np
import trimesh
import open3d as o3d

def _process_depth(image, resize=True):
    if resize:
        image = cv2.resize(image, (192, 192), interpolation=cv2.INTER_NEAREST)
    if len(image.shape) != 3:
        image = np.expand_dims(image, axis=-1)
    return image

def _process_seg(image, resize=True):
    if resize:
        image = cv2.resize(image, (192, 192), interpolation=cv2.INTER_NEAREST)
    image = image.astype(np.float32)/255
    image = (image - 0.5)/0.5
    return image

def _process_image(image, resize=True):
    if resize:
        image = cv2.resize(image, (192, 192), interpolation=cv2.INTER_NEAREST)
    image_raw = image.copy()
    image = image.astype(np.float32)/255
    image = (image - 0.5)/0.5
    return image, image_raw

def get_mask_label(garment_type):
    if garment_type == 'Skirt':
        target_label = 240
    elif garment_type == 'Tshirt':
        target_label = 60
    elif garment_type == 'Trousers':
        target_label = 180
    elif garment_type == 'Jacket':
        target_label = 120
    else:
        raise NotImplementedError

    return target_label

def draw_uv_mapping(images, coor, size_uv=256):
    # Left and right are reversed.
    offset = (size_uv - 1.0)/2
    uv = images[:,:,:,:2]*2 -1
    fb = images[:,:,:,-1]*2 -1
    uv = -uv*offset + offset
    
    uv_mapping = []
    uv_mapping_mask = []
    for i in range(len(uv)):
        uv_i = uv[i]
        fb_i = fb[i]
        coor_i = coor[i]

        uv_i = uv_i[coor_i[:, 0], coor_i[:, 1]]
        fb_i = fb_i[coor_i[:, 0], coor_i[:, 1]]

        idx_front = fb_i > 0
        idx_back = fb_i < 0

        img_front = np.zeros((size_uv, size_uv, 3))
        img_back = np.zeros((size_uv, size_uv, 3))
        img_front_mask = np.zeros((size_uv, size_uv, 3))
        img_back_mask = np.zeros((size_uv, size_uv, 3))

        uv_i = np.round(uv_i).astype(int)
        uv_i_f = uv_i[idx_front]
        uv_i_b = uv_i[idx_back]
        coor_i_f = coor_i[idx_front]
        coor_i_b = coor_i[idx_back]

        y_f, x_f = uv_i_f[:, 0], uv_i_f[:, 1]
        y_b, x_b = uv_i_b[:, 0], uv_i_b[:, 1]
        img_front[x_f, y_f, :2] = coor_i_f
        img_back[x_b, y_b, :2] = coor_i_b

        img_front_mask[x_f, y_f] = 255
        img_back_mask[x_b, y_b] = 255

        img = np.concatenate((img_front, img_back), axis=1).astype(np.uint8)
        img_mask = np.concatenate((img_front_mask, img_back_mask), axis=1).astype(np.uint8)
        uv_mapping.append(img)
        uv_mapping_mask.append(img_mask)

    uv_mapping = np.stack(uv_mapping, axis=0)
    uv_mapping_mask = np.stack(uv_mapping_mask, axis=0)
    return uv_mapping, uv_mapping_mask

def mask_to_coord(mask):
    coords = []

    for i in range(len(mask)):
        x, y = np.where(mask[i])
        coord = np.stack((x,y), axis=-1)
        coords.append(coord)

    return coords

def _to_xyz(coord_img, z, img_size=191.):
    scale = img_size/2
    yx = (coord_img - scale)/scale
    y, x = -yx[:,0], yx[:,1]

    xyz = np.stack((x,y,z), axis=-1)
    return xyz

def remove_arm(color_smpl_faces):
    new_faces_id = []
    for i in range(len(color_smpl_faces)):
        if color_smpl_faces[i,0] in [3, 4, 11, 12, 13, 14]:
            continue
        else:
            new_faces_id.append(i)

    return new_faces_id


from scipy.spatial import cKDTree
def fill_background_with_nearest_foreground(background_img, indicator_img):
    # Find the indices of background pixels in the indicator image
    background_indices = np.argwhere(indicator_img == 0)
    
    # Find the indices of foreground pixels in the indicator image
    foreground_indices = np.argwhere(indicator_img != 0)
    
    # Build a KDTree using the foreground pixel indices
    tree = cKDTree(foreground_indices)
    
    # For each background pixel, find the nearest foreground pixel and update its value
    for bg_index in background_indices:
        _, nearest_fg_index = tree.query(bg_index)
        background_img[tuple(bg_index)] = background_img[tuple(foreground_indices[nearest_fg_index])]
    
    return background_img



def _to_uv_FB(garment_type, image_F, image_B, coord_img_F, coord_img_B, xyz_F, xyz_B, size_uv=128):
    offset = (size_uv - 1.0)/2
    uv_F = image_F[:,:,:2]
    uv_B = image_B[:,:,:2]
    fb_F = image_F[:,:,-1]
    fb_B = image_B[:,:,-1]
    uv_F = -uv_F*offset + offset
    uv_B = -uv_B*offset + offset

    uv_F = uv_F[coord_img_F[:, 0], coord_img_F[:, 1]]
    uv_B = uv_B[coord_img_B[:, 0], coord_img_B[:, 1]]
    fb_F = fb_F[coord_img_F[:, 0], coord_img_F[:, 1]]
    fb_B = fb_B[coord_img_B[:, 0], coord_img_B[:, 1]]
    idx_front_F = fb_F > 0
    idx_front_B = fb_B > 0
    idx_back_F = fb_F < 0
    idx_back_B = fb_B < 0

    img_front = np.zeros((size_uv, size_uv, 3)) - 1
    img_back = np.zeros((size_uv, size_uv, 3)) - 1
    img_front_mask = np.zeros((size_uv, size_uv))
    img_back_mask = np.zeros((size_uv, size_uv))

    uv_F = np.round(uv_F).astype(int)
    uv_f_F = uv_F[idx_front_F]
    uv_b_F = uv_F[idx_back_F]
    xyz_f_F = xyz_F[idx_front_F]
    xyz_b_F = xyz_F[idx_back_F]

    uv_B = np.round(uv_B).astype(int)
    uv_f_B = uv_B[idx_front_B]
    uv_b_B = uv_B[idx_back_B]
    xyz_f_B = xyz_B[idx_front_B]
    xyz_b_B = xyz_B[idx_back_B]

    y_f_F, x_f_F = uv_f_F[:, 0], uv_f_F[:, 1]
    y_b_F, x_b_F = uv_b_F[:, 0], uv_b_F[:, 1]

    y_f_B, x_f_B = uv_f_B[:, 0], uv_f_B[:, 1]
    y_b_B, x_b_B = uv_b_B[:, 0], uv_b_B[:, 1]
    
    if garment_type == 'Jacket':
        img_front[x_f_F, y_f_F] = xyz_f_F
        img_back[x_b_F, y_b_F] = xyz_b_F
        img_front[x_f_B, y_f_B] = xyz_f_B
        img_back[x_b_B, y_b_B] = xyz_b_B
    else:
        img_front[x_f_B, y_f_B] = xyz_f_B
        img_back[x_b_B, y_b_B] = xyz_b_B
        img_front[x_f_F, y_f_F] = xyz_f_F
        img_back[x_b_F, y_b_F] = xyz_b_F

    img_front_mask[x_f_F, y_f_F] = 1
    img_back_mask[x_b_F, y_b_F] = 1
    img_front_mask[x_f_B, y_f_B] = 1
    img_back_mask[x_b_B, y_b_B] = 1

    sparse_uv = np.concatenate((img_front, img_back), axis=1)
    sparse_mask = np.concatenate((img_front_mask, img_back_mask), axis=1)

    return sparse_uv, sparse_mask

def clean_uv(uv_mask):
    invalid_uv = np.where(uv_mask == 0)[0]
    valid_uv = np.where(uv_mask != 0)[0]
    return set(valid_uv.flatten().tolist())

def filter_faces(faces, valid_v):
    faces_new = []
    for f in faces:
        if f[0] in valid_v and f[1] in valid_v and f[2] in valid_v:
            faces_new.append(f)

    faces_new = np.array(faces_new).astype(int)
    return faces_new

def dilate_indicator(mask, size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    mask = cv2.dilate(mask, kernel)
    return mask

def erode_indicator(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, kernel)
    return mask

def project_waist(body, barycentric, idx_f, eps=1e-3):

    faces_waist = body.faces[idx_f].reshape(-1)
    fn_waist = body.face_normals[idx_f]

    triangles = body.vertices[faces_waist].reshape(-1, 3, 3)
    v_waist = trimesh.triangles.barycentric_to_points(triangles, barycentric)
    v_waist += fn_waist*eps

    return v_waist


def rescale(cloth_pose_f, cloth_pose_b, altas_f, altas_b):
    # cloth_pose_f, cloth_pose_b - trimesh
    ave_area_pose_f = cloth_pose_f.area_faces.mean() 
    ave_area_pose_b = cloth_pose_b.area_faces.mean() 
    ave_area_rest_f = altas_f.area_faces.mean() 
    ave_area_rest_b = altas_b.area_faces.mean() 

    scale = (ave_area_pose_f/ave_area_rest_f + ave_area_pose_b/ave_area_rest_b)/2
    scale = np.sqrt(scale)
    return scale

def clean_pt(path, nb_neighbors=10, std_ratio=0.01):
    if type(path) == str:
        pcd = o3d.io.read_point_cloud(path)
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(path)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    inlier_cloud = pcd.select_by_index(ind)
    vertices = np.asarray(inlier_cloud.points)
    pt = trimesh.Trimesh(vertices)
    return pt, ind