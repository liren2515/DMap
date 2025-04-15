import numpy as np 
import trimesh
import torch
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

def create_uv_mesh(x_res, y_res, debug=False):
    x = np.linspace(1, -1, x_res)
    y = np.linspace(1, -1, y_res)

    # exchange x,y to make everything consistent:
    # x is the first coordinate, y is the second!
    xv, yv = np.meshgrid(y, x)
    uv = np.stack((xv, yv), axis=-1)

    vertices = uv.reshape(-1, 2)
    
    tri = Delaunay(vertices)
    faces = tri.simplices
    vertices = np.concatenate((vertices, np.zeros((len(vertices), 1))), axis=-1)

    if debug:
        # x in plt is vertical
        # y in plt is horizontal
        plt.figure()
        plt.triplot(vertices[:,0], vertices[:,1], faces)
        plt.plot(vertices[:,0], vertices[:,1], 'o', markersize=2)
        plt.savefig('../tmp/tri.png')

    return vertices, faces

def barycentric_faces(mesh_query, mesh_base):
    v_query = mesh_query.vertices
    base = trimesh.proximity.ProximityQuery(mesh_base)
    closest_pt, _, closest_face_idx = base.on_surface(v_query)
    triangles = mesh_base.triangles[closest_face_idx]
    v_barycentric = trimesh.triangles.points_to_barycentric(triangles, closest_pt)
    return v_barycentric, closest_face_idx

def get_barycentric(mesh, points):
    mesh_base = trimesh.proximity.ProximityQuery(mesh)
    closest_points, _, idx_f = mesh_base.on_surface(points)
    
    triangles = mesh.vertices[mesh.faces[idx_f]]
    barycentric = trimesh.triangles.points_to_barycentric(triangles, closest_points)
    return barycentric, idx_f

def uv_to_3D(pattern_deform, uv_faces, barycentric_uv, closest_face_idx_uv):
    uv_faces_id = uv_faces[closest_face_idx_uv]
    uv_faces_id = uv_faces_id.reshape(-1)

    pattern_deform_triangles = pattern_deform[uv_faces_id].reshape(-1, 3, 3)
    pattern_deform_bary = (pattern_deform_triangles * barycentric_uv[:, :, None]).sum(axis=-2)
    return pattern_deform_bary

########### fix ill-shaped triangles on boundaries ###########
try:
    import pymesh
except:
    pass
def repair_pattern(mesh_trimesh, res=128):

    mesh = pymesh.form_mesh(mesh_trimesh.vertices, mesh_trimesh.faces)
    count = 0
    target_len_long = 2/res*np.sqrt(2)*1.2
    target_len_short = 2/res*0.4
    print('before fixing, #v - ', mesh.num_vertices)
    mesh, __ = pymesh.split_long_edges(mesh, target_len_long)

    num_vertices = mesh.num_vertices
    print('split long edges, #v - ', num_vertices)
    while True:
        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6)
        mesh, __ = pymesh.collapse_short_edges(mesh, target_len_short, preserve_feature=True)
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 120.0, 100)
        if mesh.num_vertices == num_vertices:
            break

        num_vertices = mesh.num_vertices
        print("fix iter, #v - ", num_vertices)
        count += 1
        if count > 10: break

    mesh_trimesh_new  = trimesh.Trimesh(mesh.vertices, mesh.faces, validate=False, process=False)

    return mesh_trimesh_new


########### meshing ###########
from .mesh_reader import read_mesh_from_sdf_test_batch_v2_with_label, triangulation_seam_v2
def reconstruct_pattern_with_label(model_isp, latent_code, uv_vertices, uv_faces, edges, resolution=256, using_repair=True):
    model_sdf_f, model_sdf_b, model_atlas_f, model_atlas_b = model_isp
    with torch.no_grad():
        uv_faces_torch_f = torch.LongTensor(uv_faces).cuda()
        uv_faces_torch_b = torch.LongTensor(uv_faces[:,[0,2,1]]).cuda()
        vertices_new_f = uv_vertices[:,:2].clone()
        vertices_new_b = uv_vertices[:,:2].clone()

        uv_input = uv_vertices[:,:2]*10
        num_points = len(uv_vertices)
        latent_code = latent_code.repeat(num_points, 1)
        pred_f = model_sdf_f(uv_input, latent_code)
        pred_b = model_sdf_b(uv_input, latent_code)
        sdf_pred_f = pred_f[:, 0]
        sdf_pred_b = pred_b[:, 0]
        label_f = pred_f[:, 1:]
        label_b = pred_b[:, 1:]
        label_f = torch.argmax(label_f, dim=-1)
        label_b = torch.argmax(label_b, dim=-1)

        sdf_pred = torch.stack((sdf_pred_f, sdf_pred_b), dim=0)
        uv_vertices_batch = torch.stack((uv_vertices[:,:2], uv_vertices[:,:2]), dim=0)
        label_pred = torch.stack((label_f, label_b), dim=0)
        vertices_new, faces_list, labels_list = read_mesh_from_sdf_test_batch_v2_with_label(uv_vertices_batch, uv_faces_torch_f, sdf_pred, label_pred, edges, reorder=True, thresh=-1e-3)
        vertices_new_f = vertices_new[0]
        vertices_new_b = vertices_new[1]
        faces_new_f = faces_list[0]
        faces_new_b = faces_list[1][:,[0,2,1]]
        label_new_f = labels_list[0]
        label_new_b = labels_list[1]

        v_f = np.zeros((len(vertices_new_f), 3))
        v_b = np.zeros((len(vertices_new_b), 3))
        v_f[:, :2] = vertices_new_f
        v_b[:, :2] = vertices_new_b
        mesh_pattern_f = trimesh.Trimesh(v_f, faces_new_f, validate=False, process=False)
        mesh_pattern_b = trimesh.Trimesh(v_b, faces_new_b, validate=False, process=False)
        if using_repair:
            print('repair mesh_pattern_f')
            mesh_pattern_f = repair_pattern(mesh_pattern_f, res=resolution)
            print('repair mesh_pattern_b')
            mesh_pattern_b = repair_pattern(mesh_pattern_b, res=resolution)
            
        
        pattern_vertices_f = torch.FloatTensor(mesh_pattern_f.vertices).cuda()[:,:2]
        pattern_vertices_b = torch.FloatTensor(mesh_pattern_b.vertices).cuda()[:,:2]

        pred_f = model_sdf_f(pattern_vertices_f*10, latent_code[:len(pattern_vertices_f)])
        pred_b = model_sdf_b(pattern_vertices_b*10, latent_code[:len(pattern_vertices_b)])
        label_new_f = pred_f[:, 1:]
        label_new_b = pred_b[:, 1:]
        label_new_f = torch.argmax(label_new_f, dim=-1).cpu().numpy()
        label_new_b = torch.argmax(label_new_b, dim=-1).cpu().numpy()

        pred_atlas_f = model_atlas_f(pattern_vertices_f*10, latent_code[:len(pattern_vertices_f)])/10
        pred_atlas_b = model_atlas_b(pattern_vertices_b*10, latent_code[:len(pattern_vertices_b)])/10

        mesh_atlas_f = trimesh.Trimesh(pred_atlas_f.cpu().numpy(), mesh_pattern_f.faces, process=False, valid=False)
        mesh_atlas_b = trimesh.Trimesh(pred_atlas_b.cpu().numpy(), mesh_pattern_b.faces, process=False, valid=False)

        idx_boundary_v_f, boundary_edges_f = select_boundary(mesh_pattern_f)
        idx_boundary_v_b, boundary_edges_b = select_boundary(mesh_pattern_b)
        boundary_edges_f = set([tuple(sorted(e)) for e in boundary_edges_f.tolist()])
        boundary_edges_b = set([tuple(sorted(e)) for e in boundary_edges_b.tolist()])
        label_boundary_v_f = label_new_f[idx_boundary_v_f]
        label_boundary_v_b = label_new_b[idx_boundary_v_b]

    return mesh_atlas_f, mesh_atlas_b, mesh_pattern_f, mesh_pattern_b, label_new_f, label_new_b


########### Sewing ###########
from .cutting import select_boundary, connect_2_way, one_ring_neighour
def sewing_vertical(mesh_atlas_f, mesh_atlas_b, idx_boundary_v, boundary_edges, label_boundary_v, mesh_pattern, labels, seam_i, return_seam_top=False):
    idx_boundary_v_f, idx_boundary_v_b = idx_boundary_v
    boundary_edges_f, boundary_edges_b = boundary_edges
    label_boundary_v_f, label_boundary_v_b = label_boundary_v
    mesh_pattern_f, mesh_pattern_b = mesh_pattern
    labels_f, labels_b = labels

    indicator_seam_f = label_boundary_v_f == seam_i
    indicator_seam_b = label_boundary_v_b == seam_i
    
    idx_seam_v_f = idx_boundary_v_f[indicator_seam_f]
    idx_seam_v_b = idx_boundary_v_b[indicator_seam_b]
    
    one_rings_seam_f = one_ring_neighour(idx_seam_v_f, mesh_pattern_f, is_dic=True, mask_set=set(idx_seam_v_f))
    one_rings_seam_b = one_ring_neighour(idx_seam_v_b, mesh_pattern_b, is_dic=True, mask_set=set(idx_seam_v_b))
    
    path_seam_f, _ = connect_2_way(set(idx_seam_v_f), one_rings_seam_f, boundary_edges_f)
    path_seam_b, _ = connect_2_way(set(idx_seam_v_b), one_rings_seam_b, boundary_edges_b)

    if mesh_pattern_f.vertices[path_seam_f[0], 1] < mesh_pattern_f.vertices[path_seam_f[-1], 1]: # high to low
        path_seam_f = path_seam_f[::-1]
    if mesh_pattern_b.vertices[path_seam_b[0], 1] < mesh_pattern_b.vertices[path_seam_b[-1], 1]:
        path_seam_b = path_seam_b[::-1]

    idx_offset = len(mesh_pattern_f.vertices)

    faces_seam = triangulation_seam_v2(mesh_atlas_f, mesh_atlas_b, path_seam_f, path_seam_b, idx_offset, reverse=False)

    if return_seam_top:
        return faces_seam, [path_seam_f[0], path_seam_b[0]+idx_offset]
    else:
        return faces_seam

def sewing_vertical_tshirt(mesh_atlas_f, mesh_atlas_b, idx_boundary_v, boundary_edges, label_boundary_v, mesh_pattern, labels, seam_i, return_seam_top=False, horizontal=False):
    idx_boundary_v_f, idx_boundary_v_b = idx_boundary_v
    boundary_edges_f, boundary_edges_b = boundary_edges
    label_boundary_v_f, label_boundary_v_b = label_boundary_v
    mesh_pattern_f, mesh_pattern_b = mesh_pattern
    labels_f, labels_b = labels

    indicator_seam_f = label_boundary_v_f == seam_i
    indicator_seam_b = label_boundary_v_b == seam_i
    
    idx_seam_v_f = idx_boundary_v_f[indicator_seam_f]
    idx_seam_v_b = idx_boundary_v_b[indicator_seam_b]
    
    one_rings_seam_f = one_ring_neighour(idx_seam_v_f, mesh_pattern_f, is_dic=True, mask_set=set(idx_seam_v_f))
    one_rings_seam_b = one_ring_neighour(idx_seam_v_b, mesh_pattern_b, is_dic=True, mask_set=set(idx_seam_v_b))
    
    path_seam_f, _ = connect_2_way(set(idx_seam_v_f), one_rings_seam_f, boundary_edges_f)
    path_seam_b, _ = connect_2_way(set(idx_seam_v_b), one_rings_seam_b, boundary_edges_b)

    if horizontal:
        if mesh_pattern_f.vertices[path_seam_f[0], 0] > mesh_pattern_f.vertices[path_seam_f[-1], 0]: # left to right
            path_seam_f = path_seam_f[::-1]
        if mesh_pattern_b.vertices[path_seam_b[0], 0] > mesh_pattern_b.vertices[path_seam_b[-1], 0]:
            path_seam_b = path_seam_b[::-1]
    
    else:
        if mesh_pattern_f.vertices[path_seam_f[0], 1] < mesh_pattern_f.vertices[path_seam_f[-1], 1]: # high to low
            path_seam_f = path_seam_f[::-1]
        if mesh_pattern_b.vertices[path_seam_b[0], 1] < mesh_pattern_b.vertices[path_seam_b[-1], 1]:
            path_seam_b = path_seam_b[::-1]

    idx_offset = len(mesh_pattern_f.vertices)

    faces_seam = triangulation_seam_v2(mesh_atlas_f, mesh_atlas_b, path_seam_f, path_seam_b, idx_offset, reverse=False)

    if return_seam_top:
        return faces_seam, [path_seam_f[0], path_seam_b[0]+idx_offset]
    else:
        return faces_seam

def compute_offset(idx_boundary_v_f, label_boundary_v_f, mesh_atlas_f, ratio=0.001):
    idx_seam_v_f_1 = idx_boundary_v_f[label_boundary_v_f == 1]
    idx_seam_v_f_2 = idx_boundary_v_f[label_boundary_v_f == 2]
    seam_v_f_1 = mesh_atlas_f.vertices[idx_seam_v_f_1]
    seam_v_f_2 = mesh_atlas_f.vertices[idx_seam_v_f_2]

    highest_1_i = np.argmax(seam_v_f_1[:, 1].flatten())
    highest_2_i = np.argmax(seam_v_f_2[:, 1].flatten())

    offset = ((seam_v_f_1[highest_1_i] - seam_v_f_2[highest_2_i])**2).sum()*ratio
    return offset

def compute_offset_tshirt(idx_boundary_v_f, label_boundary_v_f, mesh_atlas_f, ratio=0.001):
    idx_seam_v_f_0 = idx_boundary_v_f[label_boundary_v_f == 0]
    seam_v_f_0 = mesh_atlas_f.vertices[idx_seam_v_f_0]

    highest_0_i = np.argmax(seam_v_f_0[:, -1].flatten())
    lowest_0_i = np.argmin(seam_v_f_0[:, -1].flatten())

    offset = ((seam_v_f_0[highest_0_i] - seam_v_f_0[lowest_0_i])**2).sum()*ratio
    return offset

def compute_offset_trousers(idx_boundary_v_f, label_boundary_v_f, mesh_atlas_f, ratio=0.001):
    idx_seam_v_f_1 = idx_boundary_v_f[label_boundary_v_f == 1]
    idx_seam_v_f_2 = idx_boundary_v_f[label_boundary_v_f == 4]
    seam_v_f_1 = mesh_atlas_f.vertices[idx_seam_v_f_1]
    seam_v_f_2 = mesh_atlas_f.vertices[idx_seam_v_f_2]

    highest_1_i = np.argmax(seam_v_f_1[:, 1].flatten())
    highest_2_i = np.argmax(seam_v_f_2[:, 1].flatten())

    offset = ((seam_v_f_1[highest_1_i] - seam_v_f_2[highest_2_i])**2).sum()*ratio
    return offset

def sewing_front_back(garment_type, mesh_pattern_f, mesh_pattern_b, mesh_atlas_f, mesh_atlas_b, labels_f, labels_b, num_seams=2):

    idx_boundary_v_f, boundary_edges_f = select_boundary(mesh_pattern_f)
    idx_boundary_v_b, boundary_edges_b = select_boundary(mesh_pattern_b)
    boundary_edges_f = set([tuple(sorted(e)) for e in boundary_edges_f.tolist()])
    boundary_edges_b = set([tuple(sorted(e)) for e in boundary_edges_b.tolist()])
    label_boundary_v_f = labels_f[idx_boundary_v_f]
    label_boundary_v_b = labels_b[idx_boundary_v_b]

    idx_boundary_v = [idx_boundary_v_f, idx_boundary_v_b]
    boundary_edges = [boundary_edges_f, boundary_edges_b]
    label_boundary_v = [label_boundary_v_f, label_boundary_v_b]
    mesh_pattern = [mesh_pattern_f, mesh_pattern_b]
    labels = [labels_f, labels_b]

    idx_offset = len(mesh_pattern_f.vertices)
    faces_sewing = [mesh_atlas_f.faces, mesh_atlas_b.faces + idx_offset]
    seam_tops = []

    if garment_type == 'Skirt':
        faces_flip = [True, False]
    elif garment_type == 'Tshirt' or garment_type == 'Jacket':
        faces_flip = [True, False, False, False]
        horizontal = [False, False, True, True]
    elif garment_type == 'Trousers':
        faces_flip = [True, False, True, False]

    for seam_i in range(1, num_seams+1):
        if garment_type == 'Skirt':
            faces_seam_i = sewing_vertical(mesh_atlas_f, mesh_atlas_b, idx_boundary_v, boundary_edges, label_boundary_v, mesh_pattern, labels, seam_i)
        elif garment_type == 'Tshirt' or garment_type == 'Jacket':
            faces_seam_i = sewing_vertical_tshirt(mesh_atlas_f, mesh_atlas_b, idx_boundary_v, boundary_edges, label_boundary_v, mesh_pattern, labels, seam_i, horizontal=horizontal[seam_i-1])
        elif garment_type == 'Trousers':
            if seam_i == 2 or seam_i == 3:
                faces_seam_i, seam_i_top = sewing_vertical(mesh_atlas_f, mesh_atlas_b, idx_boundary_v, boundary_edges, label_boundary_v, mesh_pattern, labels, seam_i, return_seam_top=True)
                seam_tops.append(seam_i_top)
            else:
                faces_seam_i = sewing_vertical(mesh_atlas_f, mesh_atlas_b, idx_boundary_v, boundary_edges, label_boundary_v, mesh_pattern, labels, seam_i)

        if faces_flip[seam_i-1]:
            faces_seam_i = faces_seam_i[:,::-1]
        faces_sewing.append(faces_seam_i)

    if len(seam_tops) > 0:
        faces_extra = np.array([seam_tops[0]+[seam_tops[1][0]], [seam_tops[0][1]]+seam_tops[1][::-1]])
        faces_sewing.append(faces_extra)

    if garment_type == 'Skirt':
        z_offset = compute_offset(idx_boundary_v_f, label_boundary_v_f, mesh_atlas_f, ratio=0.1)
    elif garment_type == 'Tshirt' or garment_type == 'Jacket':
        z_offset = compute_offset_tshirt(idx_boundary_v_f, label_boundary_v_f, mesh_atlas_f, ratio=0.01)
    elif garment_type == 'Trousers':
        z_offset = compute_offset_trousers(idx_boundary_v_f, label_boundary_v_f, mesh_atlas_f, ratio=0.1)

    mesh_atlas_f.vertices[:, -1] += z_offset
    verts_sewing = np.concatenate((mesh_atlas_f.vertices, mesh_atlas_b.vertices), axis=0)
    faces_sewing = np.concatenate(faces_sewing, axis=0)
    mesh_sewing = trimesh.Trimesh(verts_sewing, faces_sewing, validate=False, process=False)

    labels_sewing = np.concatenate((labels_f, labels_b), axis=0)

    return mesh_sewing, labels_sewing