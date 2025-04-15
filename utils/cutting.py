import os, sys
import trimesh
import networkx as nx
import numpy as np

def select_boundary(mesh):
    unique_edges = mesh.edges[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)]
    idx_boundary_v = np.unique(unique_edges.flatten())
    return idx_boundary_v, unique_edges

def connect_2_way(idx_boundary_v_set, one_rings, boundary_edges):
    path = [list(idx_boundary_v_set)[0]]
    idx_boundary_v_set.remove(path[0])
    # connect one way
    while len(idx_boundary_v_set):
        node = path[-1]
        neighbour = one_rings[node]
        for n in neighbour:
            if n in idx_boundary_v_set and tuple(sorted([node, n])) in boundary_edges:
                path.append(n)
                idx_boundary_v_set.remove(n)
                break

        if node == path[-1]:
            break


    # connect the other way
    while len(idx_boundary_v_set):
        node = path[0]
        neighbour = one_rings[node]
        for n in neighbour:
            if n in idx_boundary_v_set and tuple(sorted([node, n])) in boundary_edges:
                path.insert(0, n)
                idx_boundary_v_set.remove(n)
                break

        if node == path[0]:
            break

    return path, idx_boundary_v_set

def one_ring_neighour(idx_v, mesh, is_dic=False, mask_set=None):
    g = nx.from_edgelist(mesh.edges_unique)
    valid_v_i = set(np.unique(mesh.faces.flatten()).tolist())
    one_ring = []
    if mask_set is not None:
        for i in idx_v:
            if i in valid_v_i:
                one_ring.append(set(g[i].keys()).intersection(mask_set))
            else:
                one_ring.append(set([]))
    else:
        for i in idx_v:
            if i in valid_v_i:
                one_ring.append(set(g[i].keys()))
            else:
                one_ring.append(set([]))

    if is_dic:
        one_ring_dic = {}
        for i in range(len(idx_v)):
            one_ring_dic[idx_v[i]] = one_ring[i]

        one_ring = one_ring_dic
    return one_ring

def get_connected_paths_skirt(mesh):
    idx_boundary_v, boundary_edges = select_boundary(mesh)
    boundary_edges = boundary_edges.tolist()
    boundary_edges = set([tuple(sorted(e)) for e in boundary_edges])
    idx_boundary_v_set = set(idx_boundary_v)
    one_rings = one_ring_neighour(idx_boundary_v, mesh, is_dic=True, mask_set=idx_boundary_v_set)

    paths = []
    path_z_mean = []
    while len(idx_boundary_v_set):
        path, idx_boundary_v_set = connect_2_way(idx_boundary_v_set, one_rings, boundary_edges)
        paths.append(path)
        path_z_mean.append(mesh.vertices[path, -1].mean())

    up_path_i = path_z_mean.index(max(path_z_mean))
    up_path = paths[up_path_i]

    _set = set([0,1])
    _set.remove(up_path_i)

    bottom_path_i = list(_set)[0]
    bottom_path = paths[bottom_path_i] 

    return up_path, bottom_path