import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,
    TexturesVertex,
    blending,
)

from .rasterize import get_pix_to_face


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


from typing import NamedTuple, Sequence
class BlendParams_blackBG(NamedTuple):
    sigma: float = 1e-4
    gamma: float = 1e-4
    background_color: Sequence = (0.0, 0.0, 0.0)


class cleanShader(torch.nn.Module):
    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs):

        # get renderer output
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = blending.softmax_rgb_blend(texels, fragments, blend_params, znear=-256, zfar=256)

        return images

def get_render(render_res=256, is_back=False, is_soft=False, return_transform=False):
    dis = 100.0
    scale = 100
    mesh_y_center = 0.0
    cam_pos = torch.tensor([
                    (0, mesh_y_center, dis),
                    (0, mesh_y_center, -dis),
                ])
    R, T = look_at_view_transform(
        eye=cam_pos[[0]] if not is_back else cam_pos[[1]],
        at=((0, mesh_y_center, 0), ),
        up=((0, 1, 0), ),
    )

    cameras = FoVOrthographicCameras(
        device=device,
        R=R,
        T=T,
        znear=100.0,
        zfar=-100.0,
        max_y=100.0,
        min_y=-100.0,
        max_x=100.0,
        min_x=-100.0,
        scale_xyz=(scale * np.ones(3), ) * len(R),
    )

    if is_soft:
        raster_settings = RasterizationSettings(
            image_size=render_res, 
            blur_radius=np.log(1. / 1e-4)*1e-5, 
            faces_per_pixel=50, 
            max_faces_per_bin=500000,
            perspective_correct=False,
        )

    else:
        sigma = 1e-7
        raster_settings = RasterizationSettings(
            image_size=render_res, 
            blur_radius=np.log(1. / 1e-4)*sigma, 
            faces_per_pixel=1, 
            max_faces_per_bin=500000,
            perspective_correct=False,
        )

    meshRas = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

    renderer_textured = MeshRenderer(
        rasterizer=meshRas,
        shader=cleanShader(blend_params=BlendParams_blackBG())
    )

    if return_transform:
        transform = cameras.get_full_projection_transform()
        return meshRas, renderer_textured, transform

    return meshRas, renderer_textured

def face_to_uv_coord(pix_to_face):
    x, y = np.where(pix_to_face != -1)
    coord_img = np.stack((x,y), axis=-1)

    faces = pix_to_face[x, y]

    return coord_img, faces

def render_segmentation(body, renderer_textured_hard, raster, color_smpl):

    verts = torch.FloatTensor(body.vertices).cuda()
    faces = torch.LongTensor(body.faces).cuda()

    pix_to_face = get_pix_to_face(verts, faces, raster)
    pix_to_face = pix_to_face.detach().cpu().numpy()
    coord_img, coord_faces = face_to_uv_coord(pix_to_face)

    body_seg = np.zeros((len(pix_to_face), len(pix_to_face), 3))
    body_seg[coord_img[:,0], coord_img[:,1]] = color_smpl[coord_faces]
    body_seg = np.round(body_seg*255).astype(np.uint8)

    return body_seg

def render_depth(body, renderer_textured_hard, flip_bg=False):
    verts = torch.FloatTensor(body.vertices).cuda()
    faces = torch.LongTensor(body.faces).cuda()

    depth = body.vertices[:, [-1]]
    depth = np.concatenate((depth,depth,depth), axis=-1)
    depth = torch.FloatTensor(depth).cuda()

    textures_depth = TexturesVertex(verts_features=depth[None])
    mesh_depth = Meshes(
        verts=[verts],   
        faces=[faces],
        textures=textures_depth
    )
    
    images_depth = renderer_textured_hard(mesh_depth)
    mask_depth = images_depth[0, :, :, -1].detach().cpu().numpy() <= 0
    images_depth = images_depth[0, :, :, :1].detach().cpu().numpy()
    if flip_bg:
        images_depth[mask_depth] = 1
    else:
        images_depth[mask_depth] = -1
    return images_depth

def render_torsor(body, renderer_textured_hard, raster, color_smpl_raw):

    verts = torch.FloatTensor(body.vertices).cuda()
    faces = torch.LongTensor(body.faces).cuda()

    pix_to_face = get_pix_to_face(verts, faces, raster)
    pix_to_face = pix_to_face.detach().cpu().numpy()
    coord_img, coord_faces = face_to_uv_coord(pix_to_face)

    body_torsor = np.zeros((len(pix_to_face), len(pix_to_face), 3))
    body_torsor[coord_img[:,0], coord_img[:,1]] = color_smpl_raw[coord_faces]
    body_torsor = np.logical_or(body_torsor==1, body_torsor==2)

    return body_torsor

def render_depth_discrete(body, renderer_textured_hard, raster, flip_bg=False):
    
    verts = torch.FloatTensor(body.vertices).cuda()
    faces = torch.LongTensor(body.faces).cuda()
    tri_depth = torch.FloatTensor(body.triangles_center).cuda()[:,-1]

    pix_to_face = get_pix_to_face(verts, faces, raster)
    pix_to_face = pix_to_face.detach().cpu().numpy()
    coord_img, coord_faces = face_to_uv_coord(pix_to_face)

    body_depth = np.zeros((len(pix_to_face), len(pix_to_face))) - 1
    body_depth[coord_img[:,0], coord_img[:,1]] = tri_depth[coord_faces].detach().cpu().numpy()

    mask_depth = body_depth == -1
    if flip_bg:
        body_depth[mask_depth] = 1
        
    return body_depth