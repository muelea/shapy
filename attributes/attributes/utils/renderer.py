import sys
import os
import os.path as osp

import trimesh
import numpy as np
import pyrender

from tqdm import tqdm
import argparse
import PIL.Image as pil_img

class Renderer():
    def __init__(
        self,
        is_registration=False,
        rotation=0,
    ):

        self.is_registration = is_registration
        self.rotation = rotation

        H, W = 1200,1200

        self.material = pyrender.MetallicRoughnessMaterial(
                        doubleSided=True,
                        metallicFactor=0.9,
                        roughnessFactor=0.7,
                        smooth=False,
                        alphaMode='BLEND',
                        baseColorFactor=(0.4, 0.4, 0.4, 1.0), #grey color
                        #baseColorFactor=(1.0, 1.0, 1.0, 1.0)) #white color
                        #baseColorFactor=(0.45, 0.5, 1.0, 1.0)) #blue color
        )
        self.vertex_material = pyrender.MetallicRoughnessMaterial(
            doubleSided=True,
            metallicFactor=0.9,
            roughnessFactor=0.7,
            smooth=False,
            alphaMode='BLEND',
            #baseColorFactor=(1.0, 1.0, 1.0, 1.0)) #white color
            #baseColorFactor=(0.45, 0.5, 1.0, 1.0)) #blue color
        )
        self.material_plane = pyrender.MetallicRoughnessMaterial(
                        doubleSided=True,
                        metallicFactor=0.0,
                        alphaMode='OPAQUE',
                        baseColorFactor=(1.0, 1.0, 0.9, 1.0))

        self.scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0],
                               ambient_light=(0.3, 0.3, 0.3))

        # create camera
        camera = pyrender.PerspectiveCamera(yfov= 0.8 * np.pi / 3.0, aspectRatio=1.0)

        # set camera pose
        camera_pose = np.array(
            [[1, 0, 0, 0],
            [ 0, 1, 0, 1.2],
            [ 0, 0, 1, 2.5],
            [ 0, 0, 0, 1]],
        )
        camera_rot = trimesh.transformations.rotation_matrix(
                      np.radians(355), [1, 0, 0])
        camera_pose[:3,:3] = camera_rot[:3,:3]
        nc = pyrender.Node(camera=camera, matrix=camera_pose)
        self.scene.add_node(nc)

        # 'sun light'
        light = pyrender.light.DirectionalLight(intensity=5)
        sl_pose = np.array(
            [[1, 0, 0, 0],
            [ 0, 1, 0, 1.5],
            [ 0, 0, 1, 6],
            [ 0, 0, 0, 1]],
        )
        sl_rot = trimesh.transformations.rotation_matrix(
             np.radians(340), [1, 0, 0])
        sl_pose[:3,:3] = sl_rot[:3,:3]
        nl = pyrender.Node(light=light, matrix=sl_pose)
        self.scene.add_node(nl)

        xs = 0.5
        sl2_pose = np.eye(4)
        sl2_poses = {
            'pointlight': [[-1.0, 1.0, 1.0], [-1.0, 2.0, 1.0]]
        }

        light = pyrender.PointLight(color=np.ones(3), intensity=6.0)
        for xyz_t in sl2_poses['pointlight']:
            sl2_pose[:3,3] = xyz_t
            nl = pyrender.Node(light=light, matrix=sl2_pose)
            self.scene.add_node(nl)

        # create lights at top
        light = pyrender.PointLight(color=np.ones(3), intensity=1.0)

        for xshift in [-1.5, -1, 1, 1.5]:
            light2_pose = np.array(
              [[1, 0, 0, xshift],
              [ 0, 1, 0, 2.0],
              [ 0, 0, 1, xshift],
              [ 0, 0, 0, 1]],
            )
            for rot in [340,20]:
                light2_rot = trimesh.transformations.rotation_matrix(
                      np.radians(rot), [1, 0, 0])
                light2_pose[:3,:3] = light2_rot[:3,:3]
                nl = pyrender.Node(light=light, matrix=light2_pose)
                self.scene.add_node(nl)

        # add ground plane
        plane_vertices = np.zeros([4, 3], dtype=np.float32)
        ps = 1
        plane_vertices[:, 0] = [-ps, ps, ps, -ps]
        plane_vertices[:, 2] = [-ps, -ps, ps, ps]
        plane_faces = np.array([[0, 1, 2], [0, 2, 3]],
                                   dtype=np.int32).reshape(-1, 3)
        plane_mesh = trimesh.Trimesh(vertices=plane_vertices,
                                         faces=plane_faces)
        pyr_plane_mesh = pyrender.Mesh.from_trimesh(
                        plane_mesh,
                        material=self.material_plane)
        npl = pyrender.Node(mesh=pyr_plane_mesh, matrix=np.eye(4))
        #self.scene.add_node(npl)

        # create renderer
        self.r = pyrender.OffscreenRenderer(viewport_width=W,
                                       viewport_height=H,
                                       point_size=1.0)

    def render(self, mesh, colors=None, vertex_colors=None):

        if self.rotation != 0:
            rot = trimesh.transformations.rotation_matrix(
                  np.radians(rotation), [0, 1, 0])
            mesh.apply_transform(rot)

        if self.is_registration:
            # rotate 90 deg around x axis
            rotaround = 270
            rot = trimesh.transformations.rotation_matrix(
                  np.radians(rotaround), [1, 0, 0])
            mesh.apply_transform(rot)

            # rotate 45 degree around y axis
            rotaround = 52
            rot = trimesh.transformations.rotation_matrix(
                  np.radians(rotaround), [0, 1, 0])
            mesh.apply_transform(rot)

        # set feet to zero
        vertices = np.array(mesh.vertices)
        vertices[:, 1] = vertices[:, 1] - vertices[:, 1].min()
        mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces,
                               process=False,
                               vertex_colors=vertex_colors,
                               )
        if vertex_colors is not None:
            mesh.visual.vertex_colors = vertex_colors

        # create pyrender mesh
        pyr_mesh = pyrender.Mesh.from_trimesh(
            mesh,
            #  material=self.material if vertex_colors is None else None,
            material=self.material if vertex_colors is None else 
            self.vertex_material,
            smooth=True)

        # remove old mesh if still in scene
        try:
            self.scene.remove_node(self.nm)
        except:
            pass
        self.nm = pyrender.Node(mesh=pyr_mesh, matrix=np.eye(4))
        self.scene.add_node(self.nm)

        # render and save
        color, _ = self.r.render(self.scene, flags=pyrender.RenderFlags.RGBA)
        output_img = pil_img.fromarray((color).astype(np.uint8))
        w, h = output_img.size
        output_img.crop((200,0,w-200,h)) # \
            #.save(osp.join(self.output_folder, m.replace('ply', 'png')))

        return output_img
