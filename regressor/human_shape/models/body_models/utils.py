from typing import List, Optional
import os.path as osp
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from human_shape.utils import Tensor, StringList


def transform_mat(R: Tensor, t: Tensor) -> Tensor:
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1], value=0.0),
                      F.pad(t, [0, 0, 0, 1], value=1.0)], dim=2)


def find_joint_kin_chain(joint_id: int, kinematic_tree: List) -> List:
    kin_chain = []
    curr_idx = joint_id
    while curr_idx != -1:
        kin_chain.append(curr_idx)
        curr_idx = kinematic_tree[curr_idx]
    return kin_chain


def to_tensor(array, dtype=torch.float32) -> Tensor:
    if not torch.is_tensor(array):
        return torch.tensor(array, dtype=dtype)
    else:
        return array.to(dtype=dtype)


class JointsFromVerticesSelector(nn.Module):

    def __init__(
        self,
        face_ids: Optional[List] = None,
        bcs: Optional[List] = None,
        names: Optional[StringList] = None,
        fname: str = None,
        **kwargs
    ) -> None:
        ''' Selects extra joints from vertices
        '''
        super(JointsFromVerticesSelector, self).__init__()

        err_msg = (
            'Either pass a filename or triangle face ids, names and'
            ' barycentrics')
        assert fname is not None or (
            face_ids is not None and bcs is not None and names is not None
        ), err_msg
        if fname is not None:
            fname = osp.expanduser(osp.expandvars(fname))
            with open(fname, 'r') as f:
                data = yaml.load(f)
            names = list(data.keys())
            bcs = []
            face_ids = []
            for name, d in data.items():
                face_ids.append(d['face'])
                bcs.append(d['bc'])
            bcs = np.array(bcs, dtype=np.float32)
            face_ids = np.array(face_ids, dtype=np.int32)
        assert len(bcs) == len(face_ids), (
            'The number of barycentric coordinates must be equal to the faces'
        )
        assert len(names) == len(face_ids), (
            'The number of names must be equal to the number of '
        )

        self.names = names
        self.register_buffer('bcs', torch.tensor(bcs, dtype=torch.float32))
        self.register_buffer(
            'face_ids', torch.tensor(face_ids, dtype=torch.long))

    def as_tensor(
        self,
        num_vertices: int,
        faces: Tensor
    ) -> Tensor:
        ''' Builds a linear regression matrix for the extra joints
        '''
        # Get the number of extra joints
        num_extra_joints = len(self.names)
        output = torch.zeros([num_extra_joints, num_vertices])
        # Get the indices of the vertices we use
        vertex_ids = faces[self.face_ids]
        for ii, vids in enumerate(vertex_ids):
            # Assign the barycentric weight of each point
            output[ii, vids] = self.bcs[ii]
        return output

    def extra_joint_names(self) -> StringList:
        ''' Returns the names of the extra joints
        '''
        return self.names

    def forward(
        self,
        vertices: Tensor,
        faces: Tensor
    ) -> Tensor:
        if len(self.face_ids) < 1:
            return []
        vertex_ids = faces[self.face_ids].reshape(-1)
        # Should be BxNx3x3
        triangles = torch.index_select(vertices, 1, vertex_ids).reshape(
            -1, len(self.bcs), 3, 3)
        return (triangles * self.bcs[None, :, :, None]).sum(dim=2)


class KeypointTensor(object):
    def __init__(self, data,
                 source='smplx',
                 keypoint_names=None,
                 connections=None,
                 part_connections=None,
                 part_indices=None,
                 **kwargs):
        ''' A keypoint wrapper with keypoint_names
        '''
        if isinstance(data, (KeypointTensor,)):
            data = data._t
        self._t = torch.as_tensor(data, **kwargs)
        self._source = source
        self._keypoint_names = keypoint_names
        self._connections = connections
        self._part_indices = part_indices
        self._part_connections = part_connections

    @staticmethod
    def from_obj(tensor, obj):
        return KeypointTensor(tensor, source=obj.source,
                              keypoint_names=obj.keypoint_names,
                              connections=obj.connections,
                              part_indices=obj.part_indices,
                              part_connections=obj.part_connections)

    @property
    def source(self):
        return self._source

    @property
    def keypoint_names(self):
        return self._keypoint_names

    @property
    def connections(self):
        return self._connections

    @property
    def part_indices(self):
        return self._part_indices

    @property
    def part_connections(self):
        return self._part_connections

    def __repr__(self) -> str:
        return f'KeypointTensor:\n{self._t}'

    #  def __add__(self, x):
        #  return KeypointTensor(
            #  self._t + x, keypoint_names=self._keypoint_names,
            #  source=self.source,
            #  connections=self._connections,
            #  part_indices=self._part_indices,
            #  part_connections=self._part_connections,
        #  )

    #  def __radd__(self, x):
        #  return KeypointTensor(
            #  self._t + x, keypoint_names=self._keypoint_names,
            #  source=self.source,
            #  connections=self._connections,
            #  part_indices=self._part_indices,
            #  part_connections=self._part_connections,
        #  )

    #  def __sub__(self, x):
        #  return KeypointTensor(
            #  self._t - x, keypoint_names=self._keypoint_names,
            #  source=self.source,
            #  connections=self._connections,
            #  part_indices=self._part_indices,
            #  part_connections=self._part_connections,
        #  )

    #  def __rsub__(self, x):
        #  return KeypointTensor(
            #  x - self._t, keypoint_names=self._keypoint_names,
            #  source=self.source,
            #  connections=self._connections,
            #  part_indices=self._part_indices,
            #  part_connections=self._part_connections,
        #  )

    #  def __mul__(self, x):
        #  return KeypointTensor(
            #  self._t * x, keypoint_names=self._keypoint_names,
            #  source=self.source,
            #  connections=self._connections,
            #  part_indices=self._part_indices,
            #  part_connections=self._part_connections,
        #  )

    #  def __rmul__(self, x):
        #  return KeypointTensor(
            #  self._t * x, keypoint_names=self._keypoint_names,
            #  source=self.source,
            #  connections=self._connections,
            #  part_indices=self._part_indices,
            #  part_connections=self._part_connections,
        #  )

    #  def __matmul__(self, x):
        #  return KeypointTensor(
            #  self._t @ x, keypoint_names=self._keypoint_names,
            #  source=self.source,
            #  connections=self._connections,
            #  part_indices=self._part_indices,
            #  part_connections=self._part_connections,
        #  )

    #  def __rmatmul__(self, x):
        #  return KeypointTensor(
            #  x @ self._t, keypoint_names=self._keypoint_names,
            #  source=self.source,
            #  connections=self._connections,
            #  part_indices=self._part_indices,
            #  part_connections=self._part_connections,
        #  )

    #  def __truediv__(self, x):
        #  return KeypointTensor(
            #  self._t / x, keypoint_names=self._keypoint_names,
            #  source=self.source,
            #  connections=self._connections,
            #  part_indices=self._part_indices,
            #  part_connections=self._part_connections,
        #  )

    #  def __rtruediv__(self, x):
        #  return KeypointTensor(
            #  x / self._t, keypoint_names=self._keypoint_names,
            #  source=self.source,
            #  connections=self._connections,
            #  part_indices=self._part_indices,
            #  part_connections=self._part_connections,
        #  )

    def __getitem__(self, key):
        return self._t[key]
        return KeypointTensor(
            self._t[key], keypoint_names=self._keypoint_names,
            source=self.source,
            connections=self.connections,
            part_indices=self._part_indices,
            part_connections=self._part_connections,
        )

    def __getattribute__(self, name):
        tensor = super(KeypointTensor, self).__getattribute__('_t')
        if hasattr(tensor, name):
            # If the tensor has a member function with name `name` then call it
            func = getattr(tensor, name)
            if 'numpy' in name:
                return lambda: self._t.numpy()
            elif callable(func):
                return lambda *args, **kwargs: KeypointTensor(
                    func(*args, **kwargs),
                    source=self.source,
                    keypoint_names=self._keypoint_names,
                    connections=self._connections,
                    part_indices=self._part_indices,
                    part_connections=self._part_connections,
                )
            else:
                return getattr(self._t, name)
        else:
            output = super(KeypointTensor, self).__getattribute__(name)
            return output

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        args = [a._t if hasattr(a, '_t') else a for a in args]
        ret = func(*args, **kwargs)
        if torch.is_tensor(ret):
            return KeypointTensor(ret,
                                  source=self.source,
                                  keypoint_names=self._keypoint_names,
                                  connections=self._connections,
                                  part_indices=self._part_indices,
                                  part_connections=self._part_connections,
                                  )
        else:
            return ret
