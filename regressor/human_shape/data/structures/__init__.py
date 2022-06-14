from typing import NewType, List, Union, Tuple

from .abstract_structure import AbstractStructure
from .keypoints import Keypoints2D, Keypoints3D

from .betas import Betas
from .expression import Expression
from .global_rot import GlobalRot
from .body_pose import BodyPose
from .hand_pose import HandPose
from .jaw_pose import JawPose

from .vertices import Vertices
from .joints import Joints
from .bbox import BoundingBox

from .image_list import ImageList, ImageListPacked, to_image_list
from .points_2d import Points2D

StructureList = NewType('StructureList', List[AbstractStructure])
