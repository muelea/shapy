from typing import Tuple, List, Optional, Dict
import json

import numpy as np
from loguru import logger

from human_shape.utils import Array, StringList, nand, binarize


def threshold_and_keep_parts(
    keypoints: Array,
    body_idxs: Array,
    left_hand_idxs: Array,
    right_hand_idxs: Array,
    face_idxs: Array,
    body_thresh: float = -1,
    hand_thresh: float = -1,
    face_thresh: float = -1,
    hand_only: bool = False,
    head_only: bool = False,
    is_right: bool = True,
    binarization: bool = True
):
    ''' Processes keypoints by thresholding confidence and keeping correct subset
    '''
    assert nand(head_only, hand_only), (
        'Hand only and head only can\'t be True at the same time')
    body_conf = keypoints[body_idxs, -1]
    # Only keep the body keypoints with confidence above a threshold
    if body_thresh > 0:
        body_conf[body_conf < body_thresh] = 0.0

    if head_only or hand_only:
        body_conf[:] = 0.0

    left_hand_conf = keypoints[left_hand_idxs, -1]
    right_hand_conf = keypoints[right_hand_idxs, -1]
    if hand_thresh > 0:
        left_hand_conf[left_hand_conf < hand_thresh] = 0.0
        right_hand_conf[right_hand_conf < hand_thresh] = 0.0

    face_conf = keypoints[face_idxs, -1]
    if face_thresh > 0:
        face_conf[face_conf < face_thresh] = 0.0

    if head_only:
        left_hand_conf[:] = 0.0
        right_hand_conf[:] = 0.0

    if hand_only:
        face_conf[:] = 0.0
        if is_right:
            left_hand_conf[:] = 0
        else:
            right_hand_conf[:] = 0

    if binarization:
        body_conf = binarize(
            body_conf, body_thresh, keypoints.dtype)
        left_hand_conf = binarize(
            left_hand_conf, hand_thresh, keypoints.dtype)
        right_hand_conf = binarize(
            right_hand_conf, hand_thresh, keypoints.dtype)
        face_conf = binarize(
            face_conf, face_thresh, keypoints.dtype)

    keypoints[body_idxs, -1] = body_conf
    keypoints[left_hand_idxs, -1] = left_hand_conf
    keypoints[right_hand_idxs, -1] = right_hand_conf
    keypoints[face_idxs, -1] = face_conf

    return keypoints


def read_keypoints(keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=True):
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    all_keypoints = []
    for idx, person_data in enumerate(data['people']):
        body_keypoints = np.array(person_data['pose_keypoints_2d'],
                                  dtype=np.float32)
        body_keypoints = body_keypoints.reshape([-1, 3])

        left_hand_keyps = person_data.get('hand_left_keypoints_2d', [])
        if len(left_hand_keyps) < 1:
            left_hand_keyps = [0] * (21 * 3)
        left_hand_keyps = np.array(
            left_hand_keyps, dtype=np.float32).reshape([-1, 3])

        right_hand_keyps = person_data.get('hand_right_keypoints_2d', [])
        if len(right_hand_keyps) < 1:
            right_hand_keyps = [0] * (21 * 3)
        right_hand_keyps = np.array(
            right_hand_keyps, dtype=np.float32).reshape([-1, 3])

        face_keypoints = person_data.get('face_keypoints_2d', [])
        if len(face_keypoints) < 1:
            face_keypoints = [0] * (70 * 3)

        face_keypoints = np.array(
            face_keypoints,
            dtype=np.float32).reshape([-1, 3])

        face_keypoints = face_keypoints[:-2]

        all_keypoints.append(
            np.concatenate([
                body_keypoints,
                left_hand_keyps, right_hand_keyps,
                face_keypoints], axis=0)
        )

    if len(all_keypoints) < 1:
        return None

    all_keypoints = np.stack(all_keypoints)

    return all_keypoints


def map_keypoints(
    source_dataset: str,
    target_dataset: str,
    names_dict: Dict[str, StringList],
    source_names: Optional[StringList] = None,
    target_names: Optional[StringList] = None,
):
    assert source_dataset in names_dict, (
        f'Source dataset not in names dictionary: {source_dataset}')
    assert target_dataset in names_dict, (
        f'Target dataset not in names dictionary: {target_dataset}')

    mapping = {}
    if source_names is None:
        source_names = names_dict.get(source_dataset)
    if target_names is None:
        target_names = names_dict.get(target_dataset)

    for idx, name in enumerate(target_names):
        if name in source_names:
            mapping[idx] = source_names.index(name)

    #  indices_in_target = list(mapping.keys())
    #  indices_in_source = list(mapping.values())
    indices_in_target = np.array(list(mapping.keys()), dtype=np.long)
    indices_in_source = np.array(list(mapping.values()), dtype=np.long)
    #  logger.info(f'Source, target: {source_dataset} -> {target_dataset}')
    #  for sidx, tidx in zip(indices_in_source, indices_in_target):
    #  logger.info(
    #  f'{sidx}, {source_names[sidx]} -> {tidx}, {target_names[tidx]}')

    return indices_in_target, indices_in_source, len(target_names)


def kp_connections(
    keypoints: StringList,
    kp_lines: List[Tuple[str, str]],
    part: str = None,
    keypoint_parts: Dict[str, str] = None,
) -> List[Tuple[int, int]]:
    ''' Given keypoint names and edges builds a connection index array
    '''
    side = ''
    if part is not None:
        if 'left_hand' in part:
            side, part = 'left', 'hand'
        if 'right_hand' in part:
            side, part = 'right', 'hand'

    connections = []
    for connection in kp_lines:
        if connection[0] in keypoints and connection[1] in keypoints:
            if part is None:
                connections.append([keypoints.index(connection[0]),
                                    keypoints.index(connection[1])])
            else:
                if keypoint_parts is not None:
                    start_part = keypoint_parts.get(connection[0], '')
                    end_part = keypoint_parts.get(connection[1], '')
                    if part in start_part and part in end_part:
                        if side:
                            if side in connection[0] and side in connection[1]:
                                connections.append(
                                    [keypoints.index(connection[0]),
                                     keypoints.index(connection[1])])
                        else:
                            connections.append(
                                [keypoints.index(connection[0]),
                                 keypoints.index(connection[1])])

    return connections


def build_flip_map(keypoint_names, name='openpose'):
    flip_map = {}
    if name == 'mpii':
        for keyp_name in keypoint_names:
            flip_map[keyp_name] = keyp_name
    else:
        for keyp_name in keypoint_names:
            if 'left' in keyp_name:
                flip_map[keyp_name] = keyp_name.replace('left', 'right')
            elif 'right' in keyp_name:
                flip_map[keyp_name] = keyp_name.replace('right', 'left')
    return flip_map


def create_flip_indices(names):
    flip_map = {}
    for keyp_name in names:
        if 'left' in keyp_name:
            flip_map[keyp_name] = keyp_name.replace('left', 'right')
        elif 'right' in keyp_name:
            flip_map[keyp_name] = keyp_name.replace('right', 'left')
    full_flip_map = flip_map.copy()
    full_flip_map.update({v: k for k, v in flip_map.items()})
    flipped_names = [i if i not in full_flip_map else full_flip_map[i]
                     for i in names]
    flip_indices = [names.index(i) for i in flipped_names if i in names]
    return flip_indices


def get_part_idxs(
        keypoint_names: List[str],
        keypoint_parts: Dict[str, str]
) -> Dict[str, Array]:
    body_idxs = np.asarray([
        idx
        for idx, name in enumerate(keypoint_names)
        if name in keypoint_parts
        if 'body' in keypoint_parts[name]])
    hand_idxs = np.asarray([
        idx
        for idx, name in enumerate(keypoint_names)
        if name in keypoint_parts
        if 'hand' in keypoint_parts[name]])

    left_hand_idxs = np.asarray([
        idx
        for idx, name in enumerate(keypoint_names)
        if name in keypoint_parts
        if 'hand' in keypoint_parts[name] and 'left' in name])

    right_hand_idxs = np.asarray([
        idx
        for idx, name in enumerate(keypoint_names)
        if name in keypoint_parts
        if 'hand' in keypoint_parts[name] and 'right' in name])

    face_idxs = np.asarray([
        idx
        for idx, name in enumerate(keypoint_names)
        if name in keypoint_parts
        if 'face' in keypoint_parts[name]])
    head_idxs = np.asarray([
        idx
        for idx, name in enumerate(keypoint_names)
        if name in keypoint_parts
        if 'head' in keypoint_parts[name]])
    flame_idxs = np.asarray([
        idx
        for idx, name in enumerate(keypoint_names)
        if name in keypoint_parts
        if 'flame' in keypoint_parts[name]])
    torso_idxs = np.asarray([
        idx
        for idx, val in enumerate(keypoint_parts.values())
        if 'torso' in val])
    upper_body_idxs = np.asarray([
        idx
        for idx, val in enumerate(keypoint_parts.values())
        if 'upper' in val])
    return {
        'body': body_idxs.astype(np.int64),
        'hand': hand_idxs.astype(np.int64),
        'face': face_idxs.astype(np.int64),
        'head': head_idxs.astype(np.int64),
        'left_hand': left_hand_idxs.astype(np.int64),
        'right_hand': right_hand_idxs.astype(np.int64),
        'flame': flame_idxs.astype(np.int64),
        'torso': torso_idxs.astype(np.int64),
        'upper': upper_body_idxs.astype(np.int64),
    }
