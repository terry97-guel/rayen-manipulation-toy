import numpy as np
import torch
import mujoco

def quaternion_to_matrix(quaternion):
    w, x, y, z = quaternion
    matrix = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ])
    return matrix


def skewed_symetric_matrix(axis):
    axis_skew = torch.FloatTensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return axis_skew

def rodrigues_rotation_matrix(axis_skew, joint_qpos):
    device = joint_qpos.device
    identity = torch.eye(3).to(device)
    rotation_matrix = identity + torch.sin(joint_qpos) * axis_skew + (1 - torch.cos(joint_qpos)) * axis_skew@axis_skew

    return rotation_matrix

def rodrigues_rotation_matrix_batch(axis_skew, joint_qpos):
    device = joint_qpos.device
    batch_size = joint_qpos.shape[0]
    identity = torch.eye(3).repeat(batch_size,1,1).to(device)
    rotation_matrix = identity + torch.sin(joint_qpos).view(batch_size,1,1) * axis_skew + (1 - torch.cos(joint_qpos)).view(batch_size,1,1) * (axis_skew@axis_skew).repeat(batch_size,1,1)
    return rotation_matrix

def reset(mj_model, mj_data, viewer):
    mj_data.qpos[:] = 0
    mujoco.mj_forward(mj_model, mj_data)
    viewer.render()

def play_continuous(viewer, mj_model, mj_data):
    mujoco.mj_forward(mj_model, mj_data)
    viewer._paused = True
    viewer.render()

def plot_sphere(viewer,p,r,rgba=[1,1,1,1],label=''):
    """
        Add sphere
    """
    viewer.add_marker(
        pos   = p,
        size  = [r,r,r],
        rgba  = rgba,
        type  = mujoco.mjtGeom.mjGEOM_SPHERE,
        label = label)
    
def print_body_in_contact(mj_model, mj_data):
    geom1 = mj_data.contact[0].geom1
    geom2 = mj_data.contact[0].geom2

    geom1_data = mj_model.geom(geom1)
    geom2_data = mj_model.geom(geom2)

    body1_name = mj_model.body(geom1_data.bodyid).name
    body2_name = mj_model.body(geom2_data.bodyid).name

    print(f"Body 1: {body1_name}, Body 2: {body2_name}")