import numpy as np
import mujoco
from util import print_body_in_contact, plot_sphere

# def sample_traj(mj_model, mj_data, n_steps=100, VERBOSE=False, viewer=None):
#     # random sample qpos, dim:6
    
#     qpos_start = np.random.uniform(-1, 1, 6)/2

#     while True:
#         mj_data.qpos[:6] = qpos_start
#         mujoco.mj_forward(mj_model, mj_data)
#         ee_pos = mj_data.site("gripper_center").xpos
#         if mj_data.ncon ==0:
#             break
#         else:
#             if VERBOSE: print_body_in_contact(mj_model, mj_data)
#             qpos_start = np.random.uniform(-1, 1, 6)/2

#     qpos_end = np.random.uniform(-1, 1, 6)/2
#     while True:
#         mj_data.qpos[:6] = qpos_end
#         mujoco.mj_forward(mj_model, mj_data)
#         ee_pos = mj_data.site("gripper_center").xpos
#         if mj_data.ncon ==0:
#             break
#         else:
#             if VERBOSE: print_body_in_contact(mj_model, mj_data)
#             qpos_end = np.random.uniform(-1, 1, 6)/2

#     qpos_array = np.linspace(qpos_start, qpos_end, n_steps)


def get_traj(mj_model, mj_data, qpos_array, viewer=None):
    traj_list = []
    for qpos in qpos_array:
        mj_data.qpos[:6] = qpos
        mujoco.mj_forward(mj_model, mj_data)
        
        ee_pos = mj_data.site("gripper_center").xpos.copy()
        traj_list.append(ee_pos)
        if viewer is not None:
            plot_sphere(viewer, ee_pos, 0.02, rgba=[1,0,0,1],label="gripper_center")
            viewer.render()
        
        if mj_data.ncon !=0:
            return None
    
    return np.array(traj_list)
    
# def sample_one_traj(mj_model, mj_data, n_steps=100, VERBOSE=False, viewer=None):
#     while True:
#         traj, qpos_array = sample_traj(mj_model, mj_data, n_steps=n_steps, VERBOSE=VERBOSE, viewer=viewer)
#         if traj is not None:
#             return traj, qpos_array




import numpy as np
import matplotlib.pyplot as plt
import torch 

# Define the kernel
def kernel(x1, x2, length_scale=1.0):
    return np.exp(-0.5 * ((x1 - x2) / length_scale)**2)

# Generate random function samples from the GP prior
def sample_gp_prior(num_samples, input_points, length_scale=1.0):
    num_points = len(input_points)
    cov_matrix = np.zeros((num_points, num_points))

    # Populate the covariance matrix
    for i in range(num_points):
        for j in range(num_points):
            cov_matrix[i, j] = kernel(input_points[i], input_points[j], length_scale)

    # Ensure the matrix is symmetric
    cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)

    # Sample from the multivariate normal distribution
    mean = np.zeros(num_points)
    samples = np.random.multivariate_normal(mean, cov_matrix, size=num_samples)

    return samples


def sample_grp_traj(mj_model, mj_data, input_points,num_samples, length_scale=1.0, viewer=None,qpos_scale=0.1):
    # Input points

    # Sample from the GP prior
    qpos_sample = qpos_scale * sample_gp_prior(num_samples, input_points, length_scale).T


    traj = get_traj(mj_model, mj_data, qpos_sample, viewer=viewer)

    return traj, qpos_sample

def sample_one_traj(mj_model, mj_data, n_steps=100, length_scale=1.0, viewer=None, PLOT=False, qpos_scale=0.1):
    input_points = np.linspace(0, 1, n_steps)

    # Number of samples from the GP prior
    num_samples = 6

    while True:
        traj, qpos_sample = sample_grp_traj(mj_model,
                                             mj_data,
                                               input_points=input_points,
                                               num_samples=num_samples,
                                               length_scale=length_scale,
                                                 viewer=None,
                                                 qpos_scale=qpos_scale)
        if traj is not None:
            if PLOT:
                # Plot the samples
                plt.figure(figsize=(8, 6))
                for i in range(num_samples):
                    plt.plot(input_points, qpos_sample[:, i], label=f'Sample {i + 1}')

                plt.title('Samples from GP Prior')
                plt.xlabel('Input')
                plt.ylabel('Function Value')
                plt.legend()
                plt.show()
            if viewer is not None:
                for qpos in qpos_sample:
                    mj_data.qpos[:6] = qpos
                    mujoco.mj_forward(mj_model, mj_data)
                    
                    ee_pos = mj_data.site("gripper_center").xpos
                    if viewer is not None:
                        plot_sphere(viewer, ee_pos, 0.02, rgba=[1,0,0,1],label="gripper_center")
                        viewer.render()

            return torch.FloatTensor(traj), torch.FloatTensor(qpos_sample)

def sample_n_traj(mj_model, mj_data, n_traj=10, n_steps=100, length_scale=1.0, viewer=None, PLOT=False, qpos_scale=0.1):
    trajs = []
    qpos_samples = []
    for i in range(n_traj):
        traj, qpos_sample = sample_one_traj(mj_model, mj_data, n_steps=n_steps, length_scale=length_scale, viewer=viewer, PLOT=PLOT, qpos_scale=qpos_scale)
        trajs.append(traj)
        qpos_samples.append(qpos_sample)
    return torch.stack(trajs), torch.stack(qpos_samples)