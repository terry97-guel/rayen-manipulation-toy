# %%
from toolbox.resources import get_asset_dict, ASSET_SRC
from mujoco_viewer import MujocoViewer

mangerie_robot_dict = get_asset_dict(ASSET_SRC.CUSTOM, True)
robot_path = mangerie_robot_dict['scene_ur5e_rg2']
device = 'cuda:0'
n_steps = 30

APPLY_CONSTRAINT = False

# %%
import mujoco
from util import quaternion_to_matrix,\
    skewed_symetric_matrix,\
    rodrigues_rotation_matrix, \
    rodrigues_rotation_matrix_batch,\
    reset, \
    play_continuous, \
    plot_sphere

mj_model = mujoco.MjModel.from_xml_path(robot_path.as_posix())
mj_data = mujoco.MjData(mj_model)

try:
    viewer.close()
except:
    pass
viewer = MujocoViewer(
   mj_model,mj_data,mode='window',title="RAYEN",
   width=1200,height=800,hide_menus=True
   )

reset(mj_model, mj_data, viewer)
# play_continuous(viewer,mj_model, mj_data)

# %%

pos = mj_data.site("gripper_center").xpos

plot_sphere(viewer, pos, 0.02, rgba=[1,0,0,1],label="gripper_center")
# play_continuous(viewer, mj_model, mj_data)

# %%
from sample import sample_one_traj, sample_n_traj
# test traj_sampling
for _ in range(10):
    traj, qpos_array = sample_one_traj(mj_model, mj_data, n_steps=n_steps, length_scale=1.0, qpos_scale=0.2, viewer=viewer, PLOT=True)

# %%
# get offset info
import torch
T = torch.eye(4).to(device)

ee_site_name = "gripper_center"
site_id = mj_model.site(ee_site_name).id
site_p_offset = mj_model.site(ee_site_name).pos
site_q_offset = mj_model.site(ee_site_name).quat

site_T_offset = torch.eye(4).to(device)
site_T_offset[:3,:3] = torch.FloatTensor(quaternion_to_matrix(site_q_offset)).to(device)
site_T_offset[:3,3]  = torch.FloatTensor(site_p_offset).to(device)

site_bodyid   = mj_model.site(ee_site_name).bodyid
bodyid = site_bodyid[0]

T_offset_ls = []
axis_skew_ls  = []
for body_i in range(2, bodyid+1):
    quat_offset = mj_model.body(body_i).quat.copy()
    p_offset  = torch.FloatTensor(mj_model.body(body_i).pos.copy())
    R_offset  = torch.FloatTensor(quaternion_to_matrix(quat_offset))
    T_offset = torch.eye(4).to(device)
    T_offset[:3,:3] = R_offset
    T_offset[:3,3]  = p_offset
    
    T_offset_ls.append(T_offset)

    joint_i = mj_model.body(body_i).jntadr
    T_transform = torch.eye(4).to(device)
    if joint_i == -1:
        axis_skew_ls.append(None)
    else:
        joint_qpos = torch.FloatTensor(mj_data.joint(joint_i).qpos).to(device)
        axis = mj_model.joint(joint_i).axis
        axis_skew = skewed_symetric_matrix(axis).to(device)
        axis_skew_ls.append(axis_skew)

        R_transform = rodrigues_rotation_matrix(axis_skew, joint_qpos).to(device)
        T_transform[:3,:3] = R_transform

    T = T @ T_offset @ T_transform
T = T @ site_T_offset

# %%
def fk_torch(qpos):
    T = torch.eye(4).to(device)
    
    for axis_skew, T_offset,q in zip(axis_skew_ls, T_offset_ls, qpos):
        T_transform = torch.eye(4).to(device)
        if axis_skew is not None:
            R_transform = rodrigues_rotation_matrix(axis_skew, q)
            T_transform[:3,:3] = R_transform
        T = T @ T_offset @ T_transform
    T = T @ site_T_offset
    return T


qpos = torch.FloatTensor(mj_data.qpos[:6]).to(device)
T = fk_torch(qpos)
pred_pos = T[:3,3].cpu().detach().numpy()
gt_pos = mj_data.site(site_id).xpos
assert (pred_pos - gt_pos < 1e-5).all()


# %%
def fk_torch_batch(qpos_batch):
    batch_size = qpos_batch.shape[0]
    n_steps = qpos_batch.shape[1]
    T = torch.eye(4).repeat(batch_size * n_steps,1,1).to(device)

    for i in range(6):
        axis_skew = axis_skew_ls[i]
        T_offset = T_offset_ls[i]
        q = qpos_batch[:,:,i]
        T_transform = torch.eye(4).repeat(batch_size * n_steps,1,1).to(device)

        if axis_skew is not None:
            R_transform = rodrigues_rotation_matrix_batch(axis_skew, q.flatten())
            T_transform[:,:3,:3] = R_transform
        T = T @ T_offset.repeat(batch_size * n_steps,1,1) @ T_transform

    T = T @ site_T_offset.repeat(batch_size * n_steps,1,1)
    return T.reshape(batch_size, n_steps, 4, 4)

batch_size = 16
qpos_batch = mj_data.qpos[:6]
qpos_batch = torch.FloatTensor(qpos_batch).repeat(batch_size,n_steps,1).to(device)
T = fk_torch_batch(qpos_batch)

pred_pos = T[:,:,:3,3].cpu().detach().numpy()
gt_pos = mj_data.site(site_id).xpos

assert (pred_pos - gt_pos < 1e-5).all()
# assert (torch.tensor(mj_data.site(site_id).xpos) - T[0,0,:3,3] < 1e-5).all()
# %%
batch_size = 16
trajs, qpos_arrays = sample_n_traj(mj_model, mj_data, n_steps=n_steps, length_scale=1.0, qpos_scale=0.2, n_traj=batch_size)
trajs, qpos_arrays = trajs.to(device), qpos_arrays.to(device)

# %%
import torch
input_shape  = n_steps * 3 
output_shape = n_steps * 6

class Torch_Model_HC(torch.nn.Module):
    def __init__(self, b_scale=0.1):
        super(Torch_Model_HC, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_shape, 512 *3),
            torch.nn.Mish(),
            torch.nn.Linear(512*3, 512*3),
            torch.nn.Mish(),
            torch.nn.Linear(512*3, output_shape)
        )

        dt = 0.002
        A = torch.zeros((6*n_steps, 6*n_steps)).to(device)

        for i in range(6*n_steps):
            A[i, 6*i+0:6*i+6 ] = -1
            A[i, 6*i+6:6*i+12] =  1

        A = A / dt 
        B = -A

        self.A = torch.concatenate([A,B], axis=0)
        self.b = torch.ones((n_steps * 12, 1)).to(device) * b_scale

        self.A_batch = None
        self.b_batch = None

    def forward(self, trajs, q_init):
        if APPLY_CONSTRAINT:
            return self.constraint_forward(trajs, q_init)
        else:
            return self.naive_forward(trajs, q_init)

    def constraint_forward(self, trajs, q_init):
        batch_size = trajs.shape[0]
        n_steps = trajs.shape[1]//3

        q_init = q_init.repeat(1, n_steps)
        z0 = q_init

        v = self.fc(trajs)
        v_len = torch.norm(v, dim=1, keepdim=True)
        v_dir = v/v_len

        if self.A_batch is None:
            self.A_batch = self.A.repeat(batch_size, 1, 1)
            self.b_batch = self.b.repeat(batch_size, 1, 1)

        v_dir = v_dir.reshape(batch_size, 6*n_steps, 1)
        z0 = z0.reshape(batch_size, 6*n_steps, 1)

        eps = 1e-5

        k = self.A_batch @ v_dir/(self.b_batch - self.A_batch @ z0)
        k = torch.maximum(k, torch.zeros_like(k)) + eps
        k, indices = torch.max(k.reshape(batch_size,-1),dim=1)

        v_len = torch.minimum(v_len.flatten(), (1/k))
        v = v_dir * v_len.reshape(batch_size, 1, 1)

        return v + z0

    def naive_forward(self, trajs, q_init):
        batch_size = trajs.shape[0]
        n_steps = trajs.shape[1]//3

        q_init = q_init.repeat(1, n_steps)

        qpos = self.fc(trajs)
        return qpos + q_init

torch_model = Torch_Model_HC(b_scale = 100)
torch_model = torch_model.to(device)

q_init = qpos_arrays[:, 0, :]
trajs = trajs.reshape(batch_size, n_steps*3)
qpos_batch = torch_model(trajs, q_init)

# %%
# optim
import torch.optim as optim
optimizer = optim.Adam(torch_model.parameters(), lr=0.001)
loss_ls = []

# %%

def evaluate():
    # evaluate and visualize
    batch_size = 1
    trajs, qpos_arrays = sample_n_traj(mj_model, mj_data, n_steps=n_steps, length_scale=1.0, qpos_scale=0.2, n_traj=batch_size)
    trajs, qpos_arrays = trajs.to(device), qpos_arrays.to(device)

    trajs = trajs.reshape(batch_size, 3*n_steps)
    q_init = qpos_arrays[:, 0, :]
    qpos = torch_model(trajs, q_init).reshape(batch_size * n_steps, 6)

    trajs = trajs.reshape(batch_size * n_steps, 3)

    for i in range(n_steps):
        q = qpos[i]
        
        mj_data.qpos[:6] = q.cpu().detach().numpy()
        mujoco.mj_forward(mj_model, mj_data)

        ee_pos = trajs[i].cpu().detach().numpy()
        for _ in range(10):
            plot_sphere(viewer, mj_data.site("gripper_center").xpos, 0.02, rgba=[1,0,0,1],label="gripper_center")
            plot_sphere(viewer, ee_pos, 0.02, rgba=[0,1,0,1],label="ee_pos")
        
            viewer.render()

# %%
# train
batch_size = 16
max_iter = 10000

# for iter_i in range(max_iter):
while len(loss_ls)<max_iter:
    trajs, qpos_arrays = sample_n_traj(mj_model, mj_data, n_steps=n_steps, length_scale=1.0, qpos_scale=0.2, n_traj=batch_size)
    trajs, qpos_arrays = trajs.to(device), qpos_arrays.to(device)

    q_init = qpos_arrays[:, 0, :]
    trajs = trajs.reshape(batch_size, 3*n_steps)
    qpos = torch_model(trajs, q_init)
    qpos = qpos.reshape(batch_size, n_steps, 6)
    
    T = fk_torch_batch(qpos)
    loss = torch.nn.L1Loss()(T[:,:,:3,3].reshape(batch_size, 3*n_steps), trajs)

    optimizer.zero_grad()
    loss.backward()
    loss_ls.append(loss)
    optimizer.step()

    iter_i = len(loss_ls)
    print(f"iter {iter_i} loss {loss.item()}")

    if iter_i % 500 == 0:
        evaluate()

# %%
for _ in range(10):
    evaluate()

# %%
# torch save
if APPLY_CONSTRAINT:
    torch.save(torch_model.state_dict(), 'torch_model_constraint.pt')
else:
    torch.save(torch_model.state_dict(), 'torch_model_naive.pt')