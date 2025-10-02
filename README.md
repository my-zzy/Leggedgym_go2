# Training go2 with IsaacGym

## Installation

### Install Ubuntu20.04

+ Ubuntu https://mirrors4.tuna.tsinghua.edu.cn/ubuntu-releases/

+ Refus https://rufus.ie/zh/

+ use 'safe graphics' if not successful

+ Partition: https://blog.csdn.net/wyr1849089774/article/details/133387874?spm=1001.2014.3001.5506

+ Connect wifi & check install third-party software

+ Clash-verge 1.7.0 suits 20.04

+ Cuda: https://blog.csdn.net/weixin_37926734/article/details/123033286?spm=1001.2014.3001.5506

### Install Isaacgym

+ Download: https://developer.nvidia.com/isaac-gym

+ Doc: https://junxnone.github.io/isaacgymdocs/install.html

*Caution!!*

Do not use `./create_conda_env_rlgpu.sh`. It may cause version conflicts. Install pytorch independently. I use python3.7 which matches the version below.

```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

+ Tutor: https://blog.csdn.net/m0_37802038/article/details/134629194?spm=1001.2014.3001.5506

+ Install right python version 3.7(deadsnakes-ppa)

### Install rsl_rl & leggedgym

```
git clone https://github.com/leggedrobotics/rsl_rl
git checkout v1.0.2
pip install -e .
git clone https://github.com/leggedrobotics/legged_gym
pip install -e .
```

### Scripts

```
python train.py --task=THUrl
python train.py --task=THUrl --headless
tensorboard --logdir=logs
```

## Code structure

ac
算法框架包含了策略价值方法

algorithm
ppo算法

configs
四足机器人配置和环境配置

envs
四足机器人与训练环境设计

onpolicy
包含数据运算与日志输出

resources
包含四足机器人机械信息

sim2simorreal
包含部署到仿真环境和真实环境代码

test.py
用于测试训练完成的模型

train.py
用于训练模型

在configs文件夹下go2.py中，设计奖励函数、成本函数和域随机化参数，
在legged_go2.py中设计奖励函数、成本函数和域随机化函数。

奖励函数可以分为三类，任务奖励、增强辅助奖励和固定辅助奖励

任务奖励是直接与强化学习任务目标相关的奖励，激励机器人完成特定的任务。

增强辅助奖励是机器狗达成最终目标需要学会的阶段性子任务，根据经验给予阶段性奖励引导机器狗在前期更快地朝正确方向学习减少探索的错误行为。比如机器狗要学会保持平衡，就要对摔倒进行惩罚。

固定辅助奖励是一种固定的、与任务无关的奖励，用来限制机器人执行不期望的行为或引导其遵守一定的行为规范。比如对机器狗的能耗进行惩罚。

## Progress (differece from first version)

### legged_go2_config & go2
- add rewards scales
- add costs scales and `num_costs`
- add `domain_rand`
- missing imports
- modify `n_priv_latent`(unmatched degree)
- unmatched policy class name

### legged_go2
- add cost and reward functions
- complete `_process_rigid_shape_props` and `_process_rigid_body_props`
- fix `_compute_torques`
- fix `compute_cost`
- initialize `randomized_lag_tensor`

### ppo
- complete `PPO`
- fix unmatched interfaces

### actor_critic
- add `evaluate_actions`

### execute
- missing imports
- init Summarywriter

*For more details, check the git tree.*

## Differences from Standard legged_gym

This repository extends the standard [legged_gym](https://github.com/leggedrobotics/legged_gym) framework with several key enhancements:

### 1. **Constrained Reinforcement Learning**
- **Files**: `net/ppo.py`, `configs/go2.py`, `envs/legged_go2.py`
- **Functions**: `PPO.update()`, `compute_cost()`, `_cost_*()` functions
- **Cost Functions Implemented**:
  - Joint position/velocity limits: `_cost_joint_pos_limits()`, `_cost_joint_vel_limits()`
  - Motor torque limits: `_cost_torque_limits()`
  - Collision avoidance: `_cost_collision()`
  - Base orientation stability: `_cost_base_orientation()`
  - Contact force limits: `_cost_feet_contact_forces()`
  - Action smoothness: `_cost_action_smoothness()`

### 2. **Advanced Actor-Critic Architectures**
- **File**: `ac/actor_critic.py`
- **Classes**: 
  - `ActorCriticMixedBarlowTwins`: Mixed expert networks with Barlow Twins
  - `ActorCriticBarlowTwins`: Barlow Twins representation learning
  - `ActorCriticmlp`: Enhanced MLP with teacher-student architecture
- **Key Methods**: `evaluate_actions()`, `act_teacher()`, `BarlowTwinsLoss()`

### 3. **Enhanced Observation Space (733D)**
- **File**: `envs/legged_go2.py`
- **Function**: `compute_observations()`
- **Config**: `configs/go2.py` - observation dimensions
- **Components**:
  - Proprioceptive data (45D): base velocity, joint states, commands
  - Terrain height scans (187D): `_get_heights()` 
  - Historical state buffer (450D): `obs_history_buf`
  - Private privileged information (51D): motor params, friction, contact states

### 4. **Domain Randomization & Robustness** 
- **File**: `envs/legged_go2.py`, `configs/go2.py`
- **Functions**: `_process_rigid_body_props()`, `_process_dof_props()`, `_push_robots()`
- **Features**:
  - Motor strength variations: `motor_strength_range`
  - PD gain randomization: `kp_range`, `kd_range` 
  - Action lag simulation: `randomized_lag_tensor`, `lag_timesteps`
  - Mass/friction variations: `added_mass_range`, `friction_range`

### 5. **Go2 Robot Specialization**
- **Files**: `configs/go2.py`, `resources/go2/urdf/go2.urdf`
- **Config**: Go2-specific joint angles, PD gains, reward scales
- **Asset**: `file = '{ROOT_DIR}/resources/go2/urdf/go2.urdf'`
- **Deployment**: `sim2simorreal/` directory for real robot transfer

### 6. **Advanced Training Features**
- **Imitation Learning**: 
  - File: `ac/actor_critic.py`
  - Functions: `imitation_learning_loss()`, `BarlowTwinsLoss()`, `act_teacher()`
- **History Encoding**: 
  - File: `ac/common_modules.py`
  - Class: `StateHistoryEncoder`, function: `infer_hist_latent()`
- **Mixed Expert Networks**: 
  - File: `ac/common_modules.py` 
  - Class: `MixedMlp`
- **Privileged Training**: 
  - Function: `infer_priv_latent()` in actor-critic classes

### 7. **Infrastructure Enhancements**
- **Custom Storage**: 
  - File: `onpolicy/store.py`
  - Class: `RolloutStorageWithCost`
  - Methods: `compute_cost_returns()`, cost advantage computation
- **Enhanced Logging**: 
  - File: `onpolicy/execute.py`
  - Function: `log()` with TensorBoard integration
- **Training Pipeline**: 
  - File: `onpolicy/execute.py`
  - Class: `Onexecute` with constrained RL support


