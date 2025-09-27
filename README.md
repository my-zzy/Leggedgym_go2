# Training go2 with IsaacGym

## Install Ubuntu20.04

+ Ubuntu https://mirrors4.tuna.tsinghua.edu.cn/ubuntu-releases/

+ Refus https://rufus.ie/zh/

+ (safe graphics) if not successful

+ Partition: https://blog.csdn.net/wyr1849089774/article/details/133387874?spm=1001.2014.3001.5506

+ Connect wifi & check install third-party software

+ Clash-verge 1.7.0 suits 20.04

+ Cuda: https://blog.csdn.net/weixin_37926734/article/details/123033286?spm=1001.2014.3001.5506

## Install Isaacgym

+ Download: https://developer.nvidia.com/isaac-gym

+ Doc: https://junxnone.github.io/isaacgymdocs/install.html

**Important!!**

Do not use `./create_conda_env_rlgpu.sh`. It may cause version conflicts. Install pytorch independently.

```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

+ Tutor: https://blog.csdn.net/m0_37802038/article/details/134629194?spm=1001.2014.3001.5506

+ Install right python version 3.7(deadsnakes-ppa)

## Install rsl_rl & leggedgym

```
git clone https://github.com/leggedrobotics/rsl_rl
git checkout v1.0.2
pip install -e .
git clone https://github.com/leggedrobotics/legged_gym
pip install -e .
```
https://github.com/leggedrobotics/legged_gym


## Code structure

一、快速入门
cd THURL

conda activate THURL

python train.py --task --

python test.py --task --
二、详细介绍
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

三、优化设计
在configs文件夹下go2.py中，设计奖励函数、成本函数和域随机化参数，
在legged_go2.py中设计奖励函数、成本函数和域随机化函数。

奖励函数可以分为三类，任务奖励、增强辅助奖励和固定辅助奖励

任务奖励是直接与强化学习任务目标相关的奖励，激励机器人完成特定的任务。

增强辅助奖励是机器狗达成最终目标需要学会的阶段性子任务，根据经验给予阶段性奖励引导机器狗在前期更快地朝正确方向学习减少探索的错误行为。比如机器狗要学会保持平衡，就要对摔倒进行惩罚。

固定辅助奖励是一种固定的、与任务无关的奖励，用来限制机器人执行不期望的行为或引导其遵守一定的行为规范。比如对机器狗的能耗进行惩罚。

