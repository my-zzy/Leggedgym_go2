
from configs.legged_go2_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2Cfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_envs = 4096

        n_scan = 187
        n_priv_latent = 54
        n_proprio = 45
        history_len = 10
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        """
          unitree go2 sdk order:
               -0.1 <-3 FR_hip_joint 0 -> 0.0
               0.8 <- 4 FR_thigh_joint 1 -> 0.9
               -1.5 <- 5 FR_calf_joint 2 -> -1.8
               0.1 <- 0 FL_hip_joint 3 -> 0.0
               0.8 <- 1 FL_thigh_joint 4 -> 0.9
               -1.5 <- 2 FL_calf_joint 5 -> -1.8
               -0.1 <- 9 RR_hip_joint 6 -> 0.0
               1 <- 10 RR_thigh_joint 7 -> 0.9
               -1.5 <- 11 RR_calf_joint 8 -> -1.8
               0.1 <- 6 RL_hip_joint 9 -> 0.0
               1 <- 7 RL_thigh_joint 10 -> 0.9
               -1.5 <- 8 RL_calf_joint 11 -> -1.8
        """
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 30.}  # [N*m/rad]
        damping = {'joint': 0.75}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_scale_reduction = 1

        use_filter = True

    class commands( LeggedRobotCfg.control ):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        global_reference = False

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class asset( LeggedRobotCfg.asset ):
        file = '{ROOT_DIR}/resources/go2/urdf/go2.urdf'
        foot_name = "foot"
        name = "go2"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = True
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.30
        clearance_height_target = -0.2

        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001  # Changed from 0.0 to small penalty
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.
            feet_air_time = 1.0
            collision = -1.
            feet_stumble = -0.0
            action_rate = -0.01
            stand_still = -0.

        only_positive_rewards = True
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.9  # GO2 specific - tighter than base
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.30  # GO2 specific height
        max_contact_force = 100.


    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        
        # Motor and PD gain randomization
        randomize_kpkd = False  # Enable/disable PD gain randomization
        motor_strength_range = [0.9, 1.1]  # Motor strength variation range
        kp_range = [0.8, 1.2]  # Proportional gain variation range
        kd_range = [0.5, 1.5]  # Derivative gain variation range
        lag_timesteps = 6  # Action lag for realism

    
    class depth( LeggedRobotCfg.depth):
        use_camera = False
        camera_num_envs = 192
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.27, 0, 0.03]  # front camera
        angle = [-5, 5]  # positive pitch down

        update_interval = 1  # 5 works without retraining, 8 worse

        original = (106, 60)
        resized = (87, 58)
        horizontal_fov = 87
        buffer_len = 2
        
        near_clip = 0
        far_clip = 2
        dis_noise = 0.0
        
        scale = 1
        invert = True
    
    class costs:
        class scales:
            # Cost function scaling factors for constraint-based training
            joint_pos_limits = 1.0
            joint_vel_limits = 1.0
            torque_limits = 1.0
            collision = 10.0
            base_orientation = 1.0
            feet_contact_forces = 1.0
            action_smoothness = 0.1

        class d_values:
            # Threshold values for cost functions
            joint_pos_limit_threshold = 0.9
            joint_vel_limit_threshold = 0.9
            torque_limit_threshold = 0.9
            collision_threshold = 1.0
            base_orientation_threshold = 0.5
            feet_contact_force_threshold = 100.0
            action_rate_threshold = 2.0


    class cost:
        num_costs = 7
    
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        measure_heights = True
        include_act_obs_pair_buf = False

class Go2CfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 1.e-3
        max_grad_norm = 1
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        cost_value_loss_coef = 0.1
        cost_viol_loss_coef = 1

    class policy( LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        continue_from_last_std = True
        scan_encoder_dims = [128, 64, 32]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        #priv_encoder_dims = [64, 20]
        priv_encoder_dims = []
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1

        tanh_encoder_output = False
        num_costs = 9

        teacher_act = False
        imi_flag = False
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'THUtest'
        experiment_name = 'THU_go2'
        policy_class_name = 'ActorCriticMixedBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'PPO'
        max_iterations = 100
        num_steps_per_env = 2
        resume = False
        resume_path = ''
 

  
