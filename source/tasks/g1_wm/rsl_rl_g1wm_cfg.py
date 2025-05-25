from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from rsl_rlex.wm import wm_cfg

@configclass
class G1WMCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 1000
    experiment_name = "g1_wm"
    empirical_normalization = False
    policy = wm_cfg.WMActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",

        ## for wm
        continuous_decoder_dims=[256],
        discrete_decoder_dims=[256],

        proprioceptive_names=[
            'ang_vel',
            'gravity',
            'commands',
            'joint_pos',
            'joint_vel',
            'actions'
        ],
        continuous_names=[
            'ang_vel',
            'gravity',
            'commands',
            'joint_pos',
            'joint_vel',
            'actions',
            'lin_vel',
            'contact_forces',
            'payload',
            'stiffness',
            'damping'
        ],
        discreate_names=['contact_status'],
        rnn_type ="lstm",
        rnn_encoder_hidden_size = 256,
        rnn_encoder_num_layers = 1,

        rnn_actor_hidden_size = 256,
        rnn_actor_num_layers = 1,

        rnn_critic_hidden_size = 256,
        rnn_critic_num_layers = 1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="WMPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.5e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
