
import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

#####################################################################
import sys
import torch
import os.path as osp

_root = osp.join(osp.dirname(__file__), "../../../source/")
sys.path.append(_root)
#####################################################################

from motion_tasks.motions_player import asap_h1
from isaaclabmotion.envs import env_motions
from isaaclabex.assets.robots import unitree_omnih2oh1
import torch


"""
# Root
root_pos torch.Size([16, 3])
root_rot torch.Size([16, 4])
root_vel torch.Size([16, 3])
root_ang_vel torch.Size([16, 3])

# Links
rg_pos torch.Size([16, 20, 3])
rb_rot torch.Size([16, 20, 4])
body_vel torch.Size([16, 20, 3])
body_ang_vel torch.Size([16, 20, 3])

# Extended links
rg_pos_t torch.Size([16, 22, 3])
rg_rot_t torch.Size([16, 22, 4])
body_vel_t torch.Size([16, 22, 3])
body_ang_vel_t torch.Size([16, 22, 3])

# JOINT
dof_pos torch.Size([16, 19])
dof_vel torch.Size([16, 19])

motion_aa torch.Size([16, 60])
motion_bodies torch.Size([16, 17])
"""


def main():

    env_cfg = asap_h1.OMNIH2OH1EnvCfg()
    env_cfg.scene.robot = unitree_omnih2oh1.OMNIH1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    env_cfg.scene.num_envs = args_cli.num_envs
    # setup base environment
    env = env_motions.ManagerMotionsEnv(cfg=env_cfg)

    env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset(env_ids = env_ids)
    # simulate physics
    reset = None
    while simulation_app.is_running():
        if reset is not None:
            reset_env_ids = reset.nonzero(as_tuple=False).squeeze(-1)
            env.reset(env_ids = reset_env_ids)

        reset = env.step()

    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
