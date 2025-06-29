
import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

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

from motion_tasks.motions_player import asap_g1
from isaaclabmotion.envs import env_motions
from isaaclabex.assets.robots import unitree_asapg129dof23
import torch


def main():

    env_cfg = asap_g1.ASAPEnvCfg()
    env_cfg.scene.robot = unitree_asapg129dof23.ASAPG1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    env_cfg.scene.num_envs = args_cli.num_envs
    # setup base environment
    env = env_motions.ManagerMotionsEnv(cfg=env_cfg)

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
