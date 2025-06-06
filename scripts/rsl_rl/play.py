"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import threading
import queue

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from scripts.rsl_rl.logger_org import Logger

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--track_robot", type=bool, default=False, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import os.path as osp

_root = osp.join(osp.dirname(__file__), "../../source/")
import sys
sys.path.append(_root)

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
import tasks  # noqa: F401

def track_robot(_env):
    robot_pos_w = _env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
    cam_eye = (robot_pos_w[0] + 3, robot_pos_w[1] + 3, 4)
    cam_target = (robot_pos_w[0], robot_pos_w[1], 0.0)
    # set the camera view
    _env.unwrapped.sim.set_camera_view(eye=cam_eye, target=cam_target)

# Async print queue and worker
print_queue = queue.Queue()
print_thread = None

def print_worker():
    """Worker thread for async printing"""
    while True:
        item = print_queue.get()
        if item is None:  # Sentinel value to stop the thread
            break
        print(item)
        print_queue.task_done()

def start_print_thread():
    """Start the async print thread"""
    global print_thread
    print_thread = threading.Thread(target=print_worker, daemon=True)
    print_thread.start()

def stop_print_thread():
    """Stop the async print thread"""
    global print_thread
    if print_thread:
        print_queue.put(None)  # Send stop signal
        print_thread.join()

def async_print(text):
    """Print text asynchronously"""
    print_queue.put(text)

def main():
    """Play with RSL-RL agent."""
    # Start async print thread
    start_print_thread()
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.num_steps_per_env = agent_cfg.num_steps_per_env
    env_cfg.max_iterations = agent_cfg.max_iterations

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    if args_cli.track_robot:
        track_robot(env)

    count = 0
    ep_infos = []


    logger = Logger(env.unwrapped.step_dt)
    robot_index = 0
    joint_index = 0
    asset = env.unwrapped.scene["robot"]

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, extra = env.step(actions)
            if args_cli.track_robot:
                    track_robot(env)

            commands = env.unwrapped.command_manager.get_command("base_velocity")

            logger.log_states(
            {
                'dof_pos_target': actions[robot_index, joint_index].item() * env.unwrapped.cfg.actions.joint_pos.scale,
                'dof_pos': asset.data.joint_pos[robot_index, joint_index].item(),
                'dof_vel': asset.data.joint_vel[robot_index, joint_index].item(),
                'dof_torque': asset.data.applied_torque[robot_index, joint_index].item(),
                'command_x': commands[robot_index, 0].item(),
                'command_y': commands[robot_index, 1].item(),
                'command_yaw': commands[robot_index, 2].item(),
                'base_vel_x': asset.data.root_lin_vel_b[robot_index, 0].item(),
                'base_vel_y': asset.data.root_lin_vel_b[robot_index, 1].item(),
                'base_vel_z': asset.data.root_lin_vel_b[robot_index, 2].item(),
                'base_vel_yaw': asset.data.root_ang_vel_b[robot_index, 2].item()
            }
            )

            num_episodes = torch.sum(env.unwrapped.reset_buf).item()
            if num_episodes>0:

                if "episode" in extra:
                    ep_infos = extra["episode"]
                elif "log" in extra:
                    ep_infos = extra["log"]

                logger.log_rewards(ep_infos, num_episodes)
                logger.print_rewards()
                logger.plot_states()

            if False:
                count += 1
                if "episode" in extra:
                    ep_infos.append(extra["episode"])
                elif "log" in extra:
                    ep_infos.append(extra["log"])

                if count % env_cfg.num_steps_per_env == 0:
                    width: int = 80
                    pad: int = 35
                    ep_string = f"""\n{'#' * width}\n"""
                    for key in ep_infos[0]:
                        infotensor = torch.tensor([], device=agent_cfg.device)
                        for ep_info in ep_infos:
                            # handle scalar and zero dimensional tensor infos
                            if key not in ep_info:
                                continue
                            if not isinstance(ep_info[key], torch.Tensor):
                                ep_info[key] = torch.Tensor([ep_info[key]])
                            if len(ep_info[key].shape) == 0:
                                ep_info[key] = ep_info[key].unsqueeze(0)
                            infotensor = torch.cat((infotensor, ep_info[key].to(agent_cfg.device)))
                        value = torch.mean(infotensor)
                        # log to logger and terminal
                        if "/" in key:
                            ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                        else:
                            ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

                    async_print(ep_string)
                    ep_infos = []


        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()
    # Stop async print thread
    stop_print_thread()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
