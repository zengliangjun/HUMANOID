import os
from typing import Optional
try:
    from gymnasium.wrappers import video_recorder
    from gymnasium import logger
except:
    from gym.wrappers.monitoring import video_recorder
    from gym import logger

from datetime import datetime

class RecordVideo():
    def __init__(
        self,
        env,
        video_folder: str,
        start_steps: 600,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        disable_logger: bool = True,
    ):
        self.env = env
        self.video_recorder: Optional[video_recorder.VideoRecorder] = None

        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        if os.path.isdir(self.video_folder):
            logger.warn(
                f"Overwriting existing videos at {self.video_folder} folder "
                f"(try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        self.disable_logger = disable_logger
        self.name_prefix = name_prefix
        self.video_length = video_length
        self.start_steps = start_steps

        self.recording = False
        #self.terminated = False
        #self.truncated = False
        self.recorded_frames = 0
        self.episode_id = 0

        self.timestep = 0

    def _start_video_recorder(self):
        """Starts video recorder using :class:`video_recorder.VideoRecorder`."""
        self._close_video_recorder()

        video_name = datetime.now().strftime("%Y%m%d%H%M%S")
        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=base_path,
            metadata={"step_id": self.timestep, "episode_id": self.episode_id},
            disable_logger=self.disable_logger,
        )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True

    def _step(self):
        assert self.video_recorder is not None
        self.video_recorder.capture_frame()
        self.recorded_frames += 1

    def _close_video_recorder(self):
        """Closes the video recorder if currently recording."""
        if self.recording:
            assert self.video_recorder is not None
            self.video_recorder.close()
        self.recording = False
        self.recorded_frames = 1

    def __call__(self):
        self.timestep += 1

        if self.timestep > self.video_length + self.start_steps:
            return

        if self.timestep < self.start_steps:
            return

        if self.timestep == self.start_steps:
            self._start_video_recorder()
            return

        # Exit the play loop after recording one video
        if self.timestep == self.video_length + self.start_steps:
            self._close_video_recorder()
            return

        self._step()
