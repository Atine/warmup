import os

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from .muscle_arm import MuscleArm


class HumanReacher(MuscleArm):
    def __init__(self, **kwargs):
        self.model_type = "humanreacher"
        self.tracking_str = "endeffector"
        self.nq = 24
        super(MuscleArm, self).__init__(**kwargs)
        self.set_gravity([0, 0, -9.81])
        self.has_init = True

    def reset_model(self):
        self.randomise_init_state(diff=0.03)
        if self.random_goals:
            self.target = self.sample_rectangular_goal()
        self.data.qpos[-3:] = self.target[:3]
        return self._get_obs()

    def sample_rectangular_goal(self):
        return np.random.uniform([0.35, -0.3, 0.75], [0.45, 0.0, 1.00])

    def sample_circular_goal(self):
        rho = np.random.uniform(-1.0, 0)
        phi = np.random.uniform(0.5 * np.pi, np.pi)
        # no typo here, MuJoCo coordinates are rotated wrt the distribution
        y = rho * np.cos(phi)
        x = rho * np.sin(phi)
        return np.array([x, y, 0.0])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.4
        self.viewer.cam.lookat[:] = [0.15, -0.0, 1.35]
        self.viewer.cam.elevation = -90
        self.viewer.cam.azimuth = 180

    @property
    def xml_path(self):
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../xml_files/humanreacher.xml"
        )
        return path

    def reinitialise(self, args):
        """if we want to load from specific xml, not the creator"""
        self.need_reinit = 0
        while True:
            try:
                # second one is frameskip
                observation_space = Box(
                low=-np.inf, high=np.inf, shape=(254,), dtype=np.float64)
                MujocoEnv.__init__(self, self.xml_path, self.frameskip, observation_space)
                break
            except FileNotFoundError:
                print("xml file not found, reentering loop.")
        utils.EzPickle.__init__(self)
        self._set_action_space()

    def _get_obs(self):
        """Creates observation for MDP.
        The choice here is to either use normalized com_vel in state and reward or just in reward. I could
        imagine it leading to faster learning when normalized, as larger velocities don't constitute "new"
        state space regions. But it also shifts the learning target.
        Removed adaptive scaling."""
        return np.concatenate(
            [
                self.data.qpos[: self.nq],
                self.data.qvel[: self.nq],
                self.muscle_length(),
                self.muscle_velocity(),
                self.muscle_force(),
                self.muscle_activity(),
                self.target,
                self.data.site(self.tracking_str).xpos
            ]
        )
