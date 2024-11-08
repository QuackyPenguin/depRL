"""Builders for distributed training."""

import multiprocessing

import numpy as np

from deprl.utils import stdout_suppression


i = 0


def proc(
    action_pipe,
    output_queue,
    group_seed,
    build_dict,
    max_episode_steps,
    index,
    workers,
    env_args,
    header,
    env_queue,
):
    """Process holding a sequential group of environments."""
    envs = Sequential(build_dict, max_episode_steps, workers, env_args, header)
    envs.initialize(group_seed)

    observations = envs.start()
    output_queue.put((index, observations))

    while True:
        message = action_pipe.recv()

        if isinstance(message, str):
            # interpret message as a command instead of parameters for the step method
            if message == "get_head_pos":
                for env in envs.environments:
                    env_queue.put(env.unwrapped.head_body.com_pos().array())
            elif message == "get_model_pos":
                for env in envs.environments:
                    env_queue.put(env.unwrapped.model.contact_power())
            elif message == "get_reward_scale":
                for env in envs.environments:
                    env_queue.put(env.unwrapped.reward_scale)
            elif message == "get_vel":
                # print('message get_vel', message)
                for env in envs.environments:
                    # print('env get_vel', env)
                    # print('env_queue', env_queue)
                    # print('env_queue.put', env_queue.put)
                    # env.unwrapped.set_current_target_velocity(1) # Test setting vel for testing
                    #print('current_target_vel:', env.unwrapped.get_current_target_velocity())
                    env_queue.put(
                        (
                            env.unwrapped.model_velocity(),
                            env.unwrapped.current_target_vel,
                        )
                    )
            elif message == "get_angle":
                # index_test=0
                for env in envs.environments:
                    # print('env index', index_test)
                    # index_test+=1
                    env_queue.put(
                        (
                            np.arctan2(
                                env.unwrapped.model.com_vel().z,
                                env.unwrapped.model.com_vel().x,
                            ),
                            env.unwrapped.angle,
                        )
                    )

            continue

        # message is a tuple of actions, angle_range, and vel_range
        # those are the parameters for the step method

        # print('custom_distributed len message', len(message))

        actions, angle_range, vel_range, stand_prob, new_task = message

        # print('custom_distributed angle_range', angle_range)
        # print('custom_distributed vel_range', vel_range)

        out = envs.step(actions, angle_range, vel_range, stand_prob, new_task)
        output_queue.put((index, out))


class Sequential:
    """A group of environments used in sequence."""

    def __init__(
        self, build_dict, max_episode_steps, workers, env_args, header
    ):
        if header is not None:
            with stdout_suppression():
                exec(header)
        if hasattr(build_env_from_dict(build_dict).unwrapped, "environment"):
            # its a deepmind env
            self.environments = [
                build_env_from_dict(build_dict)() for i in range(workers)
            ]
        else:
            # its a gym env
            self.environments = [
                build_env_from_dict(build_dict) for i in range(workers)
            ]
        if env_args is not None:
            [x.merge_args(env_args) for x in self.environments]
            [x.apply_args() for x in self.environments]
        self._max_episode_steps = max_episode_steps
        self.observation_space = self.environments[0].observation_space
        self.action_space = self.environments[0].action_space
        self.name = self.environments[0].name
        self.num_workers = workers

    def initialize(self, seed):
        # group seed is given, the others are determined from it
        for i, environment in enumerate(self.environments):
            environment.seed(seed + i)

    def start(self):
        """Used once to get the initial observations."""
        observations = [env.reset() for env in self.environments]
        muscle_states = [env.muscle_states for env in self.environments]
        self.lengths = np.zeros(len(self.environments), int)
        return np.array(observations, np.float32), np.array(
            muscle_states, np.float32
        )

    def step(
        self,
        actions,
        angle_range: tuple = (-np.pi / 4, np.pi / 4),
        vel_range: tuple = (0.25, 1.0),
        stand_prob: float = 0.0,
        new_task: int = 0,
    ):
        next_observations = []  # Observations for the transitions.
        rewards = []
        resets = []
        terminations = []
        observations = []  # Observations for the actions selection.
        muscle_states = []

        for i in range(len(self.environments)):
            ob, rew, term, env_info = self.environments[i].step(
                actions[i],
                angle_range=angle_range,
                vel_range=vel_range,
                stand_prob=stand_prob,
                new_task=new_task,
            )
            muscle = self.environments[i].muscle_states
            self.lengths[i] += 1
            # Timeouts trigger resets but are not true terminations.
            reset = term or self.lengths[i] == self._max_episode_steps
            next_observations.append(ob)
            rewards.append(rew)
            resets.append(reset)

            terminations.append(term)

            if reset:
                ob = self.environments[i].reset(
                    angle_range=angle_range,
                    vel_range=vel_range,
                    stand_prob=stand_prob,
                    new_task=new_task,
                )
                muscle = self.environments[i].muscle_states
                self.lengths[i] = 0

            observations.append(ob)
            muscle_states.append(muscle)

        observations = np.array(observations, np.float32)
        muscle_states = np.array(muscle_states, np.float32)
        infos = dict(
            observations=np.array(next_observations, np.float32),
            rewards=np.array(rewards, np.float32),
            resets=np.array(resets, bool),
            terminations=np.array(terminations, bool),
        )
        return observations, muscle_states, infos

    def render(self, mode="human", *args, **kwargs):
        outs = []
        for env in self.environments:
            out = env.render(mode=mode, *args, **kwargs)
            outs.append(out)
        if mode != "human":
            return np.array(outs)

    def render_substep(self):
        for env in self.environments:
            env.render_substep()


class Parallel:
    """A group of sequential environments used in parallel."""

    def __init__(
        self,
        build_dict,
        worker_groups,
        workers_per_group,
        max_episode_steps,
        env_args,
        header,
    ):
        self.build_dict = build_dict
        self.worker_groups = worker_groups
        self.workers_per_group = workers_per_group
        self._max_episode_steps = max_episode_steps
        self.env_args = env_args
        self.header = header

    def initialize(self, seed):
        dummy_environment = build_env_from_dict(self.build_dict)
        dummy_environment.merge_args(self.env_args)
        dummy_environment.apply_args()

        self.observation_space = dummy_environment.observation_space
        self.action_space = dummy_environment.action_space
        del dummy_environment
        self.started = False
        # this prevents issues with GH actions and multiple start method inits
        # spawn works across all operating systems
        context = multiprocessing.get_context("spawn")
        self.output_queue = context.Queue()
        self.env_queue = context.Queue()
        self.action_pipes = []
        self.processes = []

        for i in range(self.worker_groups):
            pipe, worker_end = context.Pipe()
            self.action_pipes.append(pipe)
            group_seed = (
                seed * self.workers_per_group + i * self.workers_per_group
            )

            # required for spawnstart_method for macos and windows
            proc_kwargs = {
                "action_pipe": worker_end,
                "output_queue": self.output_queue,
                "group_seed": group_seed,
                "build_dict": self.build_dict,
                "max_episode_steps": self._max_episode_steps,
                "index": i,
                "workers": self.workers_per_group,
                "env_queue": self.env_queue,
                "env_args": (
                    self.env_args if hasattr(self, "env_args") else None
                ),
                "header": self.header,
            }

            self.processes.append(
                context.Process(target=proc, kwargs=proc_kwargs)
            )
            self.processes[-1].daemon = True
            self.processes[-1].start()
            #print(f"Initializing {self.worker_groups} worker groups with {self.workers_per_group} workers each.")

    def start(self):
        """Used once to get the initial observations."""
        assert not self.started
        self.started = True
        observations_list = [None for _ in range(self.worker_groups)]
        muscle_states_list = [None for _ in range(self.worker_groups)]

        for _ in range(self.worker_groups):
            index, (observations, muscle_states) = self.output_queue.get()
            observations_list[index] = observations
            muscle_states_list[index] = muscle_states

        self.observations_list = np.array(observations_list)
        self.muscle_states_list = np.array(muscle_states_list)
        self.next_observations_list = np.zeros_like(self.observations_list)
        self.rewards_list = np.zeros(
            (self.worker_groups, self.workers_per_group), np.float32
        )
        self.resets_list = np.zeros(
            (self.worker_groups, self.workers_per_group), bool
        )
        self.terminations_list = np.zeros(
            (self.worker_groups, self.workers_per_group), bool
        )
        self.terminations_list = np.zeros(
            (self.worker_groups, self.workers_per_group), bool
        )

        return np.concatenate(self.observations_list), np.concatenate(
            self.muscle_states_list
        )
    

    def step(
        self,
        actions,
        angle_range: tuple = (-np.pi / 4, np.pi / 4),
        vel_range: tuple = (0.25, 1.0),
        stand_prob: float = 0.0,
        new_task: int = 0,
    ):
        actions_list = np.split(actions, self.worker_groups)
        for actions, pipe in zip(actions_list, self.action_pipes):
            pipe.send((actions, angle_range, vel_range, stand_prob, new_task))

        for _ in range(self.worker_groups):
            index, (
                observations,
                muscle_states,
                infos,
            ) = self.output_queue.get()
            self.observations_list[index] = observations
            self.next_observations_list[index] = infos["observations"]
            self.rewards_list[index] = infos["rewards"]
            self.resets_list[index] = infos["resets"]
            self.terminations_list[index] = infos["terminations"]
            self.muscle_states_list[index] = muscle_states

        observations = np.concatenate(self.observations_list)
        muscle_states = np.concatenate(self.muscle_states_list)
        infos = dict(
            observations=np.concatenate(self.next_observations_list),
            rewards=np.concatenate(self.rewards_list),
            resets=np.concatenate(self.resets_list),
            terminations=np.concatenate(self.terminations_list),
        )
        # print(f"Sending actions to workers: {actions}")
        return observations, muscle_states, infos

    def close(self):
        print("Before Terminating processes...")
        self.proc.terminate()
        print("After Terminating processes...")

    def get_head_pos(self):
        for pipe in self.action_pipes:
            pipe.send("get_head_pos")
        heads = []
        for _ in self.action_pipes:
            for _ in range(self.workers_per_group):
                heads.append(self.env_queue.get())

        return heads

    def get_model_pos(self):
        for pipe in self.action_pipes:
            pipe.send("get_model_pos")
        models = []
        for _ in self.action_pipes:
            for _ in range(self.workers_per_group):
                models.append(self.env_queue.get())

    def get_vel(self):
        for pipe in self.action_pipes:
            pipe.send("get_vel")
        vels = []
        for _ in self.action_pipes:
            for _ in range(self.workers_per_group):
                vels.append(self.env_queue.get())

        return vels
    
    def get_reward_scale(self):
        # print("Requesting reward_scale from workers...")
        for pipe in self.action_pipes:
            pipe.send("get_reward_scale")
        reward_scaled = []
        for _ in self.action_pipes:
            for _ in range(self.workers_per_group):
                reward_scaled.append(self.env_queue.get())
                # print('self.env_queue:', self.env_queue.unwrapped)
                # try:
                #     scale = self.env_queue.get(timeout=5)  # Set a timeout
                #     print(f"Received reward_scale: {scale}")
                #     reward_scales.append(scale)
                # except self.env_queue.Empty:
                #     print("No response from worker within timeout period.")
        
        # print("Completed collecting reward_scale.")
        return reward_scaled[0]

    def get_angles(self):
        for pipe in self.action_pipes:
            pipe.send("get_angle")
        angles = []
        for _ in self.action_pipes:
            for _ in range(self.workers_per_group):
                angles.append(self.env_queue.get())

        return angles

    def curriculum_adjust(self, score):
        pass


def distribute(
    environment,
    tonic_conf,
    env_args,
    parallel=None,
    sequential=None,
):
    """Distributes workers over parallel and sequential groups."""
    parallel = tonic_conf["parallel"] if parallel is None else parallel
    sequential = tonic_conf["sequential"] if sequential is None else sequential
    build_dict = dict(
        env=environment, parallel=parallel, sequential=sequential
    )

    dummy_environment = build_env_from_dict(build_dict)
    max_episode_steps = dummy_environment._max_episode_steps
    del dummy_environment

    if parallel < 2:
        return Sequential(
            build_dict=build_dict,
            max_episode_steps=max_episode_steps,
            workers=sequential,
            env_args=env_args,
            header=tonic_conf["header"],
        )
    return Parallel(
        build_dict,
        worker_groups=parallel,
        workers_per_group=sequential,
        max_episode_steps=max_episode_steps,
        env_args=env_args,
        header=tonic_conf["header"],
    )


def build_env_from_dict(build_dict):
    assert build_dict["env"] is not None
    if type(build_dict) == dict:
        from deprl import env_tonic_compat

        return env_tonic_compat(**build_dict)
    else:
        return build_dict()
