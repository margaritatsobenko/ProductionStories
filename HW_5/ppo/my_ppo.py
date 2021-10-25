import os
import shutil

import ray
import ray.rllib.agents.ppo as ppo
from PIL import Image
from ray import tune
import wandb

from mapgen import Dungeon


class MyEnv(Dungeon):
    def __init__(self,
                 width=64,
                 height=64,
                 max_rooms=25,
                 min_room_xy=10,
                 max_room_xy=25,
                 observation_size: int = 11,
                 vision_radius: int = 5,
                 max_steps: int = 2000,
                 reward: str = "base"
                 ):
        super(MyEnv, self).__init__(width, height, max_rooms, min_room_xy, max_room_xy,
                                    observation_size, vision_radius, max_steps)
        self.reward = reward

    def step(self, action: int):
        obs, rew, done, info = super().step(action)

        if self.reward == "base":
            return obs, rew, done, info
        else:
            if info["moved"]:
                if info["new_explored"] > 0:
                    rew = 0.1 + info["total_explored"] / info["total_cells"] * 20
                else:
                    rew = -0.5
            else:
                rew = -1.0
            return obs, rew, done, info


if __name__ == "__main__":

    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    tune.register_env("MyEnv", lambda config: MyEnv(**config))

    CHECKPOINT_ROOT = "tmp/ppo/dungeon"
    shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

    ray_results = os.getenv("HOME") + "/ray_results1/"
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    config = ppo.DEFAULT_CONFIG.copy()
    # config["num_gpus"] = 1
    config["num_gpus"] = 0
    config["num_gpus_per_worker"] = 0
    config["log_level"] = "INFO"
    config["framework"] = "torch"
    config["env"] = "MyEnv"
    config["env_config"] = {
        "width": 20,
        "height": 20,
        "max_rooms": 3,
        "min_room_xy": 5,
        "max_room_xy": 10,
        "observation_size": 11,
        "vision_radius": 5,
        "reward": "base"  # or modified
    }

    config["model"] = {
        "conv_filters": [
            [16, (3, 3), 2],
            [32, (3, 3), 2],
            [32, (3, 3), 1],
        ],
        "post_fcnet_hiddens": [32],
        "post_fcnet_activation": "relu",
        "vf_share_layers": False,
    }

    config["rollout_fragment_length"] = 100
    config["entropy_coeff"] = 0.1
    config["lambda"] = 0.95
    config["vf_loss_coeff"] = 1.0

    agent = ppo.PPOTrainer(config)

    wandb.init(project="task_5", config=config, entity="margaritatsobenko")

    N_ITER = 500
    s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"

    for n in range(N_ITER):
        result = agent.train()
        file_name = agent.save(CHECKPOINT_ROOT)

        wandb.log({"reward_min": result["episode_reward_min"]})
        wandb.log({"reward_max": result["episode_reward_max"]})
        wandb.log({"reward_mean": result["episode_reward_mean"]})
        wandb.log({"episode_len_mean": result["episode_len_mean"]})

        wandb.log({"entropy": result["info"]["learner"]["default_policy"]["learner_stats"]["entropy"]})
        wandb.log({"vf_loss": result["info"]["learner"]["default_policy"]["learner_stats"]["vf_loss"]})
        wandb.log({"policy_loss": result["info"]["learner"]["default_policy"]["learner_stats"]["policy_loss"]})
        wandb.log({"total_loss": result["info"]["learner"]["default_policy"]["learner_stats"]["total_loss"]})

        print(s.format(
            n + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"],
            file_name
        ))

        # sample trajectory
        if (n + 1) % 5 == 0:
            wandb.save(file_name)
            env = Dungeon(20, 20, 3, min_room_xy=5, max_room_xy=10, vision_radius=5)
            obs = env.reset()
            Image.fromarray(env._map.render(env._agent)).convert('RGB').resize((500, 500), Image.NEAREST).save(
                'tmp.png')

            frames = []

            for _ in range(500):
                action = agent.compute_single_action(obs)

                frame = Image.fromarray(env._map.render(env._agent)).convert('RGB').resize((500, 500),
                                                                                           Image.NEAREST).quantize()
                frames.append(frame)

                # frame.save('tmp1.png')
                obs, reward, done, info = env.step(action)
                if done:
                    break

            frames[0].save(f"out.gif", save_all=True, append_images=frames[1:], loop=0, duration=1000 / 60)
            wandb.log({"agent_run": wandb.Video("out.gif", fps=4, format="gif")})
