import json
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from omni_drones import init_simulation_app
import sys
sys.argv.extend(["--portable-root", "/srv/data/projects/kyy/omnidrones/.kit_portable"])

from experiments.radar5_frozen_transfer_v1.state_adapter import omnidrones_state_to_radar5
from experiments.radar5_frozen_transfer_v1.isaac_pursuit_observation import (
    IsaacPursuitObservationBridge,
)
from experiments.radar5_frozen_transfer_v1.frozen_pursuit_actor import FrozenPursuitActor

CONFIG_DIR = "/opt/lab/src/OmniDrones-isaac41/cfg"
CHECKPOINT = "data/checkpoints/mappo_trained_seed11_actor.pt"

with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
    cfg = compose(
        config_name="train",
        overrides=[
            "task=Hover",
            "task.drone_model.name=Hummingbird",
            "task.drone_model.controller=LeePositionController",
            "task.env.num_envs=3",
            "headless=true",
            "total_frames=200",
            "++sim.device=cuda:0",
        ],
    )

OmegaConf.resolve(cfg)
OmegaConf.set_struct(cfg, False)
cfg.sim.dt = 0.016
cfg.sim.substeps = 1
cfg.sim.gravity = [0.0, 0.0, -9.81]
cfg.sim.device = "cuda:0"
cfg.sim.use_gpu_pipeline = True
cfg.sim.replicate_physics = False
cfg.sim.use_flatcache = True

simulation_app = init_simulation_app(cfg)

# Use a local geometric ground plane instead of the remote default_environment.usd.
from omni.isaac.core.objects import GroundPlane
import omni_drones.utils.kit as kit_utils

def _local_ground_plane(
    prim_path,
    z_position=0.0,
    static_friction=1.0,
    dynamic_friction=1.0,
    restitution=0.0,
    color=(0.065, 0.0725, 0.080),
    **kwargs,
):
    return GroundPlane(
        prim_path=prim_path,
        size=200.0,
        z_position=z_position,
        color=np.asarray(color, dtype=np.float32),
    )

kit_utils.create_ground_plane = _local_ground_plane

from omni_drones.envs.isaac_env import IsaacEnv

env_class = IsaacEnv.REGISTRY[cfg.task.name]
env = env_class(cfg, headless=True)
env.set_seed(7000000)
td = env.reset()
env.controller = env.controller.to(env.device)

bridge = IsaacPursuitObservationBridge(
    __import__(
        "experiments.radar5_frozen_transfer_v1.quadrotor_pursuit_env",
        fromlist=["QuadrotorPursuitConfig"],
    ).QuadrotorPursuitConfig(
        n_uavs=3,
        n_targets=1,
        seed=7000000,
    )
)
bridge.reset()

actor = FrozenPursuitActor(
    CHECKPOINT,
    device="cpu",
    expected_episode=3000,
)

print("ROLLOUT_SETUP_OK", flush=True)
decisions = 31
physics_per_decision = 13
heights = []
captures = 0

try:
    for decision in range(decisions):
        print("ROLLOUT_DECISION_START", decision + 1, flush=True)
        state = env.drone.get_state(check_nan=True)
        physical = omnidrones_state_to_radar5(
            state.detach().cpu().numpy()
        )
        physical = np.asarray(physical).squeeze(1)
        print("ROLLOUT_STATE_OK", flush=True)

        print("PHYSICAL_SHAPE", physical.shape, "FINITE", np.isfinite(physical).all(), flush=True)
        print("ROLLOUT_SYNC_START", flush=True)
        bridge.sync_pursuers(physical)
        print("ROLLOUT_SYNC_OK", flush=True)
        print("ROLLOUT_OBSERVE_START", flush=True)
        observation = bridge.observe()
        print("ROLLOUT_OBSERVE_OK", flush=True)
        print("ROLLOUT_OBSERVE_OK", flush=True)

        print("ROLLOUT_ACTOR_START", flush=True)
        action = np.asarray(
            actor.act(
                observation["agent_observations"],
                observation["comm_adjacency"],
                observation["hetero_graph"],
            ),
            dtype=np.float32,
        )

        velocity, safety_reasons = bridge.filter_references(action[:, :3])
        print("ROLLOUT_FILTER_OK", flush=True)
        target_pos = state[..., :3].clone()
        target_pos[..., 2] = 1.8

        target_vel = torch.as_tensor(
            velocity,
            dtype=torch.float32,
            device=env.device,
        ).unsqueeze(1)

        for _ in range(physics_per_decision):
            state_before = env.drone.get_state(check_nan=True)
            print("ROLLOUT_CONTROLLER_START", flush=True)
            with torch.no_grad():
                motor_action = env.controller.compute(
                    state_before[..., :13],
                    target_pos=target_pos,
                    target_vel=target_vel,
                ).clamp(-1.0, 1.0)
            print("ROLLOUT_CONTROLLER_OK", flush=True)

            print("ROLLOUT_STEP_START", flush=True)
            td = env.step(td.set(("agents", "action"), motor_action))
            print("ROLLOUT_STEP_OK", flush=True)

        state_after = env.drone.get_state(check_nan=True)
        heights.append(float(state_after[..., 2].min().item()))

        radar_result = bridge.advance()
        if radar_result.get("capture", False):
            captures = 1

        if decision < 5 or decision % 5 == 4 or captures:
            print(
                "ISAAC_RADAR5_DECISION",
                json.dumps(
                    {
                        "decision": decision + 1,
                        "time_s": round((decision + 1) * 0.2, 3),
                        "min_height_m": min(heights),
                        "capture": bool(captures),
                        "safety_reasons": safety_reasons,
                    }
                ),
                flush=True,
            )

        if captures:
            break

    print(
        "ISAAC_RADAR5_RESULT",
        json.dumps(
            {
                "decisions": decision + 1,
                "capture": bool(captures),
                "min_height_m": min(heights),
                "checkpoint": CHECKPOINT,
            }
        ),
        flush=True,
    )
finally:
    simulation_app.close()
