import os
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
            "task=Formation",
            "task.drone_model.name=Hummingbird",
            "task.drone_model.controller=LeePositionController",
            "task.env.num_envs=1",
            "headless=true",
            "total_frames=200",
            "task.formation=radar5_three",
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

import omni.replicator.core as rep
from PIL import Image

FRAME_DIR = "runs/radar5_single_scene_seed7000000/frames"
os.makedirs(FRAME_DIR, exist_ok=True)

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

import omni_drones.envs.formation as formation_task

formation_task.FORMATIONS["radar5_three"] = [
    [0.0, 0.0, 1.8],
    [2.0, 0.0, 1.8],
    [0.0, 2.0, 1.8],
]

# Formation's nested vmap is incompatible with this Isaac/PyTorch build.
# Keep the outer vmap in _reset_idx and use a plain per-environment cost.
def _radar5_formation_cost(p, desired_p):
    p = p - p.mean(-2, keepdim=True)
    desired_p = desired_p - desired_p.mean(-2, keepdim=True)
    return torch.cdist(p, desired_p).min(-1).values.max(-1).values.unsqueeze(-1)

formation_task.cost_formation_hausdorff = _radar5_formation_cost

from omni_drones.envs.isaac_env import IsaacEnv

env_class = IsaacEnv.REGISTRY[cfg.task.name]
env = env_class(cfg, headless=True)
env.set_seed(7000000)
if env.formation.ndim == 2:
    env.formation = env.formation.unsqueeze(0)
td = env.reset()
env.controller = env.controller.to(env.device)

from pxr import Gf, UsdGeom
import omni.usd

_stage = omni.usd.get_context().get_stage()
_evader_sphere = UsdGeom.Sphere.Define(_stage, "/World/evader")
_evader_sphere.CreateRadiusAttr(0.25)
_evader_xform = UsdGeom.Xformable(_evader_sphere.GetPrim())
_evader_translate = _evader_xform.AddTranslateOp()
_evader_translate.Set(Gf.Vec3d(4.0, 4.0, 1.8))

viz_camera = rep.create.camera(
    position=(8.0, -10.0, 9.0),
    look_at=(0.0, 0.0, 1.5),
)
viz_light = rep.create.light(
    light_type="sphere",
    position=(3.0, -4.0, 10.0),
    intensity=50000,
    scale=5.0,
)
render_product = rep.create.render_product(
    viz_camera,
    (960, 720),
)

rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annotator.attach([render_product])

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

# Align Radar5 target coordinates with the Isaac Formation world frame.

actor = FrozenPursuitActor(

    CHECKPOINT,
    device="cpu",
    expected_episode=3000,
)

print("ROLLOUT_SETUP_OK", flush=True)
decisions = 31
physics_per_decision = 13
heights = []
trajectory = []
scene_metadata = {
    "buildings": [list(map(float, rect)) for rect in bridge.env.buildings],
    "building_heights": [float(h) for h in bridge.env.building_heights],
    "grid_size": float(bridge.env.cfg.grid_size),
    "min_altitude": float(bridge.env.cfg.min_altitude),
    "max_altitude": float(bridge.env.cfg.max_altitude),
}
captures = 0

try:
    for decision in range(decisions):
        print("ROLLOUT_DECISION_START", decision + 1, flush=True)
        state = env.drone.get_state(check_nan=True)
        physical = omnidrones_state_to_radar5(
            state.detach().cpu().numpy()
        )
        physical = np.asarray(physical).squeeze(0)
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
        ).unsqueeze(0)

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
            try:
                td = env.step(td.set(("agents", "action"), motor_action))
            except BaseException as exc:
                print("ROLLOUT_STEP_EXCEPTION", type(exc).__name__, repr(exc), flush=True)
                raise
            print("ROLLOUT_STEP_OK", flush=True)

        state_after = env.drone.get_state(check_nan=True)
        heights.append(float(state_after[..., 2].min().item()))

        radar_result = bridge.advance()

        target_xyz = np.asarray([
            bridge.env.dynamic_targets[0, 0],
            bridge.env.dynamic_targets[0, 1],
            bridge.env.target_altitudes[0],
        ], dtype=np.float32)
        _evader_translate.Set(Gf.Vec3d(*target_xyz.tolist()))

        pursuer_xyz = state_after[..., :3].detach().cpu().numpy().reshape(3, 3)
        min_distance = float(np.linalg.norm(pursuer_xyz - target_xyz[None, :], axis=1).min())
        if decision == 0:
            print("WORLD_COORD_DEBUG", json.dumps({"pursuers": pursuer_xyz.tolist(), "target": target_xyz.tolist()}), flush=True)
        if min_distance <= 1.0:
            captures = 1
            rep.orchestrator.step(rt_subframes=4)
            final_frame = rgb_annotator.get_data()
            Image.fromarray(final_frame[:, :, :3]).save(
                os.path.join(FRAME_DIR, "capture_final.png")
            )

        rep.orchestrator.step(rt_subframes=2)
        frame = rgb_annotator.get_data()
        Image.fromarray(frame[:, :, :3]).save(
            os.path.join(FRAME_DIR, f"frame_{decision + 1:04d}.png")
        )

        trajectory.append({
            "decision": decision,
            "time_s": float((decision + 1) * bridge.env.cfg.decision_dt),
            "pursuers_xyz": pursuer_xyz.tolist(),
            "target_xyz": target_xyz.tolist(),
            "pursuers_quat": state_after[..., 3:7].detach().cpu().numpy().reshape(3, 4).tolist(),
            "capture": bool(captures),
        })

        if decision < 5 or decision % 5 == 4 or captures:
            print(
                "ISAAC_RADAR5_DECISION",
                json.dumps(
                    {
                        "decision": decision + 1,
                        "time_s": round((decision + 1) * 0.2, 3),
                        "min_height_m": min(heights),
                        "capture": bool(captures),
                        "min_distance_m": min_distance,
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
    with open(
        "runs/radar5_single_scene_seed7000000/trajectory.json",
        "w",
    ) as handle:
        json.dump(
            {
                "scene": scene_metadata,
                "frames": trajectory,
            },
            handle,
            indent=2,
        )

finally:
    simulation_app.close()
