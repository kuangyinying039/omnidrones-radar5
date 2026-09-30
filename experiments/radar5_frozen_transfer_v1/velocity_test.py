import json
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from omni_drones import init_simulation_app

CONFIG_DIR = "/opt/lab/src/OmniDrones-isaac41/cfg"

with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
    cfg = compose(
        config_name="train",
        overrides=[
            "task=Hover",
            "task.drone_model.name=Hummingbird",
            "task.drone_model.controller=LeePositionController",
            "task.env.num_envs=1",
            "headless=true",
            "total_frames=200",
            "++sim.device=cuda:0",
        ],
    )

OmegaConf.resolve(cfg)
OmegaConf.set_struct(cfg, False)

# The standalone Hydra composition does not import every sim_base field.
cfg.sim.dt = 0.016
cfg.sim.substeps = 1
cfg.sim.gravity = [0.0, 0.0, -9.81]
cfg.sim.device = "cuda:0"
cfg.sim.use_gpu_pipeline = True
cfg.sim.replicate_physics = False
cfg.sim.use_flatcache = True

simulation_app = init_simulation_app(cfg)

from omni_drones.envs.isaac_env import IsaacEnv

env_class = IsaacEnv.REGISTRY[cfg.task.name]
env = env_class(cfg, headless=True)

env.set_seed(7000000)
td = env.reset()

heights = []
speeds = []

# Hold the position sampled at reset. The controller converts this
# position reference into four normalized rotor commands.
env.controller = env.controller.to(env.device)
initial_state = env.drone.get_state(check_nan=True)
hold_position = initial_state[..., :3].clone()

forward_vx = []
start_x = float(initial_state[..., 0].mean().item())

for step in range(220):
    state_before = env.drone.get_state(check_nan=True)

    # 60 steps hover, 100 steps fly at +0.3 m/s, 60 steps stop.
    target_vel = torch.zeros_like(hold_position)
    if 60 <= step < 160:
        target_vel[..., 0] = 0.3

    # Hold altitude while letting the XY position reference follow the drone.
    target_pos = state_before[..., :3].clone()
    target_pos[..., 2] = hold_position[..., 2]

    with torch.no_grad():
        action = env.controller.compute(
            state_before[..., :13],
            target_pos=target_pos,
            target_vel=target_vel,
        ).clamp(-1.0, 1.0)

    td = env.step(td.set(("agents", "action"), action))
    state = env.drone.get_state(check_nan=True)

    heights.append(float(state[..., 2].mean().item()))
    speeds.append(float(torch.linalg.norm(state[..., 7:10], dim=-1).mean().item()))
    forward_vx.append(float(state[..., 7].mean().item()))

result = {
    "steps": 220,
    "physics_dt": float(cfg.sim.dt),
    "asset": "Hummingbird",
    "min_height_m": min(heights),
    "max_height_m": max(heights),
    "final_height_m": heights[-1],
    "max_speed_m_s": max(speeds),
}

print("HOVER_TEST_RESULT " + json.dumps(result), flush=True)

print("VELOCITY_TEST_RESULT " + json.dumps({
    "hover_mean_vx_m_s": sum(forward_vx[:60]) / 60,
    "forward_mean_vx_m_s": sum(forward_vx[100:160]) / 60,
    "stop_final_vx_m_s": forward_vx[-1],
    "forward_displacement_m": float(state[..., 0].mean().item()) - start_x,
    "min_height_m": min(heights),
    "max_height_m": max(heights),
}), flush=True)

simulation_app.close()
