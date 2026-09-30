"""Load the two audited Radar5 actor exports for frozen evaluation.

The actor consumes the original Radar5 flat observation, communication
adjacency and heterogeneous graph. Building those inputs from Isaac Lab
sensor data is a separate, required step.
"""

from __future__ import annotations

from pathlib import Path


REQUIRED_FIELDS = frozenset({
    "format_version", "source_algorithm", "source_episode", "source_sha256",
    "pursuit_graph_version", "obs_dim", "action_dim", "n_agents",
    "train_config", "env_config", "actor_state",
})


def validate_actor_package(package: dict, *, expected_episode: int | None = None) -> None:
    """Reject a wrong checkpoint before it can enter a physical evaluation."""
    if not isinstance(package, dict):
        raise ValueError("actor package must be a dictionary")
    missing = REQUIRED_FIELDS - package.keys()
    if missing:
        raise ValueError(f"actor package missing fields: {sorted(missing)}")
    if package["format_version"] != 1 or package["pursuit_graph_version"] != 3:
        raise ValueError("unsupported actor package or pursuit graph version")
    if expected_episode is not None and package["source_episode"] != expected_episode:
        raise ValueError(f"expected episode {expected_episode}, got {package['source_episode']}")
    if package["n_agents"] != 3 or package["action_dim"] != 4 or package["obs_dim"] <= 0:
        raise ValueError("checkpoint dimensions do not match the three-UAV pursuit task")
    if not isinstance(package["train_config"], dict) or not isinstance(package["env_config"], dict):
        raise ValueError("training and environment configurations must be dictionaries")
    if not package["train_config"].get("use_gat"):
        raise ValueError("this loader requires a pursuit graph actor")
    if int(package["env_config"].get("n_uavs", -1)) != package["n_agents"]:
        raise ValueError("environment UAV count differs from actor package")
    if not isinstance(package["actor_state"], dict) or not package["actor_state"]:
        raise ValueError("actor weights are missing")
    digest = package["source_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid source checkpoint SHA-256")


class FrozenPursuitActor:
    """Deterministic actor; never updates weights or optimizer state."""

    def __init__(self, package_path: str | Path, *, device: str = "cpu", expected_episode: int | None = None):
        import torch
        from pursuit_graph_encoder import PursuitGraphActor

        self.device = torch.device(device)
        self.package_path = Path(package_path)
        package = torch.load(self.package_path, map_location="cpu", weights_only=True)
        validate_actor_package(package, expected_episode=expected_episode)
        cfg = package["train_config"]
        env = package["env_config"]
        self.n_agents = int(package["n_agents"])
        self.obs_dim = int(package["obs_dim"])
        self.action_dim = int(package["action_dim"])
        self.metadata = {k: v for k, v in package.items() if k != "actor_state" and k not in ("train_config", "env_config")}
        self.train_config = cfg
        self.env_config = env
        self.actor = PursuitGraphActor(
            self.obs_dim, self.action_dim, int(cfg["hidden_dim"]),
            int(cfg["gat_heads"]), int(cfg["gat_layers"]),
            dropout=0.0, spatial_scale=float(env["grid_size"]),
        ).to(self.device)
        self.actor.load_state_dict(package["actor_state"], strict=True)
        self.actor.eval()
        self.actor.requires_grad_(False)

    def act(self, observation, adjacency, graph):
        """Return normalized [3, 4] velocity/yaw-rate actions."""
        import torch

        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        adj = torch.as_tensor(adjacency, dtype=torch.float32, device=self.device)
        if tuple(obs.shape) != (self.n_agents, self.obs_dim):
            raise ValueError(f"observation must have shape {(self.n_agents, self.obs_dim)}")
        if tuple(adj.shape) != (self.n_agents, self.n_agents):
            raise ValueError(f"adjacency must have shape {(self.n_agents, self.n_agents)}")
        if not torch.isfinite(obs).all() or not torch.isfinite(adj).all():
            raise ValueError("observation and adjacency must be finite")
        tensors = {
            key: torch.as_tensor(value, dtype=torch.bool if key.endswith("_mask") else torch.float32, device=self.device)
            for key, value in graph.items()
            if key.endswith("_nodes") or key.endswith("_mask") or key == "uav_xyz"
        }
        required = {"self_nodes", "target_nodes", "target_mask", "peer_nodes", "peer_mask",
                    "building_nodes", "building_mask", "uav_xyz", "active_mask"}
        if required - tensors.keys():
            raise ValueError(f"graph missing fields: {sorted(required - tensors.keys())}")
        if any(not torch.isfinite(value).all() for key, value in tensors.items() if not key.endswith("_mask")):
            raise ValueError("graph contains non-finite values")
        with torch.inference_mode():
            action = self.actor(obs, adj, tensors).tanh()
        if tuple(action.shape) != (self.n_agents, self.action_dim):
            raise ValueError("actor returned unexpected action shape")
        return action.cpu().numpy()
