"""Pursuit-specific 3-D heterogeneous graph and actor.

This module is deliberately independent from the planar search HGAT.  Actor
target entities contain only per-UAV track estimates; environment truth is not
an input to any function in this file.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


PURSUIT_GRAPH_VERSION = 3
SELF_DIM = 16
TARGET_DIM = 8
PEER_DIM = 7
BUILDING_DIM = 7


def _signed_box_clearance(point, lower, upper):
    """Signed Euclidean clearance to an axis-aligned solid box."""
    outside = point - np.clip(point, lower, upper)
    distance = float(np.linalg.norm(outside))
    if distance > 1e-12:
        return distance
    return -float(np.min(np.r_[point - lower, upper - point]))


def pursuit_graph_observation(env):
    """Build policy-causal target, peer and known-building entities."""
    cfg = env.cfg
    n_agents = cfg.n_uavs
    capacity = max(int(cfg.building_state_capacity), int(cfg.building_count))
    graph = {
        "self_nodes": np.zeros((n_agents, SELF_DIM), dtype=np.float32),
        "target_nodes": np.zeros((n_agents, 1, TARGET_DIM), dtype=np.float32),
        "target_mask": np.zeros((n_agents, 1), dtype=bool),
        "peer_nodes": np.zeros((n_agents, n_agents, PEER_DIM), dtype=np.float32),
        "peer_mask": np.zeros((n_agents, n_agents), dtype=bool),
        "building_nodes": np.zeros((n_agents, capacity, BUILDING_DIM), dtype=np.float32),
        "building_mask": np.zeros((n_agents, capacity), dtype=bool),
        "uav_xyz": np.zeros((n_agents, 3), dtype=np.float32),
        "active_mask": np.ones(n_agents, dtype=bool),
    }
    if hasattr(env, "quadrotor_states") and env.quadrotor_states.shape == (n_agents, 13):
        states = np.asarray(env.quadrotor_states, dtype=float)
    else:
        states = np.zeros((n_agents, 13), dtype=float)
        states[:, :2] = np.asarray(env.positions, dtype=float)
        states[:, 2] = np.asarray(env.altitudes, dtype=float)
        states[:, 6] = 1.0
    graph["uav_xyz"][:] = states[:, :3]
    disabled = np.asarray(getattr(env, "disabled_uavs", np.zeros(n_agents)), dtype=bool)
    disconnected = np.asarray(getattr(env, "disconnected_uavs", np.zeros(n_agents)), dtype=bool)
    graph["active_mask"][:] = ~disabled
    velocity_scale = max(cfg.max_horizontal_velocity, cfg.max_vertical_velocity, 1e-6)
    rate_scale = max(cfg.max_reference_yaw_rate, 1e-6)
    z_scale = max(cfg.max_altitude, 1e-6)
    target_speed = max(cfg.target_speed, cfg.target_vertical_speed, 1e-6)
    remaining = max(0.0, 1.0 - env.t / max(cfg.search_steps, 1))

    for agent in range(n_agents):
        position = states[agent, :3]
        graph["self_nodes"][agent] = np.r_[
            position[:2] / cfg.grid_size,
            position[2] / z_scale,
            states[agent, 3:6] / velocity_scale,
            states[agent, 6:10],
            states[agent, 10:13] / rate_scale,
            remaining,
            float(not disconnected[agent]),
            float(disabled[agent]),
        ]
        if disabled[agent]:
            continue

        if hasattr(env, "track_memory"):
            track = env.track_memory[agent][0]
            if track.initialized and len(track.mean) >= 6:
                relative = np.asarray(track.mean[:3], dtype=float) - position
                distance = float(np.linalg.norm(relative))
                try:
                    from pursuit_lidar import lidar_sees_point
                except ImportError:
                    from .pursuit_lidar import lidar_sees_point
                own_lidar = float(lidar_sees_point(env, agent, track.mean[:3]))
                graph["target_nodes"][agent, 0] = np.r_[
                    relative[:2] / cfg.grid_size,
                    relative[2] / z_scale,
                    np.asarray(track.mean[3:6], dtype=float) / target_speed,
                    min(distance / cfg.grid_size, 1.0),
                    own_lidar,
                ]
                graph["target_mask"][agent, 0] = True

        if hasattr(env, "last_graph"):
            peer_mask = np.asarray(env.last_graph[agent] > 0, dtype=bool) & ~disabled
        else:
            peer_mask = np.zeros(n_agents, dtype=bool)
        peer_mask[agent] = False
        for peer in np.flatnonzero(peer_mask):
            relative = states[peer, :3] - position
            graph["peer_nodes"][agent, peer] = np.r_[
                relative[:2] / cfg.grid_size,
                relative[2] / z_scale,
                states[peer, 3:6] / velocity_scale,
                1.0,
            ]
            graph["peer_mask"][agent, peer] = True

        for slot, (rect, height) in enumerate(zip(
                getattr(env, "buildings", []), getattr(env, "building_heights", []))):
            if slot >= capacity:
                break
            rect = np.asarray(rect, dtype=float)
            height = float(height)
            center = np.array([(rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2, height / 2])
            relative = center - position
            clearance = _signed_box_clearance(
                position, np.array([rect[0], rect[1], 0.0]), np.array([rect[2], rect[3], height])
            ) - cfg.quadrotor_clearance
            graph["building_nodes"][agent, slot] = np.r_[
                relative[:2] / cfg.grid_size,
                relative[2] / z_scale,
                (rect[2] - rect[0]) / cfg.grid_size,
                (rect[3] - rect[1]) / cfg.grid_size,
                height / z_scale,
                np.clip(clearance / cfg.grid_size, -1.0, 1.0),
            ]
            graph["building_mask"][agent, slot] = True
    return graph


def pursuit_graph_from_flat_observation(observation, cfg):
    """Upgrade a legacy demo snapshot using only its policy-visible flat vector."""
    vectors = np.asarray(observation, dtype=float)
    n_agents = int(cfg.n_uavs)
    capacity = max(int(cfg.building_state_capacity), int(cfg.building_count))
    context_width = 10 + 7 * n_agents + 5 * capacity
    base_dim = vectors.shape[-1] - context_width
    if vectors.shape[0] != n_agents or base_dim < 22:
        raise ValueError("Legacy pursuit observation has an incompatible flat layout")
    graph = {
        "self_nodes": np.zeros((n_agents, SELF_DIM), dtype=np.float32),
        "target_nodes": np.zeros((n_agents, 1, TARGET_DIM), dtype=np.float32),
        "target_mask": np.zeros((n_agents, 1), dtype=bool),
        "peer_nodes": np.zeros((n_agents, n_agents, PEER_DIM), dtype=np.float32),
        "peer_mask": np.zeros((n_agents, n_agents), dtype=bool),
        "building_nodes": np.zeros((n_agents, capacity, BUILDING_DIM), dtype=np.float32),
        "building_mask": np.zeros((n_agents, capacity), dtype=bool),
        "uav_xyz": np.zeros((n_agents, 3), dtype=np.float32),
        "active_mask": vectors[:, 20] < 0.5,
    }
    z_scale = max(cfg.max_altitude, 1e-6)
    velocity_scale = max(cfg.max_horizontal_velocity, cfg.max_vertical_velocity, 1e-6)
    target_speed = max(cfg.target_speed, cfg.target_vertical_speed, 1e-6)
    positions = np.column_stack([vectors[:, :2] * cfg.grid_size, vectors[:, 14] * z_scale])
    graph["uav_xyz"][:] = positions
    rigid = vectors[:, base_dim:base_dim + 10]
    peer_start = base_dim + 10
    building_start = peer_start + 7 * n_agents
    for agent in range(n_agents):
        graph["self_nodes"][agent] = np.r_[
            positions[agent, :2] / cfg.grid_size,
            positions[agent, 2] / z_scale,
            rigid[agent, :3], rigid[agent, 3:7], rigid[agent, 7:10],
            vectors[agent, 21], vectors[agent, 19], vectors[agent, 20],
        ]
        if graph["active_mask"][agent] and vectors[agent, 13] > 0.5:
            relative = np.r_[vectors[agent, 10:12] * cfg.grid_size, vectors[agent, 7] * z_scale]
            velocity = np.r_[vectors[agent, 8:10] * cfg.target_speed,
                             vectors[agent, 15] * cfg.target_vertical_speed]
            graph["target_nodes"][agent, 0] = np.r_[
                relative[:2] / cfg.grid_size, relative[2] / z_scale,
                velocity / target_speed, min(np.linalg.norm(relative) / cfg.grid_size, 1.0), 1.0,
            ]
            graph["target_mask"][agent, 0] = True
        peer_rows = vectors[agent, peer_start:building_start].reshape(n_agents, PEER_DIM).copy()
        # Legacy flat context normalized all xyz deltas by grid size; the new
        # schema uses altitude scale for z.
        peer_rows[:, 2] *= cfg.grid_size / z_scale
        graph["peer_nodes"][agent] = peer_rows
        graph["peer_mask"][agent] = peer_rows[:, -1] > 0.5
        graph["peer_mask"][agent, agent] = False
        buildings = vectors[agent, building_start:].reshape(capacity, 5)
        for slot, building in enumerate(buildings):
            if not np.any(building):
                continue
            rect = building[:4] * cfg.grid_size
            height = building[4] * cfg.grid_size
            center = np.array([(rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2, height / 2])
            relative = center - positions[agent]
            clearance = _signed_box_clearance(
                positions[agent], np.array([rect[0], rect[1], 0.0]), np.array([rect[2], rect[3], height])
            ) - cfg.quadrotor_clearance
            graph["building_nodes"][agent, slot] = np.r_[
                relative[:2] / cfg.grid_size, relative[2] / z_scale,
                (rect[2] - rect[0]) / cfg.grid_size, (rect[3] - rect[1]) / cfg.grid_size,
                height / z_scale, np.clip(clearance / cfg.grid_size, -1.0, 1.0),
            ]
            graph["building_mask"][agent, slot] = True
    return graph


class PursuitEntityAttention(nn.Module):
    def __init__(self, entity_dim, hidden_dim, heads, dropout):
        super().__init__()
        self.entity_proj = nn.Linear(entity_dim, hidden_dim)
        self.q = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(heads)])
        self.k = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(heads)])
        self.v = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(heads)])
        self.out = nn.Linear(hidden_dim * heads, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    def forward(self, query, entities, mask):
        entity_h = F.relu(self.entity_proj(entities))
        valid = mask.to(query.dtype)
        outputs = []
        for q_layer, k_layer, v_layer in zip(self.q, self.k, self.v):
            q = q_layer(query).unsqueeze(-2)
            scores = torch.sum(q * k_layer(entity_h), dim=-1) / self.hidden_dim**0.5
            scores = scores.masked_fill(~mask, -1e9)
            weights = torch.softmax(scores, dim=-1) * valid
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            outputs.append(torch.sum(weights.unsqueeze(-1) * v_layer(entity_h), dim=-2))
        context = self.dropout(F.relu(self.out(torch.cat(outputs, dim=-1))))
        return context * mask.any(dim=-1, keepdim=True).to(context.dtype)


class PursuitGraphAttentionBlock(nn.Module):
    """3-D UAV relation attention plus target/peer/building entity fusion."""

    def __init__(self, hidden_dim=128, heads=2, dropout=0.0, spatial_scale=25.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spatial_scale = max(float(spatial_scale), 1e-6)
        self.q = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(heads)])
        self.k = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(heads)])
        self.v = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(heads)])
        self.edge_bias = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.peer_out = nn.Linear(hidden_dim * heads, hidden_dim)
        self.target_attention = PursuitEntityAttention(TARGET_DIM, hidden_dim, heads, dropout)
        self.peer_attention = PursuitEntityAttention(PEER_DIM, hidden_dim, heads, dropout)
        self.building_attention = PursuitEntityAttention(BUILDING_DIM, hidden_dim, heads, dropout)
        self.semantic_score = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1, bias=False))
        self.fuse = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def pairwise_distances(uav_xyz):
        return torch.cdist(uav_xyz, uav_xyz)

    def forward(self, h, graph, adjacency):
        xyz = graph["uav_xyz"]
        distance = self.pairwise_distances(xyz)
        delta = xyz.unsqueeze(-2) - xyz.unsqueeze(-3)
        edge = torch.cat([delta / self.spatial_scale, distance.unsqueeze(-1) / self.spatial_scale], dim=-1)
        edge_bias = self.edge_bias(edge).squeeze(-1)
        eye = torch.eye(h.shape[-2], dtype=torch.bool, device=h.device)
        allowed = adjacency.to(torch.bool) | eye
        peer_messages = []
        for q_layer, k_layer, v_layer in zip(self.q, self.k, self.v):
            logits = torch.matmul(q_layer(h), k_layer(h).transpose(-1, -2)) / self.hidden_dim**0.5
            logits = (logits + edge_bias).masked_fill(~allowed, -1e9)
            peer_messages.append(torch.matmul(torch.softmax(logits, dim=-1), v_layer(h)))
        uav_peer = F.relu(self.peer_out(torch.cat(peer_messages, dim=-1)))
        target = self.target_attention(h, graph["target_nodes"], graph["target_mask"])
        peer_entities = self.peer_attention(h, graph["peer_nodes"], graph["peer_mask"])
        buildings = self.building_attention(h, graph["building_nodes"], graph["building_mask"])
        relations = torch.stack([target, F.relu(uav_peer + peer_entities), buildings], dim=-2)
        weights = torch.softmax(self.semantic_score(relations).squeeze(-1), dim=-1).unsqueeze(-1)
        fused = torch.sum(weights * relations, dim=-2)
        update = self.dropout(F.relu(self.fuse(torch.cat([h, fused], dim=-1))))
        active = graph["active_mask"].unsqueeze(-1).to(update.dtype)
        return self.norm(h + update) * active


class PursuitGraphEncoder(nn.Module):
    def __init__(self, hidden_dim=128, heads=2, layers=2, dropout=0.0, spatial_scale=25.0):
        super().__init__()
        self.self_proj = nn.Linear(SELF_DIM, hidden_dim)
        self.blocks = nn.ModuleList([
            PursuitGraphAttentionBlock(hidden_dim, heads, dropout, spatial_scale) for _ in range(layers)
        ])

    def forward(self, obs, adjacency=None, hetero_graph=None):
        if hetero_graph is None:
            raise ValueError("PursuitGraphEncoder requires the pursuit heterogeneous graph")
        squeeze = obs.dim() == 2
        if squeeze:
            adjacency = adjacency.unsqueeze(0)
            hetero_graph = {key: value.unsqueeze(0) for key, value in hetero_graph.items()}
        h = F.relu(self.self_proj(hetero_graph["self_nodes"]))
        for block in self.blocks:
            h = block(h, hetero_graph, adjacency)
        return h.squeeze(0) if squeeze else h


class PursuitGraphActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128, heads=2, layers=2, dropout=0.0, spatial_scale=25.0):
        super().__init__()
        self.obs_dim = int(obs_dim)  # retained for checkpoint diagnostics
        self.encoder = PursuitGraphEncoder(hidden_dim, heads, layers, dropout, spatial_scale)
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs, adjacency=None, hetero_graph=None):
        return self.head(self.encoder(obs, adjacency, hetero_graph))
