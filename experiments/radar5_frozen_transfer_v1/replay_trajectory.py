import os
import os
import sys
import json
from pathlib import Path

sys.argv.extend([
    "--portable-root",
    "/srv/data/projects/kyy/omnidrones/.kit_portable",
])

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {"headless": True, "anti_aliasing": 1},
    experience=os.path.join(
        os.environ["EXP_PATH"],
        "omni.isaac.sim.python.kit",
    ),
)

from pxr import Gf, UsdGeom
import omni.usd
import omni.replicator.core as rep
from omni.isaac.core.utils.prims import create_prim
from PIL import Image

trajectory_path = Path(
    "runs/radar5_single_scene_seed7000000/trajectory.json"
)
frame_dir = Path(
    "runs/radar5_single_scene_seed7000000/replay_frames"
)
frame_dir.mkdir(parents=True, exist_ok=True)

with trajectory_path.open() as handle:
    data = json.load(handle)

scene = data["scene"]
frames = data["frames"]
stage = omni.usd.get_context().get_stage()

ground = UsdGeom.Cube.Define(stage, "/World/replay_ground")
ground.AddScaleOp().Set(
    Gf.Vec3f(scene["grid_size"] / 2, scene["grid_size"] / 2, 0.05)
)
ground.AddTranslateOp().Set(
    Gf.Vec3d(scene["grid_size"] / 2, scene["grid_size"] / 2, -0.05)
)
ground.CreateDisplayColorAttr([(0.12, 0.15, 0.18)])

for index, (rect, height) in enumerate(
    zip(scene["buildings"], scene["building_heights"])
):
    x0, y0, x1, y1 = rect
    building = UsdGeom.Cube.Define(
        stage, f"/World/building_{index}"
    )
    building.AddScaleOp().Set(
        Gf.Vec3f((x1 - x0) / 2, (y1 - y0) / 2, height / 2)
    )
    building.AddTranslateOp().Set(
        Gf.Vec3d((x0 + x1) / 2, (y0 + y1) / 2, height / 2)
    )
    building.CreateDisplayColorAttr([(0.35, 0.38, 0.42)])
    building.CreateDisplayOpacityAttr([0.28])

# USD transform order: translate first, then scale.
for prim in [ground] + [
    UsdGeom.Cube.Get(stage, f"/World/building_{i}")
    for i in range(len(scene["buildings"]))
]:
    ops = prim.GetOrderedXformOps()
    translate = next(op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate)
    scale = next(op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeScale)
    prim.SetXformOpOrder([translate, scale])

hummingbird_usd = os.environ["HUMMINGBIRD_USD"]

def _transform_ops(prim):
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    translate = next(
        (op for op in ops
         if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
        None,
    )
    orient = next(
        (op for op in ops
         if op.GetOpType() == UsdGeom.XformOp.TypeOrient),
        None,
    )
    if translate is None:
        translate = xform.AddTranslateOp()
    if orient is None:
        orient = xform.AddOrientOp()
    return translate, orient

pursuer_ops = []
pursuer_orient_ops = []

for index in range(3):
    prim = create_prim(
        f"/World/pursuer_{index}",
        usd_path=hummingbird_usd,
    )
    translate, orient = _transform_ops(prim)
    pursuer_ops.append(translate)
    pursuer_orient_ops.append(orient)

evader = create_prim(
    "/World/evader_replay",
    usd_path=hummingbird_usd,
)
evader_op, evader_orient_op = _transform_ops(evader)

camera = rep.create.camera(
    position=(14.0, -8.0, 18.0),
    look_at=(8.0, 16.0, 3.0),
)

rep.create.light(
    light_type="distant",
    rotation=(-45, 0, 35),
    intensity=12000,
)
rep.create.light(
    light_type="sphere",
    position=(8.0, 4.0, 14.0),
    intensity=50000,
    scale=5.0,
)

render_product = rep.create.render_product(
    camera, (960, 720)
)
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([render_product])

for frame in frames:
    for index, (xyz, quat) in enumerate(
        zip(frame["pursuers_xyz"], frame["pursuers_quat"])
    ):
        pursuer_ops[index].Set(Gf.Vec3d(*xyz))
        pursuer_orient_ops[index].Set(
            Gf.Quatd(float(quat[0]), Gf.Vec3d(*map(float, quat[1:])))
        )

    evader_op.Set(Gf.Vec3d(*frame["target_xyz"]))
    evader_orient_op.Set(Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0)))
    rep.orchestrator.step(rt_subframes=2)

    image = rgb.get_data()
    output = frame_dir / (
        f"frame_{int(frame['decision']) + 1:04d}.png"
    )
    Image.fromarray(image[:, :, :3]).save(output)

print("REPLAY_OK", len(frames), "frames", flush=True)
simulation_app.close()
