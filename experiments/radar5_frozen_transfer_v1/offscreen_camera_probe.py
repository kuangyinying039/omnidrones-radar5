import os
import sys
import numpy as np

sys.argv.extend([
    "--portable-root",
    "/srv/data/projects/kyy/omnidrones/.kit_portable",
])

from omni.isaac.kit import SimulationApp

app = SimulationApp(
    {"headless": True, "anti_aliasing": 1},
    experience=os.path.join(
        os.environ["EXP_PATH"],
        "omni.isaac.sim.python.kit",
    ),
)

import omni.replicator.core as rep
from PIL import Image

with rep.new_layer():
    rep.create.sphere(
        scale=0.5,
        position=(0.0, 0.0, 1.0),
        semantics=[("class", "test_sphere")],
    )

    camera = rep.create.camera(
        position=(4.0, -4.0, 3.0),
        look_at=(0.0, 0.0, 1.0),
    )

    light = rep.create.light(
        light_type="sphere",
        position=(2.0, -2.0, 6.0),
        intensity=50000,
        scale=5.0,
    )

    render_product = rep.create.render_product(
        camera,
        (640, 480),
    )

rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([render_product])

rep.orchestrator.step(rt_subframes=4)
data = rgb.get_data()

Image.fromarray(data[:, :, :3]).save(
    "experiments/radar5_frozen_transfer_v1/offscreen_probe.png"
)

print("REPLICATOR_CAMERA_OK")
print("image_shape:", data.shape)

app.close()
