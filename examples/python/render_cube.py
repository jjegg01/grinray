#!/usr/bin/env python3

import grinray
import numpy as np
from PIL import Image, ImageOps

# Create a cube with a diffusive material
cube = grinray.Cuboid(2.0,2.0,2.0)
cube_transform = grinray.ObjectTransform((0.0,0.2,-3.0), ((0.0,1.0,0.0), np.pi/4))
simple_material = grinray.LinearGRINFresnelMaterial(1.4, (0.0, 0.1, 0.0), 1.0)
# Create a plane with a test pattern, so we can actually see something
plane_material = grinray.CheckerboardMaterial((1.0,)*3, (1.0,0.0,0.0))
plane = grinray.Plane()
plane_transform = grinray.ObjectTransform((0.0,-1.0,0.0))
# Scene with white background
scene = grinray.Scene()
# Add objects and materials to scene
simple_material_id = scene.add_material(simple_material)
plane_material_id = scene.add_material(plane_material)
scene.add_object(cube, cube_transform, simple_material_id)
scene.add_object(plane, plane_transform, plane_material_id)
# Create a camera with default config (eye at origin looking in negative z direction)
cameraparams = grinray.PerspectiveCameraParams()
cameraparams.pixels = (1024,1024)
camera = grinray.PerspectiveCamera(cameraparams)

# Render scene into 32-bit pixel data
imagedata = camera.render(scene)

# Use PIL to encode data as PNG
im = ImageOps.flip(Image.frombytes("RGBA", (1024,1024), imagedata))
im.save("cube.png")