#!/usr/bin/env python3

import grinray
import numpy as np
from PIL import Image, ImageOps

# Create GRIN sphere (refractive index 1.4 at the center and +- 0.1 along the z axis)
sphere_material = grinray.LinearGRINFresnelMaterial(1.4, (0.0, 0.1, 0.0), 1.0)
sphere = grinray.Sphere((0.0,0.0,-2.0), 1.0)
# Create a plane with a test pattern, so we can actually see something
plane_material = grinray.CheckerboardMaterial((1.0,)*3, (0.0,0.0,0.0), (1.0,0.0,0.0))
plane = grinray.Plane((0.0,-1.0,0.0), (0.0,1.0,0.0))
# Scene with white background
scene = grinray.Scene((1.0,1.0,1.0))
# Add objects and materials to scene
sphere_material_id = scene.add_material(sphere_material)
plane_material_id = scene.add_material(plane_material)
scene.add_object(sphere, sphere_material_id)
scene.add_object(plane, plane_material_id)
# Create a camera with default config (eye at origin looking in negative z direction)
cameraparams = grinray.PerspectiveCameraParams()
cameraparams.pixels = (1024,1024)
camera = grinray.PerspectiveCamera(cameraparams)

# Render scene into 32-bit pixel data
imagedata = camera.render(scene)

# Use PIL to encode data as PNG
im = ImageOps.flip(Image.fromarray(imagedata))
im.save("sphere.png")