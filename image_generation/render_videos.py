# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import print_function
import math, sys, random, argparse, json, os, tempfile
from datetime import datetime as dt
from collections import Counter
from typing import List, Dict, Any
from datetime import datetime
import shutil

from trajectory import (
    Trajectory, RandomizedTrajectoryCreator, Simple2DLinearTrajectory,
    LinearTraj2DVariedTiming, TrajectoryPath, PiecewiseLinearTraj)
from trajectory_utils import trajectories_closer_than
from video_gen_config import VideoGenerationConfig

"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

INSIDE_BLENDER = True
try:
  import bpy, bpy_extras
  from mathutils import Vector
except ImportError as e:
  INSIDE_BLENDER = False
if INSIDE_BLENDER:
  try:
    import utils
  except ImportError as e:
    print("\nERROR")
    print("Running render_images.py from Blender and cannot import utils.py.") 
    print("You may need to add a .pth file to the site-packages of Blender's")
    print("bundled python with a command like this:\n")
    print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
    print("\nWhere $BLENDER is the directory where Blender is installed, and")
    print("$VERSION is your Blender version (such as 2.78).")
    sys.exit(1)

trajectory_creator = RandomizedTrajectoryCreator()
parser = argparse.ArgumentParser()
#sometimes a setting for generating a scene with motion is *too* unfavorable
#and will lead to stack overflow
RECURSION_COUNTER = 0
# Input options
parser.add_argument('--base_scene_blendfile', default='data/base_scene_vids.blend',
    help="Base blender file on which all scenes are based; includes " +
          "ground plane, lights, and camera.")
parser.add_argument('--properties_json', default='data/properties.json',
    help="JSON file defining objects, materials, sizes, and colors. " +
         "The \"colors\" field maps from CLEVR color names to RGB values; " +
         "The \"sizes\" field maps from CLEVR size names to scalars used to " +
         "rescale object models; the \"materials\" and \"shapes\" fields map " +
         "from CLEVR material and shape names to .blend files in the " +
         "--object_material_dir and --shape_dir directories respectively.")
parser.add_argument('--shape_dir', default='data/shapes',
    help="Directory where .blend files for object models are stored")
parser.add_argument('--material_dir', default='data/materials',
    help="Directory where .blend files for materials are stored")
parser.add_argument('--shape_color_combos_json', default=None,
    help="Optional path to a JSON file mapping shape names to a list of " +
         "allowed color names for that shape. This allows rendering images " +
         "for CLEVR-CoGenT.")

# Settings for objects
parser.add_argument('--min_objects', default=3, type=int,
    help="The minimum number of objects to place in each scene")
parser.add_argument('--max_objects', default=10, type=int,
    help="The maximum number of objects to place in each scene")
parser.add_argument(
  '--video_gen_config', 
  default="../video_gen_cfgs/video_gen_cfg.json", type=str,
  help="Path to .json file providing configuration parameters for video "
  "generation.")
parser.add_argument('--min_dist', default=0.25, type=float,
    help="The minimum allowed distance between object centers")
parser.add_argument('--margin', default=0.4, type=float,
    help="Along all cardinal directions (left, right, front, back), all " +
         "objects will be at least this distance apart. This makes resolving " +
         "spatial relationships slightly less ambiguous.")
parser.add_argument('--min_pixels_per_object', default=200, type=int,
    help="All objects will have at least this many visible pixels in the " +
         "final rendered images; this ensures that no objects are fully " +
         "occluded by other objects.")
parser.add_argument('--max_retries', default=20, type=int,
    help="The number of times to try placing an object before giving up and " +
         "re-placing all objects in the scene.")

# Output settings
parser.add_argument('--start_idx', default=0, type=int,
    help="The index at which to start for numbering rendered images. Setting " +
         "this to non-zero values allows you to distribute rendering across " +
         "multiple machines and recombine the results later.")
parser.add_argument('--num_videos', default=5, type=int,
    help="The number of videos to render")
parser.add_argument('--filename_prefix', default='CLEVR',
    help="This prefix will be prepended to the rendered videos.")
parser.add_argument('--split', default='new',
    help="Name of the split for which we are rendering. This will be added to " +
         "the names of rendered images, and will also be stored in the JSON " +
         "scene structure for each image.")
parser.add_argument('--output_video_dir', default='../output/images/',
    help="The directory where output videos will be stored. It will be " +
         "created if it does not exist.")
parser.add_argument('--output_gt_jsonl', default='movement_gt.jsonl',
    help="Filename fora single JSONL file containing all movement ground "
    "truth. It will be placed inside --output_video_dir.")
parser.add_argument('--output_blend_dir', default='output/blendfiles',
    help="The directory where blender scene files will be stored, if the " +
         "user requested that these files be saved using the " +
         "--save_blendfiles flag; in this case it will be created if it does " +
         "not already exist.")
parser.add_argument('--save_blendfiles', type=int, default=0,
    help="Setting --save_blendfiles 1 will cause the blender scene file for " +
         "each generated image to be stored in the directory specified by " +
         "the --output_blend_dir flag. These files are not saved by default " +
         "because they take up ~5-10MB each.")
parser.add_argument('--version', default='1.0',
    help="String to store in the \"version\" field of the generated JSON file")
parser.add_argument('--license',
    default="Creative Commons Attribution (CC-BY 4.0)",
    help="String to store in the \"license\" field of the generated JSON file")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
    help="String to store in the \"date\" field of the generated JSON file; " +
         "defaults to today's date")

# Rendering options
parser.add_argument('--use_gpu', default=0, type=int,
    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
         "to work.")
parser.add_argument('--width', default=480, type=int,
    help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=360, type=int,
    help="The height (in pixels) for the rendered images")
parser.add_argument('--key_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--camera_jitter', default=0.5, type=float,
    help="The magnitude of random jitter to add to the camera position")
parser.add_argument('--render_num_samples', default=128, type=int,
    help="The number of samples to use when rendering. Larger values will " +
         "result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_min_bounces', default=8, type=int,
    help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
    help="The maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=256, type=int,
    help="The tile size to use for rendering. This should not affect the " +
         "quality of the rendered image but may affect the speed; CPU-based " +
         "rendering may achieve better performance using smaller tile sizes " +
         "while larger tile sizes may be optimal for GPU-based rendering.")

def main(args):

  def _speed_calc_func(cfg_dict, value):
    return cfg_dict["max_dist"]/(cfg_dict["video_len_in_secs"]*value)

  video_gen_cfg_json = args.video_gen_config
  # preprocess_cfg_vals_funcs = {}
  # prefixes = [("trajectory_randomization_args", "linear_simple"),
  #             ("trajectory_randomization_args", "linear_varied_timing"),
  #             ("trajectory_randomization_args", "piecewise_linear")]
  # for p in prefixes:
  #   preprocess_cfg_vals_funcs[p + ("min_speed",)] = _speed_calc_func
  #   preprocess_cfg_vals_funcs[p + ("max_speed",)] = _speed_calc_func

  video_gen_cfg = VideoGenerationConfig.from_json_config(
    video_gen_cfg_json)
  random.seed(video_gen_cfg.rng_seed)

  if(os.path.isdir(args.output_video_dir) 
     and len(os.listdir(args.output_video_dir)) > 0):
    args.output_video_dir = (
      args.output_video_dir + datetime.now()
      .strftime("%d_%m_%Y_%H_%M_%s"))
    print("Output directory for videos is non-empty. For safety, the actual "
          "output directory will instead be {0}".format(args.output_video_dir))
    
  if not os.path.isdir(args.output_video_dir):
    os.makedirs(args.output_video_dir)
  with open(
    os.path.join(args.output_video_dir, "generation_config.json"), "w") as f:
    json.dump(video_gen_cfg.to_dict(), f, indent=2)
  if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
    os.makedirs(args.output_blend_dir)

  num_digits = 6
  prefix = '%s_%s_' % (args.filename_prefix, args.split)
  video_template = '%s%%0%dd.mp4' % (prefix, num_digits)
  blend_template = '%s%%0%dd.blend' % (prefix, num_digits)
  video_template = os.path.join(args.output_video_dir, video_template)
  blend_template = os.path.join(args.output_blend_dir, blend_template)    
  
  all_scene_paths = []
  for i in range(args.num_videos):
    status = False
    while(not status):
      video_path = video_template % (i + args.start_idx)
      #scene_template % (i + args.start_idx)
      #all_scene_paths.append(scene_path)
      blend_path = None
      if args.save_blendfiles == 1:
        blend_path = blend_template % (i + args.start_idx)
      num_objects = random.randint(args.min_objects, args.max_objects)
      num_moving_objects = random.randint(
        video_gen_cfg.min_num_moving_objects, 
        min(num_objects, video_gen_cfg.max_num_moving_objects))    
      status = render_scene(
        args,
        video_gen_config=video_gen_cfg,
        num_objects=num_objects,
        num_moving_objects=num_moving_objects,
        output_index=(i + args.start_idx),
        output_split=args.split,
        output_video=video_path,
        output_blendfile=blend_path,
      )

  # After rendering all images, combine the JSON files for each scene into a
  # single JSON file.
  # all_scenes = []
  # for scene_path in all_scene_paths:
  #   with open(scene_path, 'r') as f:
  #     all_scenes.append(json.load(f))
  # output = {
  #   'info': {
  #     'date': args.date,
  #     'version': args.version,
  #     'split': args.split,
  #     'license': args.license,
  #   },
  #   'scenes': all_scenes
  # }
  # with open(args.output_gt_jsonl, 'w') as f:
  #   json.dump(output, f)

def register_trajectory_types(
    possible_directions: List[Vector],
    direction_names: List[str],
    trajectories_to_sample_from: List[str],
    traj_type_to_rand_args: Dict[str, Dict[str, Any]]):
  """Centralized point for associating names with trajectory types.
  The `_make*` functions serve to instantiate the randomizers based on
  keyword arguments provided by `traj_type_to_rand_args` (which can e.g.
  come from some config).
  """
  def _make_linear_traj_randomizer(min_speed, max_speed):

    def _randomize():
      chosen_ind = random.choice(range(len(possible_directions)))
      chosen_direction = possible_directions[chosen_ind]
      chosen_name = direction_names[chosen_ind]
      speed = random.uniform(min_speed, max_speed)
      return Simple2DLinearTrajectory(chosen_direction, chosen_name, speed)

    return _randomize
  
  def _make_linear_traj_varied_timing_randomizer(
      min_speed, max_speed, smallest_start_mult, largest_start_mult,
      smallest_end_mult, largest_end_mult):
    
    def _randomize():
      chosen_ind = random.choice(range(len(possible_directions)))
      chosen_direction = possible_directions[chosen_ind]
      chosen_name = direction_names[chosen_ind]

      speed = random.uniform(min_speed, max_speed)
      start_mult = random.uniform(smallest_start_mult, largest_start_mult)
      end_mult = random.uniform(smallest_end_mult, largest_end_mult)
      return LinearTraj2DVariedTiming(chosen_direction, chosen_name,
                                      speed, start_mult, end_mult)
    return _randomize
  
  def _make_piecewise_linear_randomizer(
      min_num_pieces, max_num_pieces, min_speed, max_speed):
    
    def _randomize():
      num_pieces = random.randint(min_num_pieces, max_num_pieces)
      chosen_inds = [
        random.choice(range(len(possible_directions))) 
        for _ in range(num_pieces)]
      chosen_dirs = [possible_directions[_] for _ in chosen_inds]
      chosen_names = [direction_names[_] for _ in chosen_inds]
      speeds = [random.uniform(min_speed, max_speed) for _ in chosen_dirs]
      initial_mult = random.uniform(0, 0.3)
      increments = [random.uniform(0.1, (1 - 0.3)/num_pieces) for _ in
                    range(num_pieces - 1)]
      start_mults = [initial_mult]
      for i in increments:
        start_mults.append(start_mults[-1] + i)
      end_mults = []
      for i in range(len(start_mults) - 1):
        s1 = start_mults[i]
        s2 = start_mults[i + 1]
        end_mults.append(random.uniform((s1 + s2)/2, s2))
      end_mults.append(random.uniform((1 + start_mults[-1])/2, 1))

      return PiecewiseLinearTraj(chosen_dirs, chosen_names, speeds,
                                 start_mults, end_mults)
    return _randomize
  
  trajectory_creator.register_trajectory_type(
    "linear_simple",  
    _make_linear_traj_randomizer(**traj_type_to_rand_args["linear_simple"]))
  trajectory_creator.register_trajectory_type(
    "linear_varied_timing", _make_linear_traj_varied_timing_randomizer(
      **traj_type_to_rand_args["linear_varied_timing"]))
  trajectory_creator.register_trajectory_type(
    "piecewise_linear", _make_piecewise_linear_randomizer(
      **traj_type_to_rand_args["piecewise_linear"]))
  
  if(trajectories_to_sample_from is not None):
    trajectory_creator.set_trajectory_types_to_sample_from(
      trajectories_to_sample_from)

def render_scene(
    args,
    video_gen_config: VideoGenerationConfig,
    num_objects=5,
    num_moving_objects=0,
    output_index=0,
    output_split='none',
    output_video='render.mp4',
    output_blendfile=None,
  ):

  # Load the main blendfile
  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

  # Load materials
  utils.load_materials(args.material_dir)

  # Set render arguments so we can get pixel coordinates later.
  # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
  # cannot be used.
  render_args = bpy.context.scene.render
  render_args.engine = "CYCLES"
  render_args.filepath = output_video
  render_args.resolution_x = args.width
  render_args.resolution_y = args.height
  render_args.resolution_percentage = 100
  render_args.tile_x = args.render_tile_size
  render_args.tile_y = args.render_tile_size
  if args.use_gpu == 1:
    # Blender changed the API for enabling CUDA at some point
    if bpy.app.version < (2, 78, 0):
      bpy.context.user_preferences.system.compute_device_type = 'CUDA'
      bpy.context.user_preferences.system.compute_device = 'CUDA_0'
    else:
      cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
      cycles_prefs.compute_device_type = 'CUDA'

  # Some CYCLES-specific stuff
  bpy.data.worlds['World'].cycles.sample_as_light = True
  bpy.context.scene.cycles.blur_glossy = 2.0
  bpy.context.scene.cycles.samples = args.render_num_samples
  bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
  bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
  if args.use_gpu == 1:
    bpy.context.scene.cycles.device = 'GPU'

  # This will give ground-truth information about the scene and its objects
  scene_struct = {
      'split': output_split,
      'image_index': output_index,
      'image_filename': os.path.basename(output_video),
      'objects': [],
      'directions': {},
  }

  # Put a plane on the ground so we can compute cardinal directions
  bpy.ops.mesh.primitive_plane_add(radius=5)
  plane = bpy.context.object

  def rand(L):
    return 2.0 * L * (random.random() - 0.5)

  # Add random jitter to camera position
  if args.camera_jitter > 0:
    for i in range(3):
      bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

  # Figure out the left, up, and behind directions along the plane and record
  # them in the scene structure
  camera = bpy.data.objects['Camera']
  plane_normal = plane.data.vertices[0].normal
  cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
  cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
  cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
  plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
  plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
  plane_up = cam_up.project(plane_normal).normalized()

  left_in_img_while_on_plane = (
    cam_left - (cam_left.dot(plane_normal))*plane_normal)
  down_in_img_while_on_plane = (plane_normal.cross(
    left_in_img_while_on_plane))
  up_in_img_while_on_plane = -down_in_img_while_on_plane
  right_in_img_while_on_plane = -left_in_img_while_on_plane

  # Delete the plane; we only used it for normals anyway. The base scene file
  # contains the actual ground plane.
  utils.delete_object(plane)

  # Save all six axis-aligned directions in the scene struct
  scene_struct['directions']['behind'] = tuple(plane_behind)
  scene_struct['directions']['front'] = tuple(-plane_behind)
  scene_struct['directions']['left'] = tuple(plane_left)
  scene_struct['directions']['right'] = tuple(-plane_left)
  scene_struct['directions']['above'] = tuple(plane_up)
  scene_struct['directions']['below'] = tuple(-plane_up)

  # Add random jitter to lamp positions
  if args.key_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
  if args.back_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
  if args.fill_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

  #all 8 compass directions
  movement_dirs = [left_in_img_while_on_plane, 
                   left_in_img_while_on_plane + up_in_img_while_on_plane,
                   up_in_img_while_on_plane,
                   up_in_img_while_on_plane + right_in_img_while_on_plane,
                   right_in_img_while_on_plane,
                   right_in_img_while_on_plane + down_in_img_while_on_plane,
                   down_in_img_while_on_plane,
                   left_in_img_while_on_plane + down_in_img_while_on_plane]
  movement_names = ["left",
                    "up and to the left",
                    "up",
                    "up and to the right",
                    "right",
                    "down and to the right",
                    "down",
                    "down and to the left"]
  
  register_trajectory_types(movement_dirs, movement_names, 
                            video_gen_config.trajectories_to_sample_from,
                            video_gen_config.trajectory_randomization_args)
  bpy.context.scene.render.fps = video_gen_config.fps

  total_frames = math.ceil(
    video_gen_config.video_len_in_secs*video_gen_config.fps)
  # Now make some random objects
  global RECURSION_COUNTER
  RECURSION_COUNTER = 0
  objects, blender_objects = add_random_objects(
    scene_struct, num_objects, num_moving_objects, args, camera,
    video_gen_config.fps, total_frames)
  if(len(objects) != num_objects): #scene generation failed
    return False
  
  assert len(objects) == len(blender_objects)

  #dump ground truth
  movement_gt = compute_movement_ground_truth(objects)
  
  gt_complete = {
    "video_name": render_args.filepath.split("/")[-1],
    "num_objects": num_objects,
    "num_moving_objects": num_moving_objects,
    **movement_gt
  }
  f = open(os.path.join(args.output_video_dir, args.output_gt_jsonl), "a")
  f.write(json.dumps(gt_complete) + "\n")
  f.close()
  assert (
    num_moving_objects == len(movement_gt["moving_objects"])
    and num_objects == num_moving_objects + len(movement_gt["still_objects"])), \
    str(num_moving_objects) + " moving, " + str(num_objects) + " total, gt " + \
    movement_gt["moving_objects"] + movement_gt["still_objects"]  

  bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
  bpy.context.scene.render.ffmpeg.format = 'MPEG4'
  bpy.context.scene.render.ffmpeg.codec = 'H264'

  #Render the scene
  for obj, obj_blender in zip(objects, blender_objects):
    if(obj["path"] is not None):
      p = obj["path"]
      p.insert_in_blender_animations(obj_blender)
      for fcurve in obj_blender.animation_data.action.fcurves:
        if fcurve.data_path == "location":
          for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = 'LINEAR'

  bpy.context.scene.frame_start = 1
  bpy.context.scene.frame_end = total_frames
  bpy.ops.render.render(animation=True)

  if output_blendfile is not None:
    bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)

  return True

def compute_movement_ground_truth(objects):
  moving_objects = []
  still_objects = []
  for o in objects:
    if(o["path"] is None):
      still_objects.append({"color" : o["color"], "shape": o["shape"],
                            "loc": [o["3d_coords"][0], o["3d_coords"][1]]})
    else:
      path = o["path"]
      moving_objects.append({
        "color": o["color"], "shape": o["shape"], 
        "loc": [o["3d_coords"][0], o["3d_coords"][1]],
        "path" : path.properties_as_dict()})
  return {"still_objects": still_objects, "moving_objects": moving_objects}

def add_random_objects(scene_struct, num_objects, num_moving_objects, args, 
                       camera, fps, total_frames):
  """
  Add random objects to the current blender scene. 
  The first `num_moving_objects` to be added will be assigned a 
  randomized trajectory created by the module-global trajectory creator, 
  along which they will later be animated.
  """

  # Load the property file
  global RECURSION_COUNTER
  RECURSION_COUNTER += 1
  if(RECURSION_COUNTER > 10): #tried ten times to generate a scene and failed
    return [], []
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['colors'].items():
      rgba = [float(c) / 255.0 for c in rgb] + [1.0]
      color_name_to_rgba[name] = rgba
    material_mapping = [(v, k) for k, v in properties['materials'].items()]
    object_mapping = [(v, k) for k, v in properties['shapes'].items()]
    size_mapping = list(properties['sizes'].items())

  shape_color_combos = None
  if args.shape_color_combos_json is not None:
    with open(args.shape_color_combos_json, 'r') as f:
      shape_color_combos = list(json.load(f).items())

  positions = []
  objects = []
  blender_objects = []
  moving_object_shape_color_combos = set()
  for i in range(num_objects):
    # Choose a random size
    size_name, r = random.choice(size_mapping)

    # Try to place the object, ensuring that we don't intersect any existing
    # objects and that we are more than the desired margin away from all existing
    # objects along all cardinal directions.
    num_tries = 0
    while True:
      # If we try and fail to place an object too many times, then delete all
      # the objects in the scene and start over.
      num_tries += 1
      if num_tries > args.max_retries:
        for obj in blender_objects:
          utils.delete_object(obj)
        return add_random_objects(
          scene_struct, num_objects, num_moving_objects, args, camera,
          fps, total_frames)

      if(i < num_moving_objects):
        traj = trajectory_creator.create_random_trajectory()
        x = random.uniform(-2, 2) #spawn moving objects closer to center
        y = random.uniform(-2, 2)
        path = traj.compute_trajectory_path(
          Vector((x, y, r)), fps, total_frames) #scale r is used as z-value
      else:
        path = None
        x = random.uniform(-3.5, 3.5)
        y = random.uniform(-3.5, 3.5)
      # Check to make sure the new object is further than min_dist from all
      # other objects, and further than margin along the four cardinal directions
      dists_good = True
      margins_good = True
      for (xx, yy, rr, other_path) in positions:
        
        if(other_path is not None):
          #check for intersection of the two movement paths
          if(path is not None and trajectories_closer_than(path, other_path,
                                                           r + rr)):
            print("Two trajectories would intersect")
            print(path.path_points, other_path.path_points, r + rr)
            # print("I am called ", path.other_info_for_json["direction"],
            #       path.path_points)
            # exit(0)
            dists_good = False
            break
          #create a dummy single-point trajectory to check if object
          #obstructs movements
          dummy_path = Simple2DLinearTrajectory(
            Vector((1, 0, 0)), "dummy", 0).compute_trajectory_path(
              Vector((x, y, r)), fps, total_frames)
          if(trajectories_closer_than(dummy_path, other_path, r + rr)):
            print("Object would intersect movement trajectory")
            dists_good = False
            break

        dx, dy = x - xx, y - yy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist - r - rr < args.min_dist:
          dists_good = False
          break
        for direction_name in ['left', 'right', 'front', 'behind']:
          direction_vec = scene_struct['directions'][direction_name]
          assert direction_vec[2] == 0
          margin = dx * direction_vec[0] + dy * direction_vec[1]
          if 0 < margin < args.margin:
            print(margin, args.margin, direction_name)
            print('BROKEN MARGIN!')
            margins_good = False
            break
        if not margins_good:
          break

      if dists_good and margins_good:
        break

    # Choose random color and shape
    if shape_color_combos is None:
      def pick_obj_and_color():
        return (random.choice(object_mapping), 
                random.choice(list(color_name_to_rgba.items())))
    else:
      def pick_obj_and_color():
        obj_name_out, color_choices = random.choice(shape_color_combos)
        color_name = random.choice(color_choices)
        obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
        rgba = color_name_to_rgba[color_name]
        return ((obj_name, obj_name_out), (color_name, rgba))
    ((obj_name, obj_name_out), (color_name, rgba)) = pick_obj_and_color()
    while((obj_name, color_name) in moving_object_shape_color_combos):
      ((obj_name, obj_name_out), (color_name, rgba)) = pick_obj_and_color()  
    if(traj is not None):
      moving_object_shape_color_combos.add((obj_name, color_name))

    # For cube, adjust the size a bit
    if obj_name == 'Cube':
      r /= math.sqrt(2)

    # Choose random orientation for the object.
    theta = 360.0 * random.random()

    # Actually add the object to the scene
    utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
    obj = bpy.context.object
    blender_objects.append(obj)   
    positions.append((x, y, r, path))

    # Attach a random material
    mat_name, mat_name_out = random.choice(material_mapping)
    utils.add_material(mat_name, Color=rgba)

    # Record data about the object in the scene data structure
    pixel_coords = utils.get_camera_coords(camera, obj.location)
    objects.append({
      'shape': obj_name_out,
      'size': size_name,
      'material': mat_name_out,
      '3d_coords': tuple(obj.location),
      'rotation': theta,
      'pixel_coords': pixel_coords,
      'color': color_name,
      'path': path
    })

  # Check that all objects are at least partially visible in the rendered image
  all_visible = check_visibility(blender_objects, args.min_pixels_per_object)
  if not all_visible:
    # If any of the objects are fully occluded then start over; delete all
    # objects from the scene and place them all again.
    print('Some objects are occluded; replacing objects')
    for obj in blender_objects:
      utils.delete_object(obj)
    return add_random_objects(
      scene_struct, num_objects, num_moving_objects, args, camera, 
      fps, total_frames) 
  
  return objects, blender_objects


def compute_all_relationships(scene_struct, eps=0.2):
  """
  Computes relationships between all pairs of objects in the scene.
  
  Returns a dictionary mapping string relationship names to lists of lists of
  integers, where output[rel][i] gives a list of object indices that have the
  relationship rel with object i. For example if j is in output['left'][i] then
  object j is left of object i.
  """
  all_relationships = {}
  for name, direction_vec in scene_struct['directions'].items():
    if name == 'above' or name == 'below': continue
    all_relationships[name] = []
    for i, obj1 in enumerate(scene_struct['objects']):
      coords1 = obj1['3d_coords']
      related = set()
      for j, obj2 in enumerate(scene_struct['objects']):
        if obj1 == obj2: continue
        coords2 = obj2['3d_coords']
        diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
        dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
        if dot > eps:
          related.add(j)
      all_relationships[name].append(sorted(list(related)))
  return all_relationships


def check_visibility(blender_objects, min_pixels_per_object):
  """
  Check whether all objects in the scene have some minimum number of visible
  pixels; to accomplish this we assign random (but distinct) colors to all
  objects, and render using no lighting or shading or antialiasing; this
  ensures that each object is just a solid uniform color. We can then count
  the number of pixels of each color in the output image to check the visibility
  of each object.

  Returns True if all objects are visible and False otherwise.
  """
  f, path = tempfile.mkstemp(suffix='.png')
  object_colors = render_shadeless(blender_objects, path=path)
  img = bpy.data.images.load(path)
  p = list(img.pixels)
  color_count = Counter((p[i], p[i+1], p[i+2], p[i+3])
                        for i in range(0, len(p), 4))
  os.remove(path)
  if len(color_count) != len(blender_objects) + 1:
    return False
  for _, count in color_count.most_common():
    if count < min_pixels_per_object:
      return False
  return True


def render_shadeless(blender_objects, path='flat.png'):
  """
  Render a version of the scene with shading disabled and unique materials
  assigned to all objects, and return a set of all colors that should be in the
  rendered image. The image itself is written to path. This is used to ensure
  that all objects will be visible in the final rendered scene.
  """
  render_args = bpy.context.scene.render

  # Cache the render args we are about to clobber
  old_filepath = render_args.filepath
  old_engine = render_args.engine
  old_use_antialiasing = render_args.use_antialiasing

  # Override some render settings to have flat shading
  render_args.filepath = path
  render_args.engine = 'BLENDER_RENDER'
  render_args.use_antialiasing = False

  # Move the lights and ground to layer 2 so they don't render
  utils.set_layer(bpy.data.objects['Lamp_Key'], 2)
  utils.set_layer(bpy.data.objects['Lamp_Fill'], 2)
  utils.set_layer(bpy.data.objects['Lamp_Back'], 2)
  utils.set_layer(bpy.data.objects['Ground'], 2)

  # Add random shadeless materials to all objects
  object_colors = set()
  old_materials = []
  for i, obj in enumerate(blender_objects):
    old_materials.append(obj.data.materials[0])
    bpy.ops.material.new()
    mat = bpy.data.materials['Material']
    mat.name = 'Material_%d' % i
    while True:
      r, g, b = [random.random() for _ in range(3)]
      if (r, g, b) not in object_colors: break
    object_colors.add((r, g, b))
    mat.diffuse_color = [r, g, b]
    mat.use_shadeless = True
    obj.data.materials[0] = mat

  # Render the scene
  bpy.ops.render.render(write_still=True)

  # Undo the above; first restore the materials to objects
  for mat, obj in zip(old_materials, blender_objects):
    obj.data.materials[0] = mat

  # Move the lights and ground back to layer 0
  utils.set_layer(bpy.data.objects['Lamp_Key'], 0)
  utils.set_layer(bpy.data.objects['Lamp_Fill'], 0)
  utils.set_layer(bpy.data.objects['Lamp_Back'], 0)
  utils.set_layer(bpy.data.objects['Ground'], 0)

  # Set the render settings back to what they were
  render_args.filepath = old_filepath
  render_args.engine = old_engine
  render_args.use_antialiasing = old_use_antialiasing

  return object_colors


if __name__ == '__main__':
  if INSIDE_BLENDER:
    # Run normally
    argv = utils.extract_args()
    args = parser.parse_args(argv)
    main(args)
  elif '--help' in sys.argv or '-h' in sys.argv:
    parser.print_help()
  else:
    print('This script is intended to be called from blender like this:')
    print()
    print('blender --background --python render_images.py -- [args]')
    print()
    print('You can also run as a standalone python script to view all')
    print('arguments like this:')
    print()
    print('python render_images.py --help')

