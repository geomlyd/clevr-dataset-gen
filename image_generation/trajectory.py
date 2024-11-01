from abc import ABC, abstractmethod

from mathutils import Vector
import random
import numpy as np
import bpy
from math import ceil
import json
from typing import List, Tuple, Callable, Any, Dict

class TrajectoryPath:
    
    def __init__(self, path_points: List[Tuple[Vector]],
                 frames_for_path_points: List[int], 
                 other_info_for_json: Dict[str, Any]):
        if(len(frames_for_path_points) != len(path_points)):
            raise ValueError("A frame must be provided for each vertex of the "
                             "trajectory path.")
        self.frames_for_path_points = frames_for_path_points
        self.path_points = path_points
        self.other_info_for_json = other_info_for_json

    def insert_in_blender_animations(self, blender_obj):
        original_loc = blender_obj.location
        for (f, p) in zip(self.frames_for_path_points, self.path_points):
            blender_obj.location = p
            blender_obj.keyframe_insert(data_path="location", frame=f)
        blender_obj.location = original_loc

    def properties_as_dict(self) -> str:
        as_dict = {**self.other_info_for_json,
                     "path_points": [[p[0], p[1]] for p in self.path_points],
                     "frames_for_path_points": self.frames_for_path_points}
        return as_dict
    
    def get_path_as_segments(self) -> List[Tuple[Tuple[float, float, float], 
                                                 Tuple[float, float, float]]]:
        ret = []
        ret = [
            (
            (self.path_points[i][0], self.path_points[i][1], 
             self.path_points[i][2]),
            (self.path_points[i + 1][0], self.path_points[i + 1][1],
             self.path_points[i + 1][2])
            ) for i in range(len(self.path_points) - 1)]
        return ret

class Trajectory(ABC):

    @abstractmethod
    def compute_trajectory_path(self, start_loc: Vector, fps: int,
                                total_frames: int) -> TrajectoryPath:
        pass

class Simple2DLinearTrajectory(Trajectory):

    def __init__(self, direction: Vector, direction_name: str, 
                 speed_units_per_sec: float):
        self.direction = direction.normalized()
        self.speed_units_per_sec = speed_units_per_sec
        self.direction_name = direction_name

    def _compute_end_loc(self, start_loc: Vector, duration: float):
        return (start_loc 
                + self.speed_units_per_sec*duration*self.direction)        

    def compute_trajectory_path(self, start_loc: Vector, fps: int, 
                                total_frames: int) -> TrajectoryPath:
        duration = total_frames/fps
        end_loc = self._compute_end_loc(start_loc, duration)
        print("I am ", self.direction_name, ". Start ", start_loc, "End", end_loc)
        return TrajectoryPath([start_loc, end_loc], [1, total_frames],
                              {"direction": self.direction_name,
                               "speed": self.speed_units_per_sec})

class RandomizedTrajectoryCreator:

    def __init__(self):
        self.known_trajectories = {}
        self.arg_generators = {}

    def register_trajectory_type(
            self, traj_name: str, traj_creator: Callable[[Any], Trajectory],
            randomized_args_generator: Callable[[], Any]):
        self.known_trajectories[traj_name] = traj_creator
        self.arg_generators[traj_name] = randomized_args_generator

    def create_randomized_trajectory(self, traj_name: str) -> Trajectory:
        return self.known_trajectories[traj_name](
            self.arg_generators[traj_name]())
    
    def create_random_trajectory(self) -> Trajectory:
        traj_type = random.choice(list(self.known_trajectories.keys()))
        #print("picking ", traj_type, self.known_trajectories[traj_type])
        return self.create_randomized_trajectory(traj_type)