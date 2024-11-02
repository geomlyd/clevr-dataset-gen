from abc import ABC, abstractmethod

from mathutils import Vector
import random
import numpy as np
import bpy
from math import floor
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
        return TrajectoryPath([start_loc, end_loc], [1, total_frames],
                              {"direction": self.direction_name,
                               "speed": self.speed_units_per_sec})
    
class LinearTraj2DVariedTiming(Trajectory):

    def __init__(self, direction: Vector, direction_name: str,
                 speed_units_per_sec: float, start_multiplier: float,
                 end_multiplier: float):
        """Animates objects along direction `direction`, with speed
        `speed_units_per_sec`. Movement will start at 
        `start_multiplier`\*(the number of total frames that will later
        be provided) and end at `end_multipler`\*(total frames). These
        multipliers should lie in [0, 1] and 
        `start_multiplier <= end_multiplier`.
        """
        self.direction = direction
        self.direction_name = direction_name
        self.speed_units_per_sec = speed_units_per_sec
        if(start_multiplier < 0 or start_multiplier > 1 or end_multiplier < 0
           or end_multiplier > 1):
            raise ValueError("start_ and end_multiplier should be in [0, 1]")
        if(start_multiplier > end_multiplier):
            raise ValueError("start_multiplier should be <= end_multiplier")
        self.start_multiplier = start_multiplier
        self.end_multiplier = end_multiplier

    def _compute_end_loc(self, start_loc: Vector, duration: float):
        return (start_loc 
                + self.speed_units_per_sec*duration*self.direction)

    def compute_trajectory_path(self, start_loc: Vector, fps: int, 
                                total_frames: int) -> TrajectoryPath:
        start_frame = floor(self.start_multiplier*total_frames) + 1
        end_frame = min(floor(self.end_multiplier*total_frames) + 1,
                        total_frames)
        duration = (end_frame - start_frame)/fps
        end_loc = self._compute_end_loc(start_loc, duration)
        
        return TrajectoryPath([start_loc, end_loc], [start_frame, end_frame],
                              {"direction": self.direction_name,
                               "speed": self.speed_units_per_sec})

class PiecewiseLinearTraj(Trajectory):
    #trajectories: List[LinearTraj2DVariedTiming]

    def __init__(self, directions: List[Vector], direction_names: List[str],
                 speeds_units_per_sec: List[float], 
                 piece_start_multipliers: List[float]):
        """Constructs a trajectory that consists of subsequent linear
        paths, the i-th of which is on direction `directions[i]` with
        speed `speeds_units_per_sec[i]`, starts on `piece_start_multipliers[i]`
        and `ends on piece_start_multipliers[i + 1]`, for `i = 0, 1, ..., n-1`,
        where `n` is the number of simple paths and 
        `n = len(directions) = len(direction_names) = len(speed_units_per_sec) 
        = len(piece_start_multipliers) - 1`.

        The multipliers should all be in [0, 1], since they represent
        fractions of the total video length. They should also be given in
        a non-descending order.
        """
        if(len(direction_names) != len(directions) 
           or len(directions) != len(speeds_units_per_sec) 
           or len(directions) != len(piece_start_multipliers) - 1):
            raise ValueError("Mismatch between the number of velocities, "
                             "speeds and number of paths.")
        mults_sorted = all([
            piece_start_multipliers[i] <= piece_start_multipliers[i + 1] 
            for i in range(len(piece_start_multipliers) - 1)])
        all_mults_valid = all(
            [_ >= 0 and _ <= 1 for _ in piece_start_multipliers])
        if(not mults_sorted):
            raise ValueError("The times at which the object passes from "
                             "path endpoints are not in non-descending order.")
        if(not all_mults_valid):
            raise ValueError("One of the time multipliers for the path "
                             "was not in the range [0, 1].")
        
        self.directions = directions
        self.direction_names = direction_names
        self.speeds_units_per_sec = speeds_units_per_sec
        self.piece_start_multipliers = piece_start_multipliers
        self.trajectories = []

        for i, (d, d_name, speed) in enumerate(zip(
            directions, direction_names, speeds_units_per_sec)):
            start = piece_start_multipliers[i]
            end = piece_start_multipliers[i + 1]
            self.trajectories.append(LinearTraj2DVariedTiming(
                d, d_name, speed, start, end))
            
    def compute_trajectory_path(self, start_loc: Vector, fps: int, 
                                total_frames: int) -> TrajectoryPath:
        
        all_points = [start_loc]
        all_frames = []
        start_point = start_loc
        for i, t in enumerate(self.trajectories):
            p = t.compute_trajectory_path(start_point, fps, total_frames)
            all_points.append(p.path_points[1])
            #get both endpoints' frames for the first path only (to know 
            #when the movement starts)
            all_frames += (p.frames_for_path_points if i == 0 else
                              [p.frames_for_path_points[1]])
            start_point = all_points[-1]
        assert (len(all_frames) == len(all_points)
                and len(all_points) == len(self.speeds_units_per_sec) + 1), \
                (all_frames, self.speeds_units_per_sec,
                 len(all_frames), len(self.speeds_units_per_sec))
        #the composite path
        return TrajectoryPath(all_points, all_frames,
                              {"directions": self.direction_names,
                               "speeds": self.speeds_units_per_sec})

class RandomizedTrajectoryCreator:

    def __init__(self):
        self.known_trajectories = {}
        self.arg_generators = {}

    def register_trajectory_type(
            self, traj_name: str, traj_creator: Callable[[Any], Trajectory],
            randomized_args_generator: Callable[[], Tuple[Tuple[Any], Dict]]):
        """`randomized_args_generator` should return a tuple of the form
        (args, kwargs), where args is a tuple and kwargs a dictionary
        that will serve as arguments for `traj_creator`
        """
        self.known_trajectories[traj_name] = traj_creator
        self.arg_generators[traj_name] = randomized_args_generator

    def create_randomized_trajectory(self, traj_name: str) -> Trajectory:
        rand_args = self.arg_generators[traj_name]()
        return self.known_trajectories[traj_name](
            *rand_args[0], **rand_args[1])
    
    def create_random_trajectory(self) -> Trajectory:
        traj_type = random.choice(list(self.known_trajectories.keys()))
        #print("picking ", traj_type, self.known_trajectories[traj_type])
        return self.create_randomized_trajectory(traj_type)