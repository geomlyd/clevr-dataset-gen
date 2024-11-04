from typing import Dict, Any, Callable, Tuple, Union, List
import json
import bpy

class VideoGenerationConfig():

    def __init__(self, video_len_in_secs: float, fps: int,
                 min_num_moving_objects: int, max_num_moving_objects: int,
                 max_dist: int,
                 trajectory_randomization_args: Dict[str, Any],
                 trajectories_to_sample_from: List[str]=None):
        self.video_len_in_secs = video_len_in_secs
        self.max_dist = max_dist
        self.fps = fps
        self.fps = bpy.context.scene.render.fps
        self.min_num_moving_objects = min_num_moving_objects
        self.max_num_moving_objects = max_num_moving_objects
        self.trajectory_randomization_args = trajectory_randomization_args
        self.trajectories_to_sample_from = trajectories_to_sample_from

    @classmethod
    def from_config_KVs(
        cls, 
        config_kv_dict: Dict[str, Any], 
        preprocess_config_vals_funcs: Dict[
            Union[str, Tuple], Callable[[Dict[str, Any], Any], Any]]={}):
        """preprocess_config_vals_funcs[k] is a function that will be
        called as f(config_kv_dict, v), where v is the value for this
        config field, and should yield the actual value for this field
        of the config. 
        
        This is useful for config values that depend on
        others (e.g. speed calculated via distance and time). If the
        field is "nested", provide the corresponding keys as a tuple (e.g. 
        ("trajectory_randomization_args", "some_trajectory", "some_field")
        to access trajectory_randomization_args[some_trajectory[some_field]]).
        """
        for k, func in preprocess_config_vals_funcs.items():
            if(type(k) == str):
                config_kv_dict[k] = func[k](config_kv_dict)
            elif(type(k) == tuple):
                start_from = config_kv_dict
                for subkey in k[:-1]:
                    start_from = start_from[subkey]
                start_from[k[-1]] = func(config_kv_dict, start_from[k[-1]])

        return VideoGenerationConfig(**config_kv_dict)
    
    @classmethod
    def from_json_config(
        cls, 
        filepath: str,
        preprocess_config_vals_funcs: Dict[str, Callable]={}):
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return cls.from_config_KVs(config_dict, preprocess_config_vals_funcs)