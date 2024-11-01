from typing import Tuple
import numpy as np
from trajectory import TrajectoryPath

def point_to_segment_distance(p: np.ndarray,
                              seg: Tuple[np.ndarray, np.ndarray]):
    seg_start = seg[0]
    seg_end = seg[1]
    len_seg_sq = np.linalg.norm(seg_end - seg_start)**2
    if(np.abs(len_seg_sq) < 1e-5):
        return np.linalg.norm(seg_start - p)
    t = np.dot((p - seg_start), seg_end - seg_start)/len_seg_sq
    t = max(0, min(1, t))
    closest_point = seg_start + t*(seg_end - seg_start)
    dist = np.linalg.norm(p - closest_point)
    return dist

def smallest_distance_of_segments(
        s1: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        s2: Tuple[Tuple[float, float, float], Tuple[float, float, float]]):
    
    s1_start = np.array(s1[0])
    s1_end = np.array(s1[1])
    s2_start = np.array(s2[0])
    s2_end = np.array(s2[1])
    s1 = s1_end - s1_start #s1, s2: segments as vectors
    s2 = s2_end - s2_start
    len_s1 = np.linalg.norm(s1)
    len_s2 = np.linalg.norm(s2)
    s1_is_point = len_s1 < 1e-5 
    if(s1_is_point):
        return point_to_segment_distance(s1_start, [s2_start, s2_end])
    s2_is_point = len_s2 < 1e-5 
    if(s2_is_point):
        return point_to_segment_distance(s2_start, [s1_start, s1_end])    
    s1_normalized = s1/len_s1
    s2_normalized = s2/len_s2

    cross_A_B = np.cross(s1_normalized, s2_normalized)
    denom = np.linalg.norm(cross_A_B)**2

    if(denom < 1e-5): #segments are parallel
        d1 = np.dot(s1_normalized, s2_start - s1_start)
        d2 = np.dot(s1_normalized, s2_end - s1_start)

        #two symmetric cases for one of the segments being "before" the other
        if(d1 <= 0 and d2 <= 0):
            #inner ifs: check which of the endpoints are closer
            if(np.abs(d1) < np.abs(d2)): 
                return np.linalg.norm(s1_start - s2_start)
            return np.linalg.norm(s1_start - s2_end)
        elif(d1 >= len_s1 and d2 >= len_s1):
            if(np.abs(d1) < np.abs(d2)):
                return np.linalg.norm(s1_end - s2_start)
            return np.linalg.norm(s1_end - s2_end)
        #there is an "overlap", distance is the distance of the two lines
        return np.linalg.norm(d1*s1_normalized + s1_start - s2_start)
    else:
        #a way of determining whether the closest point of the *lines* is
        #within the segments
        t = s2_start - s1_start
        detA = np.linalg.det([t, s2_normalized, cross_A_B])
        detB = np.linalg.det([t, s1_normalized, cross_A_B])
        t1 = detA/denom
        t2 = detB/denom
        proj_A = s1_start + (s1_normalized*t1)
        proj_B = s2_start + (s2_normalized*t2)

        #calculate the closest points, which in some cases must be 
        #the endpoints
        if(t1 < 0):
            proj_A = s1_start
        elif(t1 > len_s1):
            proj_A = s1_end
        if(t2 < 0):
            proj_B = s2_start
        elif(t2 > len_s2):
            proj_B = s2_end
        if(t1 < 0 or t1 > len_s1):
            dot = np.dot(s2_normalized, proj_A - s2_start)
            if(dot < 0):
                dot = 0
            elif(dot > len_s2):
                dot = len_s2
            proj_B = s2_start + (s2_normalized*dot)
        if(t2 < 0 or t2 > len_s2):
            dot = np.dot(s1_normalized, proj_B - s1_start)
            if(dot < 0):
                dot = 0
            elif(dot > len_s1):
                dot = len_s1
            proj_A = s1_start + (s1_normalized*dot)

        return np.linalg.norm(proj_A - proj_B)

def trajectories_closer_than(t1: TrajectoryPath, t2: TrajectoryPath, 
                             dist: float) -> bool:
    segments1 = t1.get_path_as_segments()
    segments2 = t2.get_path_as_segments()

    for s1 in segments1:
        for s2 in segments2:
            d = smallest_distance_of_segments(s1, s2)
            #print(s1, s2, d)
            if(d < dist):
                return True
    return False
