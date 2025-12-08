import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from AssettoCorsaEnv.track_occupancy_grid import TrackOccupancyGrid
import logging

logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------
from numba import jit
from numba.np.extensions import cross2d

@jit(nopython=True)
def in_quadrilateral(boxes, points):
    (A, D, B, C) = np.split(boxes, 4, axis=1)
    M = points[None, ...]
    AM = (M - A); CM = (M - C)
    AB = (B - A); AD = (D - A)
    CB = (B - C); CD = (D - C)
    area = (np.abs(cross2d(AM, AB)) + np.abs(cross2d(AM, AD)) +
            np.abs(cross2d(CM, CB)) + np.abs(cross2d(CM, CD)))
    total_area = np.abs(cross2d(AB[0], AD[0])) + np.abs(cross2d(CB[0], CD[0]))
    in_box = area <= total_area + 1e-9

    def any_in_box(in_box):
        result = np.zeros(in_box.shape[1], dtype=np.bool_)
        for i in range(in_box.shape[1]):
            for j in range(in_box.shape[0]):
                if in_box[j, i]:
                    result[i] = True
                    break
        return result
    return any_in_box(in_box)

def get_yaw(x, y):
    deriv_x, deriv_y = np.diff(x), np.diff(y)
    return np.arctan2(deriv_y, deriv_x)

# -----------------------------------------------------------
# Track Class (Modified)
# -----------------------------------------------------------
class Track:
    def __init__(self, track_file_path, track_grid_file=None, torch_device=None, downsample_segments=10):
        self.file_path = track_file_path
        self.track_grid_file = track_grid_file
        self.downsample_segments = downsample_segments
        
        if track_grid_file is not None:
            self.track_occupancy_grid = TrackOccupancyGrid(track_grid_file=self.track_grid_file, torch_device=torch_device)

        # Load track data
        self.track = pd.read_csv(track_file_path)

        # 1. Border Line Downsampling
        self.left_border_x = self.custom_downsample(self.track.left_border_x.values)
        self.left_border_y = self.custom_downsample(self.track.left_border_y.values)
        self.right_border_x = self.custom_downsample(self.track.right_border_x.values)
        self.right_border_y = self.custom_downsample(self.track.right_border_y.values)

        if 'middle_x' in self.track.columns:
            logger.info("Using 'middle_x' column as reference line.")
            raw_mx = self.track.middle_x.values
            raw_my = self.track.middle_y.values
        elif 'pos_x' in self.track.columns:
            logger.info("Using 'pos_x' column (Racing Line) as reference line.")
            raw_mx = self.track.pos_x.values
            raw_my = self.track.pos_y.values
        else:
            logger.warning("'middle_x' or 'pos_x' not found. Calculating geometric center.")
            raw_mx = (self.track.left_border_x.values + self.track.right_border_x.values) / 2.0
            raw_my = (self.track.left_border_y.values + self.track.right_border_y.values) / 2.0

        # Downsample Reference Line
        self.middle_x = self.custom_downsample(raw_mx)
        self.middle_y = self.custom_downsample(raw_my)

        # 3. Stack Points for KDTree-like search
        self.stack_middle = np.vstack([self.middle_x, self.middle_y]).T 
        
        # 4. Heading & Curvature Calculation (Softmin Reward용)
        dx = np.gradient(self.middle_x)
        dy = np.gradient(self.middle_y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        # Heading (Yaw)
        self.heading = np.arctan2(dy, dx)
        
        # Curvature: (x'y'' - y'x'') / (x'^2 + y'^2)^1.5
        self.curvature = (dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5 + 1e-8)

        self.num_segments = self.right_border_x.shape[0]
        self.stack_rb = np.vstack([self.right_border_x, self.right_border_y]).T 
        self.stack_lb = np.vstack([self.left_border_x, self.left_border_y]).T   
        self.lr_track = np.stack([self.stack_rb, self.stack_lb], axis=1).reshape(-1,2)

        logger.info(f"Track loaded from: {track_file_path} | Segments: {self.num_segments} | Features: Heading/Curvature Ready")

    def custom_downsample(self, array):
        downsampled = array[::self.downsample_segments]
        if downsampled[-1] != array[-1]:
            downsampled = np.append(downsampled, array[-1])
        return downsampled

    def is_point_in_segment(self, point, segment_idx):
        corners = np.array( [self.lr_track[0 + segment_idx * 2 : 4 + segment_idx * 2]] )
        found = in_quadrilateral(corners, point)
        return found[0]

    def get_segment_id(self, point):
        for segment_idx in range(self.num_segments - 1):
            if self.is_point_in_segment(point, segment_idx):
                return segment_idx
        return -1

    def is_point_in_segments_range(self, point, start_segment, end_segment):
        def wrap_index(i):
            return i % (self.num_segments - 1)
        i = wrap_index(start_segment)
        end_segment = wrap_index(end_segment)
        while True:
            i += 1
            i = wrap_index(i)
            if self.is_point_in_segment(point, i):
                return i
            if i == end_segment:
                break
        return -1

    def closest_node(self, point):
        '''
        get closest point of the middle of the track to a point
        '''
        nodes = self.stack_middle
        # point shape check (ensure broadcasting works)
        if point.shape != (2,):
             point = point.flatten()
        
        dist_2 = np.sum((nodes - point)**2, axis=1)
        return np.argmin(dist_2)

    def export(self, file):
        self.track.to_csv(file, index=None)