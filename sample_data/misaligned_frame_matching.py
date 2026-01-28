import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression # todo have had to add this to the movement-env environment

cams = ['side', 'front', 'overhead']

class MatchFrames:
    def __init__(self, filenames: dict, data: dict, fps: int, data_ext: str = '.h5', timestamp_ext: str = '.csv'):
        '''
        Class to fix misalignment between frames of different cameras due to frame drifting (caused by RAM overload).
        Frames in front and overhead cameras are matched to their closest side frame based on their timestamps.
        :param filenames: dictionary for each data source containing: file paths for data ('file');
                            key for the hdf5 files ('key'); and scorer for DLC files ('scorer')
        :param data: dictionary containing loaded data files for each data source
        :param fps: frames per second for experiment
        :param data_ext: data extension (default '.h5')
        :param timestamp_ext: timestamp extension (default '.csv')
        '''
        self.filenames = filenames
        self.data = data
        self.fps = fps
        self.data_ext = data_ext
        self.timestamp_ext = timestamp_ext

        self.timestamps = {}
        self.frame_duration_nanoseconds : float = 0.0

    def align_dfs(self):
        matched_frames = self.match_frames()

        # find first index row where all frames are in positive time
        start = None
        for idx, it in enumerate(matched_frames):
            if np.all(np.array(it) > 0):
                start = idx
                break
        matched_frames = matched_frames[start:]

        frames = {'side': [], 'front': [], 'overhead': []}
        aligned_dfs = {'side': [], 'front': [], 'overhead': []}
        for vidx, view in enumerate(cams):
            frames[view] = [frame[vidx] for frame in matched_frames]
            aligned_dfs[view] = self.data[view].iloc[frames[view]].reset_index(drop=False).rename(columns={'index': 'original_index'})

        return aligned_dfs

    def match_frames(self):
        timestamps = self.adjust_frames()
        buffer_ns : int = int(self.frame_duration_nanoseconds)

        # Ensure the timestamps are sorted
        dfs : dict = dict(keys=cams)
        for view in cams:
            timestamps[view] = timestamps[view].sort_values().reset_index(drop=True)
            dfs[view] = pd.DataFrame(
                {'Timestamp': timestamps[view], 'Frame_number_%s' % view: range(len(timestamps[view]))})

        # Perform asof merge to find the closest matching frames within the buffer
        matched_front = pd.merge_asof(dfs['side'], dfs['front'], on='Timestamp', direction='nearest',
                                      tolerance=buffer_ns,
                                      suffixes=('_side', '_front'))
        matched_all = pd.merge_asof(matched_front, dfs['overhead'], on='Timestamp', direction='nearest',
                                    tolerance=buffer_ns, suffixes=('_side', '_overhead'))

        # Handle NaNs explicitly by setting unmatched frames to -1
        matched_frames = matched_all[['Frame_number_side', 'Frame_number_front', 'Frame_number_overhead']].map(
            lambda x: int(x) if pd.notnull(x) else -1).values.tolist()

        return matched_frames

    def adjust_frames(self):
        timestamps: dict = dict(keys=cams)
        timestamps_adj: dict = dict(keys=cams)
        for view in cams:
            timestamps[view] = self.zero_timestamps(self.load_timestamps(view))
            if view != 'side':
                timestamps_adj[view] = self.adjust_timestamps(timestamps['side'], timestamps[view])
        timestamps_adj['side'] = timestamps['side'].loc(axis=1)['Timestamp'].astype(float)
        return timestamps_adj

    def load_timestamps(self, view):
        # Generate timestamp path (based on DLC file naming convention + my timestamp file naming convention)
        path = self.filenames[view]['file']
        scorer = self.filenames[view]['scorer']
        timestamp_path = Path(str(path).replace(scorer, '_Timestamps').replace(self.data_ext, self.timestamp_ext))
        timestamps = pd.read_csv(timestamp_path)
        return timestamps

    @staticmethod
    def zero_timestamps(timestamps):
        timestamps.loc(axis=1)['Timestamp'] = timestamps.loc(axis=1)['Timestamp'] - timestamps.loc(axis=1)['Timestamp'][0]
        return timestamps

    def adjust_timestamps(self, ref_timestamps, timestamps):
        self.frame_duration_nanoseconds = (1/self.fps) * 1e9
        mask = timestamps.loc(axis=1)['Timestamp'].diff() <= self.frame_duration_nanoseconds
        timestamps_single_frame = timestamps[mask]
        ref_timestamps_single_frame : pd.DataFrame = ref_timestamps[mask]
        diff = timestamps_single_frame.loc(axis=1)['Timestamp'] - ref_timestamps_single_frame.loc(axis=1)['Timestamp']

        # find the best fit line for the lower half of the data by straightning the line
        model = LinearRegression().fit(ref_timestamps_single_frame.loc(axis=1)['Timestamp'].values.reshape(-1, 1), diff.values)
        slope = model.coef_[0]
        intercept = model.intercept_
        straightened_diff = diff - (slope * ref_timestamps_single_frame.loc(axis=1)['Timestamp'] + intercept)
        correct_diff_idx = np.where(straightened_diff < straightened_diff.mean())

        model_true = LinearRegression().fit(ref_timestamps_single_frame.loc(axis=1)['Timestamp'].values[correct_diff_idx].reshape(-1, 1), diff.values[correct_diff_idx])
        slope_true = model_true.coef_[0]
        intercept_true = model_true.intercept_
        adjusted_timestamps = timestamps.loc(axis=1)['Timestamp'] - (slope_true * timestamps.loc(axis=1)['Timestamp'] + intercept_true)
        return adjusted_timestamps

