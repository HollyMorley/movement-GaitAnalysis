from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm

import sample_data.misaligned_frame_matching as mfm

'''
### My edited data files are not formatted in a compatible way for the movement package.
- I have no confidence values
- Low confidence coordinates are instead marked as NaNs (insufficient single cam data was available for triangulation)
- Coordinates are from non-continuous tracking segments (separate runs), i.e. with frames missing between runs

### Solution:
- Replace confidence values with median of the original confidence values (i.e. given either 2 or 3 cams were 
    used for triangulation (pcuotff == 0.9) this should return a median confidence val > pcutoff for data points 
    where triangulation was performed and where I therefore have a coordinate value)
- Replace NaN coordinates with (0,0,0), replicating a "not tracked" state
- Snip and stitch runs together (this is actually already done, I'm just resetting the indices) to create a 
    continuous tracking sequence
- Snip and stitch video frames together to create a continuous video sequence matching the tracking data
    - Note I had some frame drifting across cams due to RAM overload so have incorporated my code to 'match' frames
        together based on timestamps (see sample_data/misaligned_frame_matching.py)  

'''
@dataclass
class Config:
    cams: List[str]
    fps: int
    data_ext: str
    timestamp_ext: str
    new_scorer: str

CONFIG = Config(
    cams=['side', 'front', 'overhead'],
    fps=247,
    data_ext='.h5',
    timestamp_ext='.csv',
    new_scorer= 'DLC_scorer'
)

BASE_DIR = Path(r"X:\hmorley\movement\sample_data\source_data")
BASE_NAME = "HM_20230316_APACharExt_FAA-1035243_None"
OUTPUT_NAME = "DLC_single-mouse_DBTravelator"

class SaveSampleData:
    def __init__(
            self,
            base_dir: Path,
            base_name: str,
            output_name: str,
            config: Config | None = None
    ):
        self.base_dir = base_dir
        self.base_name = base_name
        self.output_name = output_name

        config = config or CONFIG
        self.cams = config.cams
        self.fps = config.fps
        self.data_ext = config.data_ext
        self.timestamp_ext = config.timestamp_ext
        self.new_scorer = config.new_scorer

    def create_camera_config(self, view, scorer='', key='df_with_missing') -> dict:
        """Create camera configuration dictionary."""
        if view == 'realworld':
            filename = f"{self.base_name}_mapped3D_Runs.h5"
            key = "real_world_coords_runs"
        else:
            filename = f"{self.base_name}_{view}_1{scorer}.h5"

        return {
            'file': self.base_dir / filename,
            'key': key,
            'scorer': scorer
        }

    def get_files_and_paths(self) -> tuple[dict, dict]:
        # Load data for Day1 of LowHigh experiments

        sample_filenames = {
            'realworld': self.create_camera_config(view='realworld'),
            self.cams[0]: self.create_camera_config(view=self.cams[0], scorer='DLC_resnet50_DLC_DualBeltAug2shuffle1_1200000'),
            self.cams[1]: self.create_camera_config(view=self.cams[1], scorer='DLC_resnet50_DLC_DualBeltAug3shuffle1_1000000'),
            self.cams[2]: self.create_camera_config(view=self.cams[2], scorer='DLC_resnet50_DLC_DualBeltAug3shuffle1_1000000')
        }
        original_data = {
            name: pd.read_hdf(info['file'], info['key']) for
            name, info in sample_filenames.items()
        }
        return sample_filenames, original_data

    @staticmethod
    def select_runs(data) -> pd.DataFrame:
        runs_to_drop = [0,1]
        data.drop(runs_to_drop, axis=0, level='Run', errors='ignore', inplace=True)
        return data

    @staticmethod
    def select_columns(data) -> pd.DataFrame:
        # Trim realworld data to tracked points only (drop extra columns with metadata)
        columns_to_drop = [col for col in data.columns if
                           not any(coord in col for coord in ['x', 'y', 'z'])]
        data.drop(columns=columns_to_drop, inplace=True)
        return data

    @staticmethod
    def fill_nas(data) -> pd.DataFrame:
        data = data.fillna(0)
        return data

    def tidy_3d_datafile(self, data) -> pd.DataFrame:
        data_runs = self.select_runs(data)
        data_columns = self.select_columns(data_runs)
        data_filled = self.fill_nas(data_columns)

        return data_filled

    @staticmethod
    def get_indexes(data) -> np.ndarray:
        index = np.array(data.index)
        return index

    def find_median_likelihoods(self, data, filenames) -> pd.DataFrame:
        # find median likelihood between 3 cams
        cam_likelihoods: dict = dict(keys=self.cams)
        for cam in self.cams:
            cam_data_trim = data[cam].loc(axis=1)[filenames[cam]['scorer']]
            likelihood_column_names = [col for col in cam_data_trim if 'likelihood' in col]
            cam_likelihoods[cam] = cam_data_trim.loc(axis=1)[likelihood_column_names]

        median_likelihoods = pd.DataFrame(np.median([cam_likelihoods[cam].values for cam in self.cams], axis=0),
                                          index=cam_likelihoods[self.cams[0]].index,
                                          columns=cam_likelihoods[self.cams[0]].columns)
        return median_likelihoods

    def trim_likelihoods(self, data, likelihoods) -> pd.DataFrame:
        # trim rows/frames from median likelihoods to match realworld data
        data_framesidxs = data.index.get_level_values(level='FrameIdx')
        trimmed_likelihoods = likelihoods.loc(axis=0)[data_framesidxs]
        assert len(trimmed_likelihoods) == len(data_framesidxs)

        return trimmed_likelihoods

    def drop_non_frame_indices(self, data) -> pd.DataFrame:
        data_framesonly = data.droplevel(level=['Run', 'RunStage'], axis=0)
        return data_framesonly

    def add_in_likelihoods(self, data, likelihoods) -> pd.DataFrame:
        new_likelihood_columns = []
        for bodypart in likelihoods.columns.get_level_values(level='bodyparts').unique():
            new_likelihood_columns.append((bodypart, 'likelihood'))

        # Rename trimmed_likelihoods columns to match this structure
        likelihoods.columns = pd.MultiIndex.from_tuples(new_likelihood_columns)

        # Combine the dataframes
        new_data = pd.concat([likelihoods, data], axis=1, join='inner')

        return new_data

    def reorganise_3d_data_to_2d_format(self, data) -> pd.DataFrame:
        # Reorder columns so each bodypart has x, y, z, likelihood together
        bodyparts = data.columns.get_level_values(level='bodyparts').unique()
        ordered_columns = []
        for bodypart in bodyparts:
            for coord in ['x', 'y', 'z', 'likelihood']:
                if (self.new_scorer ,bodypart, coord) in data.columns:
                    ordered_columns.append((self.new_scorer, bodypart, coord))

        ordered_data = data[ordered_columns]
        return ordered_data

    def make_indices_DLC_compatible(self, data) -> pd.DataFrame:
        '''
        Add top level column index with a substitute scorer and name column levels. Reset index so continuous frames.
        :param data: 3D tracking file
        :return: Adjusted 3D tracking file
        '''
        # add DLC compatible column names
        data.columns = pd.MultiIndex.from_tuples(
            [(self.new_scorer, bodypart, coord) for bodypart, coord in data.columns],
            names=['scorer', 'bodyparts', 'coords']
        )

        # remove index names to be DLC compatible
        data.index.name = None
        data.reset_index(drop=True, inplace=True)

        return data

    def add_timestamps_to_realworld_data(self, sample_filenames, original_data) -> tuple[pd.DataFrame, np.ndarray]:
        ## Add confidence columns with median confidence value from single cam data files
        matched_cam_data = mfm.MatchFrames(sample_filenames, original_data, self.fps, self.data_ext, self.timestamp_ext).align_dfs()

        median_likelihoods = self.find_median_likelihoods(matched_cam_data, sample_filenames)

        ## Input relevant rows from likelihood columns to filled_realworld
        realworld_data_framesonly = self.drop_non_frame_indices(original_data['realworld'])
        trimmed_likelihoods = self.trim_likelihoods(original_data['realworld'], median_likelihoods)

        assert realworld_data_framesonly.index.equals(trimmed_likelihoods.index)
        assert (realworld_data_framesonly.columns.get_level_values(level=0).unique().sort_values().
                equals(trimmed_likelihoods.columns.get_level_values(level=0).unique().sort_values()))

        realworld_data_with_likelihoods = self.add_in_likelihoods(realworld_data_framesonly, trimmed_likelihoods)

        index_by_runs = self.get_indexes(realworld_data_with_likelihoods) # todo this is only getting side-wise indexes. Not taking into account the frame drifting between cams

        realworld_data_dlc_compat = self.make_indices_DLC_compatible(realworld_data_with_likelihoods)

        new_realworld_data = self.reorganise_3d_data_to_2d_format(realworld_data_dlc_compat)

        return new_realworld_data, index_by_runs

    def snip_and_stitch_videos(self, view, frame_indices):
        video_file_path = self.base_dir / f"{self.base_name}_{view}_1.avi"

        # load video
        vid = cv2.VideoCapture(str(video_file_path))

        # Get video properties
        fps = vid.get(cv2.CAP_PROP_FPS)
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') # MPEG-4 codec

        # Create output video writer
        output_path = self.base_dir / f"{self.output_name}_{view}view.avi"
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        print('Starting to save snipped and stitched video file...')
        for frame_idx in tqdm(frame_indices, desc="Processing frames"):
            vid.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = vid.read()
            if ret:
                out.write(frame)
            else:
                print(f"Warning: Could not read frame {frame_idx}")

        # Release resources
        vid.release()
        out.release()
        print('Finished!')

    def trim_single_cam_data(self, new_index, data_files) -> dict:
        single_cam_sample_data = dict(keys=self.cams)
        for cam in self.cams:
            data = data_files[cam]
            data_trimmed = data.loc(axis=0)[new_index]
            single_cam_sample_data[cam] = data_trimmed

        return single_cam_sample_data

    def save_sample_data(self):
        # Get data and tidy format
        sample_filename, original_data = self.get_files_and_paths()
        original_data['realworld'] = self.tidy_3d_datafile(original_data['realworld'])

        # Retrieve timestamps from individual camera tracking files (correcting for frame shifting)
        # and add to realworld data
        sample_data, sample_data_index = self.add_timestamps_to_realworld_data(sample_filename, original_data)

        # Save sample realworld (3D) data
        savepath = self.base_dir / f"{self.output_name}_3D.h5"
        sample_data.to_hdf(savepath,
                           key='df_with_missing',
                           mode='w')
        print(f"Saved sample data to {savepath}")

        # Trim individual cam data to match
        single_cam_sample_data = self.trim_single_cam_data(sample_data_index, original_data)

        # Save sample single cam data
        for cam in self.cams:
            savepath = self.base_dir / f"{self.output_name}_{cam}.h5"
            single_cam_sample_data[cam].to_hdf(savepath,
                                               key='df_with_missing',
                                               mode='w')
            print(f"Saved '{cam} cam' sample data to {savepath}")

        # snip and stitch the video file to match the frames in data file
        for cam in ['side']: # todo (see self.add_timestamps_to_realworld_data) only getting 'side' for now due to frame drift issue
            self.snip_and_stitch_videos(cam, sample_data_index)


def main():
    save = SaveSampleData(BASE_DIR, BASE_NAME, OUTPUT_NAME, config=CONFIG)
    save.save_sample_data()

if __name__ == '__main__':
    main()








