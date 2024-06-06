#!/usr/bin/env python3
from typing import Dict
import numpy as np
# import webdataset as wds
import torch
# import gzip
# import pickle
import mne
import h5py
import os
# import webdataset as wds

from torch.utils.data import Dataset
ds_max, ds_min = 100, -100
def _pad_seq_right_to_n(
    seq: np.ndarray,
    n: int,
    pad_value: float = 0.
    ) -> np.ndarray:
    if n == seq.shape[0]:
        return seq
    return np.concatenate(
        [
            seq,
            np.ones(
                (
                    n-seq.shape[0],
                    *seq.shape[1:]
                )
            ) * pad_value,  
        ],
        axis=0,
    )

ch_map = {
    "EEG-Fz": "FZ",
    "EEG-0": "FC3",
    "EEG-1": "FC1",
    "EEG-2": "FCZ",
    "EEG-3": "FC2",
    "EEG-4": "FC4",
    "EEG-5": "C5",
    "EEG-C3": "C3",
    "EEG-6": "C1",
    "EEG-Cz": "CZ",
    "EEG-7": "C2",
    "EEG-C4": "C4",
    "EEG-8": "C6",
    "EEG-9": "CP3",
    "EEG-10": "CP1",
    "EEG-11": "CPZ",
    "EEG-12": "CP2",
    "EEG-13": "CP4",
    "EEG-14": "P1",
    "EEG-Pz": "PZ",
    "EEG-15": "P2",
    "EEG-16": "POZ",
}
ch_list = [
    "FP1",
    "FP2",
    "F7",
    "F3",
    "FZ",
    "F4",
    "F8",
    "T3",
    "C3",
    "CZ",
    "C4",
    "T4",
    "T5",
    "P3",
    "PZ",
    "P4",
    "T6",
    "O1",
    "O2",
]

keys_with_values_in_list = [key for key, value in ch_map.items() if value in ch_list]

def scaler(x):
    """
    Scales the input array x to the range [-1, 1].

    Parameters:
    - x (numpy.ndarray): The input array to be scaled.

    Returns:
    - numpy.ndarray: The scaled array.

    Raises:
    - ValueError: If the input is not a numpy array.
    - ValueError: If the input array is empty.
    - ZeroDivisionError: If the max and min values of the array are the same.
    """

    # Check if input is a numpy array
    if not isinstance(x, np.ndarray):
        raise ValueError("Input must be a numpy array.")

    # Check if the array is empty
    if x.size == 0:
        raise ValueError("Input array must not be empty.")

    # Calculate min and max
    x_min = np.min(x)
    x_max = np.max(x)

    # Check for division by zero
    if x_max == x_min:
        x_scaled = x / x_max if x_max != 0 else np.zeros_like(x)
        return x_scaled

    # Perform scaling
    x_std = (x - x_min) / (x_max - x_min)
    x_scaled = (x_std * 2) - 1

    return x_scaled

def process_file(raw, ch_map, ch_list, ds_max=100, ds_min=-100):
    # selects 19 standard channels and adds a 20th
    raw = raw.copy()
    try:
        raw = raw.pick(ch_list)
    except ValueError as v:
        pl = v.args[0].split("[")[1].split("]")[0].split(",")
        pl = [p.strip(" ' ") for p in pl]
        new_pick = list(set(ch_list) - set(pl))
        raw = raw.pick(new_pick)

    if len(raw.ch_names) != len(ch_list):
        missing_channels = [ch for ch in ch_list if ch not in raw.ch_names]

        new_channel_data = np.vstack(
            [np.full((1, raw.n_times), 0)] * len(missing_channels)
        )
        new_channel_info = mne.create_info(
            ch_names=missing_channels,
            sfreq=raw.info["sfreq"],
            ch_types=["eeg"] * len(missing_channels),
        )
        new_channel_raw = mne.io.RawArray(
            data=new_channel_data, info=new_channel_info, first_samp=raw.first_samp
        )
        raw.load_data().add_channels([new_channel_raw], force_update_info=True)

    try:
        # raw = raw.rename_channels(ch_map)
        raw = raw.reorder_channels(ch_list)
    except Exception as e:
        print(f"Error in renaming or reordering channels: {e}")
        return None

    # scale
    trial_min = np.min(raw.get_data())
    trial_max = np.max(raw.get_data())
    raw = raw.load_data().apply_function(scaler, channel_wise=False)

    # add compensation channel
    compensation = (trial_max - trial_min) / (ds_max - ds_min)
    comp_ch_data = np.full((1, raw.n_times), compensation)
    comp_ch_info = mne.create_info(
        ch_names=["compensation"], sfreq=raw.info["sfreq"], ch_types="misc"
    )
    comp_ch_raw = mne.io.RawArray(
        data=comp_ch_data, info=comp_ch_info, first_samp=raw.first_samp
    )
    raw.add_channels([comp_ch_raw], force_update_info=True)

    return raw

def process_gdf_file(gdf_file):
    print("the file to be processed is: ", gdf_file)
    try:
        f = mne.io.read_raw_gdf(
            gdf_file, eog=["EOG-left", "EOG-central", "EOG-right"], preload=True
        )
        f.drop_channels(["EOG-left", "EOG-central", "EOG-right"])
    except Exception as e:
        print(f"Error reading EDF file {gdf_file}: {e}")
        return

    assert "lowpass" in f.info, "lowpass information is not available in f.info"
    assert f.info["lowpass"] > 0, "lowpass frequency should be greater than 0"
    assert f.info["sfreq"] > 0, "Sampling frequency should be greater than 0"

    if f.info["bads"]:
        print(f"Warning: The following channels are marked as bad: {f.info['bads']}")
        print(gdf_file)
        # input("Press Enter to continue or Ctrl+C to abort.")

    if 256 >= 2 * f.info.get("lowpass", 0):
        try:
            f = f.resample(sfreq=256)
            f = f.rename_channels(ch_map)
            f = process_file(
                f,
                ch_map=ch_map,
                ch_list=ch_list,
                ds_max=ds_max,
                ds_min=ds_min,
            )
        except Exception as e:
            print(
                f"An error occurred while processing the file {gdf_file}: {e} or while resampling"
            )
            # continue

        event_id = {"769": 0, "770": 1, "771": 2, "772": 3}
        events = mne.events_from_annotations(f, event_id=event_id)
        epochs = mne.Epochs(
            f, events[0], [0, 1, 2, 3], tmin=-2, tmax=4, on_missing="warn"
        )
        # print("here", np.max(f.get_data()), np.min(f.get_data()))
        df = epochs.to_data_frame(scalings=dict(eeg=1, mag=1, grad=1))
        # print("df", df.iloc[:, 3:].values.max(), df.iloc[:, 3:].values.min())
        df["person"] = f.info["subject_info"]["his_id"]
        indices = [(f.info["subject_info"]["his_id"], ep) for ep in df.epoch.unique()]

        return df, indices

class EEGDataset(Dataset):
    def __init__(self, filenames, sample_keys, chunk_len=512, num_chunks=34, ovlp=51, root_path="", population_mean=0, population_std=1, gpt_only=False, normalization=True, start_samp_pnt=-1):
        if root_path == "":
            self.filenames = filenames
        else:
            self.filenames = [root_path + fn for fn in filenames if os.path.isfile(root_path+fn)]
            self.root_path = root_path
            
       # print("Number of subjects loaded: ", len(self.filenames))
        #print("Filenames loaded:", self.filenames)
        
        # self.data = data_all
        self.chunk_len = chunk_len
        self.num_chunks = num_chunks
        self.ovlp = ovlp
        self.sample_keys = sample_keys
        self.mean = population_mean
        self.std = population_std
        self.do_normalization = normalization
        self.gpt_only=gpt_only
        self.start_samp_pnt = start_samp_pnt

    def __len__(self):
        #print(f"Dataset Length: {len(self.filenames)}")
        return len(self.filenames)

    def __getitem__(self, idx):
        #print(f"Getting item: {idx}")
        data = self.load_tensor(self.filenames[idx])
        #===reorder channels====
        data = self.reorder_channels(data)
        return self.preprocess_sample(data, seq_len=self.num_chunks)

    @staticmethod
    def _pad_seq_right_to_n(
        seq: np.ndarray,
        n: int,
        pad_value: float = 0
        ) -> np.ndarray:
        return _pad_seq_right_to_n(
            seq=seq,
            n=n,
            pad_value=pad_value
        )

    def load_single_file(self, filename):
        with h5py.File(filename, 'r') as file:
            data_dict = file['Result']
            data = []
            for i in range(data_dict['data'].shape[0]):  
                ref = data_dict['data'][i][0]
                time_series = data_dict[ref]
                if len(data) > 0 and time_series.shape[0] < data[0].shape[0]:
                    time_series = np.zeros_like(data[0])
                data.append(np.array(time_series).squeeze())
        return data

    def load_tensor(self, filename):
        # tensor_fn = filename[:-3] + 'pt'
        #print(f"Attempting to load file: {filename}")
        tensor_data = torch.load(filename)
       # print(f"Loaded file successfully: {filename}")
        return tensor_data.numpy()

    def reorder_channels(self, data):
    # Updated channel labels with 'T1' and 'T2' removed
        chann_labels = {'FP1': 0, 'FP2': 1, 'F3': 2, 'F4': 3, 'C3': 4, 'C4': 5, 'P3': 6, 'P4': 7, 'O1': 8, 'O2': 9, 'F7': 10, 'F8': 11, 'T3': 12, 'T4': 13, 'T5': 14, 'T6': 15, 'FZ': 16, 'CZ': 17, 'PZ': 18, 'OZ': 19}
        reorder_labels = {'FP1': 0, 'FP2': 1, 'F7': 2, 'F3': 3, 'FZ': 4, 'F4': 5, 'F8': 6, 'T3': 7, 'C3': 8, 'CZ': 9, 'C4': 10, 'T4': 11, 'T5': 12, 'P3': 13, 'PZ': 14, 'P4': 15, 'T6': 16, 'O1': 17, 'OZ': 18, 'O2': 19}

        # Adjust the array size to 20 channels
        reordered = np.zeros((20, data.shape[1]))
        for label, target_idx in reorder_labels.items():
            mapped_idx = chann_labels[label]
            reordered[target_idx, :] = data[mapped_idx, :]
        
        return reordered


    def split_chunks(self, data, length=512, ovlp=51, num_chunks=34, start_point=-1): 
        '''2 seconds, 0.2 seconds overlap'''
        all_chunks = []
        total_len = data.shape[1]
        actual_num_chunks = num_chunks
        
        if start_point == -1:
            if num_chunks * length > total_len - 1:
                start_point = 0
                actual_num_chunks = total_len // length
            else:
                start_point = np.random.randint(0, total_len - num_chunks * length)
        
        for i in range(actual_num_chunks):
            chunk = data[:, start_point: start_point + length]
            all_chunks.append(np.array(chunk))
            start_point = start_point + length - ovlp
        return np.array(all_chunks), start_point
    
    def normalize(self, data):
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        # Ensure std is not zero to avoid division by zero.
        # If std is zero, normalization doesn't make sense, 
        # so you might set std to a small positive value or handle it in another way.
        # std = np.where(std == 0, 1e-23, std)
        return (data - mean) / (std + 1e-25)

    def preprocess_sample(
        self,
        sample,
        seq_len,
        labels=None
        ) -> Dict[str, torch.Tensor]:
        out = {}
        if self.do_normalization:
            sample = self.normalize(sample)

        chunks, seq_on = self.split_chunks(sample, self.chunk_len, self.ovlp, seq_len, self.start_samp_pnt)

        attention_mask = np.ones(seq_len)
        chunks = self._pad_seq_right_to_n(
            seq=chunks,
            n=seq_len,
            pad_value=0
        )

        attention_mask = self._pad_seq_right_to_n(
            seq=attention_mask, 
            n=seq_len,
            pad_value=0
        )
        
        if self.gpt_only == True:
            chunks = np.reshape(chunks, (seq_len, chunks.shape[1]*chunks.shape[2]))
        out["inputs"] = torch.from_numpy(chunks).to(torch.float)
        out["attention_mask"] = torch.from_numpy(attention_mask).to(torch.long)
        out['seq_on'] = seq_on
        out['seq_len'] = seq_len
        
        if self.sample_keys is not None:
            out = {
                key: out[key] 
                for key in self.sample_keys
                if key in out
            }

        if labels is not None:
            out['labels'] = torch.from_numpy(np.array(labels)).to(torch.long)
   
        return out