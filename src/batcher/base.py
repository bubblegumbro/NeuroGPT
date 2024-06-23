#!/usr/bin/env python3
from typing import Dict
import numpy as np
import torch
import mne
import h5py
import os

from torch.utils.data import Dataset

ds_max, ds_min = 100, -100

def _pad_seq_right_to_n(seq: np.ndarray, n: int, pad_value: float = 0.) -> np.ndarray:
    if n == seq.shape[0]:
        return seq
    return np.concatenate(
        [seq, np.ones((n - seq.shape[0], *seq.shape[1:])) * pad_value],
        axis=0,
    )

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


# Updated channel mapping for 10 channels (9 data channels + 1 compensation channel)
ch_map = {
    'LHC': 'LHC',
    'RHC': 'RHC',
    'LCA1': 'LCA1',
    'RCA1': 'RCA1',
    'PHC': 'PHC',
    'ERC': 'ERC',
    'LAMG': 'LAMG',
    'RAMG': 'RAMG',
    'CA1': 'CA1',
    'compensation': 'compensation',
}

ch_list = [
    'LHC',
    'RHC',
    'LCA1',
    'RCA1',
    'PHC',
    'ERC',
    'LAMG',
    'RAMG',
    'CA1',
    'compensation',
]

def scaler(x):
    if not isinstance(x, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    if x.size == 0:
        raise ValueError("Input array must not be empty.")

    x_min = np.min(x)
    x_max = np.max(x)

    if x_max == x_min:
        x_scaled = x / x_max if x_max != 0 else np.zeros_like(x)
        return x_scaled

    x_std = (x - x_min) / (x_max - x_min)
    x_scaled = (x_std * 2) - 1

    return x_scaled

def process_file(raw, ch_map, ch_list, ds_max=100, ds_min=-100):
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
        raw = raw.reorder_channels(ch_list)
    except Exception as e:
        print(f"Error in renaming or reordering channels: {e}")
        return None

    trial_min = np.min(raw.get_data())
    trial_max = np.max(raw.get_data())
    raw = raw.load_data().apply_function(scaler, channel_wise=False)

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

class EEGDataset(Dataset):
    def __init__(self, filenames, sample_keys, chunk_len=512, num_chunks=34, ovlp=51, root_path="", population_mean=0, population_std=1, gpt_only=False, normalization=True, start_samp_pnt=-1):
        if root_path == "":
            self.filenames = filenames
        else:
            print('Else')
            print('CHECK BELOW')
            print([root_path + fn for fn in filenames])
            self.filenames = [root_path + fn + '.pt' for fn in filenames]
            self.root_path = root_path
            
        print("Number of subjects loaded: ", len(self.filenames))
        print("Filenames loaded:", self.filenames)
        
        self.chunk_len = chunk_len
        self.num_chunks = num_chunks
        self.ovlp = ovlp
        self.sample_keys = sample_keys
        self.mean = population_mean
        self.std = population_std
        self.do_normalization = normalization
        self.gpt_only = gpt_only
        self.start_samp_pnt = start_samp_pnt

    def __len__(self):
        print(f"Dataset Length: {len(self.filenames)}")
        return len(self.filenames)

    def __getitem__(self, idx):
        print(f"Getting item: {idx}")
        data = self.load_tensor(self.filenames[idx])
        data = self.reorder_channels(data)
        return self.preprocess_sample(data, seq_len=self.num_chunks)

    @staticmethod
    def _pad_seq_right_to_n(seq: np.ndarray, n: int, pad_value: float = 0) -> np.ndarray:
        return _pad_seq_right_to_n(seq=seq, n=n, pad_value=pad_value)

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
        print(f"Attempting to load file: {filename}")
        tensor_data = torch.load(filename)
        print(f"Loaded file successfully: {filename}")
        return tensor_data.numpy()

    def reorder_channels(self, data):
        chann_labels = {'LHC': 0, 'RHC': 1, 'LCA1': 2, 'RCA1': 3, 'PHC': 4, 'ERC': 5, 'LAMG': 6, 'RAMG': 7, 'CA1': 8, 'compensation': 9}
        reorder_labels = {'LHC': 0, 'RHC': 1, 'LCA1': 2, 'RCA1': 3, 'PHC': 4, 'ERC': 5, 'LAMG': 6, 'RAMG': 7, 'CA1': 8, 'compensation': 9}

        reordered = np.zeros((10, data.shape[1]))
        for label, target_idx in reorder_labels.items():
            mapped_idx = chann_labels[label]
            reordered[target_idx, :] = data[mapped_idx, :]
        
        return reordered

    def split_chunks(self, data, length=512, ovlp=51, num_chunks=34, start_point=-1):
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
        return (data - mean) / (std + 1e-25)

    def preprocess_sample(self, sample, seq_len, labels=None) -> Dict[str, torch.Tensor]:
        out = {}
        if self.do_normalization:
            sample = self.normalize(sample)

        chunks, seq_on = self.split_chunks(sample, self.chunk_len, self.ovlp, seq_len, self.start_samp_pnt)

        attention_mask = np.ones(seq_len)
        chunks = self._pad_seq_right_to_n(seq=chunks, n=seq_len, pad_value=0)

        attention_mask = self._pad_seq_right_to_n(seq=attention_mask, n=seq_len, pad_value=0)
        
        if self.gpt_only == True:
            chunks = np.reshape(chunks, (seq_len, chunks.shape[1] * chunks.shape[2]))
        out["inputs"] = torch.from_numpy(chunks).to(torch.float)
        out["attention_mask"] = torch.from_numpy(attention_mask).to(torch.long)
        out['seq_on'] = seq_on
        out['seq_len'] = seq_len
        
        if self.sample_keys is not None:
            out = {key: out[key] for key in self.sample_keys if key in out}

        if labels is not None:
            out['labels'] = torch.from_numpy(np.array(labels)).to(torch.long)
   
        return out
