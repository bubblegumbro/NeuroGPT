import os
import pdb
import mne
import numpy as np
from batcher.base import EEGDataset
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
import torch
from batcher.base import process_gdf_file
import pandas as pd

class MotorImageryDataset(EEGDataset):
    def __init__(self, filenames, sample_keys, chunk_len=512, num_chunks=34, ovlp=51, root_path="", gpt_only=True):
        super().__init__(filenames, sample_keys, chunk_len, num_chunks, ovlp, root_path=root_path, gpt_only=gpt_only)

        self.data_all = []
        for fn in self.filenames:
            raw = mne.io.read_raw_gdf(fn, preload=True)
            self.data_all.append(raw)

        self.mi_types = {769: 'left', 770: 'right',
                         771: 'foot', 772: 'tongue', 1023: 'rejected'} # , 783: 'unknown', 1023: 'rejected'
        # Types of motor imagery
        self.labels_string2int = {'left': 0, 'right': 1,
                         'foot': 2, 'tongue':3 } #, 'unknown': -1
        self.Fs = 256  # 250Hz from original paper
        self.P = np.load("C:\\Users\\shreyas\\Documents\\GitHub\\NeuroGPT\\inputs\\tMatrix_value.npy")

        self.trials, self.labels, self.num_trials_per_sub = self.get_trials_all()
        # keys of data ['s', 'etyp', 'epos', 'edur', 'artifacts']

    def __len__(self):
        return sum(self.num_trials_per_sub)

    def __getitem__(self, idx):
        return self.preprocess_sample(self.trials[idx], self.num_chunks, self.labels[idx])

    def map2pret(self, data):
        #random_array = np.random.rand(7, 7)
        print("Shape of self.P:", self.P.shape)
        print("Shape of data:", data.shape)

        return np.matmul(self.P, data) # 22x22, 22xTime

    def get_trials_from_single_subj(self, sub_id):
        raw = self.data_all[sub_id]['s'].T
        events_type = self.data_all[sub_id]['etyp'].T
        events_position = self.data_all[sub_id]['epos'].T
        events_duration = self.data_all[sub_id]['edur'].T
        artifacts = self.data_all[sub_id]['artifacts'].T
        # Channel default is C3
        startrial_code = 768
        starttrial_events = (events_type == startrial_code)
       # print(type(starttrial_events))
        #print(starttrial_events)
        idxs = np.where(starttrial_events)[0]


        trial_labels = self.get_labels(sub_id)

        trials = []
        classes = []
        for j, index in enumerate(idxs):
            try:
                # print(index)
                # type_e = events_type[0, index+1]
                # class_e = self.mi_types[type_e]
                # if type_e == 1023:
                #     continue
                # classes.append(self.labels_string2int[class_e])
                classes.append(trial_labels[j])

                start = events_position[0, index]
                stop = start + events_duration[0, index]
                trial = raw[:22, start+500 : stop-375]
                #add band-pass filter
                # self.bandpass_filter(trial, lowcut=4, highcut=40, fs=250, order=5)
                trials.append(trial)
            except:
                # print("Cannot load trial")
                continue
        return trials, classes

    def get_labels(self, sub_id):
        label_path = self.root_path + "true_labels/"
        base_name = os.path.basename(self.filenames[sub_id])
        sub_name = os.path.splitext(base_name)[0]
        labels = loadmat(label_path + sub_name +".mat")["classlabel"]
        return labels.squeeze() - 1

    def get_trials_all(self):
        trials_all = []
        labels_all = []
        total_num = []
        for sub_id in range(len(self.data_all)):
            trials, labels = self.get_trials_from_single_subj(sub_id)
            total_num.append(len(trials))
            
            trials_all.append(np.array(trials))
            labels_all.append(np.array(labels))
        # reordered_data = self.reorder_channels(np.vstack(trials_all))
        trials_all_arr = np.vstack(trials_all)
        # map to same channel configuration as pretraining
        #trials_all_arr = self.map2pret(trials_all_arr)
        return self.normalize(trials_all_arr), np.array(labels_all).flatten(), total_num
    
    def normalize(self, data):
        data_min = np.min(data)
        data_max = np.max(data)
        return 2 * (data - data_min) / (data_max - data_min) - 1
        
    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        """
        Apply a bandpass filter to the data.
        
        Parameters:
        - data: The EEG signal
        - lowcut: Low cut-off frequency
        - highcut: High cut-off frequency
        - fs: Sampling rate (frequency)
        - order: Order of the filter
        
        Returns:
        - Filtered data
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data)
        
        return filtered_data
    

class EEGDatasetCls(EEGDataset):
    def __init__(self, folder_path, files=None):
        self.dataframes = []
        self.indices = []
        
        if files is None:
            files = os.listdir(folder_path)
        
        for file in files:
            if file.endswith('.gdf'):
                file_path = os.path.join(folder_path, file)
                df, idx = process_gdf_file(file_path)  
                if df is not None and idx is not None:
                    self.dataframes.append(df)
                    self.indices.extend(idx)
        
        # Combine all dataframes into a single DataFrame
        if self.dataframes:
            self.df = pd.concat(self.dataframes, ignore_index=True)
            self.df.set_index(["person", "epoch"], inplace=True)
        else:
            self.df = pd.DataFrame()

        self.idxs = self.indices

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
    
        # Extract the DataFrame row corresponding to the current index
        current_data = self.df.loc[self.idxs[idx]]
        #print(current_data.head())
        #print(current_data.columns)

        # Extract the 'condition' column and use it as the label
        label = current_data['condition'].unique().astype(int)

        # Exclude the 'condition' and 'time' columns from the input data
        input_columns = current_data.drop(columns=['condition', 'time']).columns
        data = current_data[input_columns].values.astype(np.float64)

        # Convert data to a PyTorch tensor and reshape
        data = torch.tensor(data, dtype=torch.float).transpose(0, 1).unsqueeze(0)  # Reshape to [1, channels, timepoints]
        #print('Data in getitems: ', data.shape)
        # Create batch dictionary
        batch = {
            'inputs': data,
            'labels': torch.tensor(label, dtype=torch.long)
        }
        
        return batch



