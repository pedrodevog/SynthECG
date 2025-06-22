import torch
import numpy as np
import random
from torch.utils.data import Dataset

class ToTensor:
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, transpose_data1d=True):
        self.transpose_data1d = transpose_data1d

    def __call__(self, sample):
        def _to_tensor(data, transpose_data1d=False):
            if(len(data.shape) == 2 and transpose_data1d is True): # Swap channel and time axis for direct application of pytorch's 1d convs
                data = data.transpose((1, 0))
            if(isinstance(data, np.ndarray)):
                return torch.from_numpy(data)
            else: # default_collate will take care of it
                return data
            
        data, label, ID = sample['data'], sample['label'], sample['ID']

        if not isinstance(data, tuple): 
            data = _to_tensor(data, self.transpose_data1d)
        else:
            data = tuple(_to_tensor(x, self.transpose_data1d) for x in data)
        
        if not isinstance(label, tuple):
            label = _to_tensor(label)
        else:
            label = tuple(_to_tensor(x) for x in label)

        return data, label # Returning as a tuple (potentially of lists)

class TimeseriesDatasetCrops(torch.utils.data.Dataset):
    """Timeseries dataset with partial crops."""

    def __init__(self, df, output_size, chunk_length, min_chunk_length, npy_data=None, 
                 random_crop=True, data_folder=None, num_classes=2, copies=0, col_lbl="label", 
                 stride=None, start_idx=0, annotation=False, transforms=[]):
        """
        Accepts npy_data [samples,ts,...] (either path or np.array directly- also supporting variable length input)
        - label column in df corresponds to sampleid
        
        transforms: list of callables (transformations) (applied in the specified order i.e. leftmost element first)
        """
        self.timeseries_df = df
        self.output_size = output_size
        self.data_folder = data_folder
        self.transforms = transforms
        self.annotation = annotation
        self.col_lbl = col_lbl
        self.c = num_classes
        self.mode = "npy"
        
        if isinstance(npy_data, np.ndarray) or isinstance(npy_data, list):
            self.npy_data = np.array(npy_data)
            assert(annotation is False)
        else:
            self.npy_data = np.load(npy_data)
        
        self.random_crop = random_crop

        self.df_idx_mapping = []
        self.start_idx_mapping = []
        self.end_idx_mapping = []

        for df_idx, (id, row) in enumerate(df.iterrows()):
            data_length = len(self.npy_data[row["data"]])
                                              
            if chunk_length == 0: # do not split
                idx_start = [start_idx]
                idx_end = [data_length]
            else:
                idx_start = list(range(start_idx, data_length, chunk_length if stride is None else stride))
                idx_end = [min(l + chunk_length, data_length) for l in idx_start]

            # remove final chunk(s) if too short
            for i in range(len(idx_start)):
                if idx_end[i] - idx_start[i] < min_chunk_length:
                    del idx_start[i:]
                    del idx_end[i:]
                    break
            
            # append to lists
            for _ in range(copies + 1):
                for i_s, i_e in zip(idx_start, idx_end):
                    self.df_idx_mapping.append(df_idx)
                    self.start_idx_mapping.append(i_s)
                    self.end_idx_mapping.append(i_e)
                    
    def __len__(self):
        return len(self.df_idx_mapping)

    def __getitem__(self, idx):
        df_idx = self.df_idx_mapping[idx]
        start_idx = self.start_idx_mapping[idx]
        end_idx = self.end_idx_mapping[idx]
        
        # Determine crop idxs
        timesteps = end_idx - start_idx
        assert(timesteps >= self.output_size)
        
        if self.random_crop: # Random crop
            if timesteps == self.output_size:
                start_idx_crop = start_idx
            else:
                start_idx_crop = start_idx + random.randint(0, timesteps - self.output_size - 1)
        else:
            start_idx_crop = start_idx + (timesteps - self.output_size) // 2
            
        end_idx_crop = start_idx_crop + self.output_size

        # Load the actual data
        ID = self.timeseries_df.iloc[df_idx]["data"]
        data = self.npy_data[ID][start_idx_crop:end_idx_crop]
        label = self.timeseries_df.iloc[df_idx][self.col_lbl]
        
        sample = {'data': data, 'label': label, 'ID': ID}
        
        for t in self.transforms:
            sample = t(sample)

        return sample
    
    def get_id_mapping(self):
        return self.df_idx_mapping