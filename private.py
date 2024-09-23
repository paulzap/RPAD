from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter 
import csv
from sklearn.preprocessing import StandardScaler



def getObservations(observations):
    diff_observations = ((observations.reshape(-1, 1)).squeeze())
    return diff_observations
	
	
def compare_indexes(true,anom, tolerance=3):

    true = np.array(true)
    anom = np.array(anom)

    correct_count = np.sum(np.any(np.abs(true[:, None] - anom[None, :]) <= tolerance, axis=1))

    false_positive_count = len(anom) - correct_count
    false_negative_count = len(true) - correct_count
    if false_positive_count<0:
        false_positive_count = 0

    return anom, (correct_count, false_positive_count, false_negative_count)

class RPDataset(Dataset):
    def __init__(self, data_path, scaler=None):
        self.data_path = data_path
        self.file_names = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        self.scaler = scaler

        self.data = []
        for file_name in self.file_names:
            file_path = os.path.join(self.data_path, file_name)
            data = pd.read_csv(file_path)

            observations = data['observation_value'].values
            if len(observations) < 15:
              continue
            if self.scaler is not None:
                observations = self.scaler.transform(observations.reshape(-1, 1)).squeeze()

            if 'is_abnormal' in data.columns:
                is_abnormal = torch.tensor(data['is_abnormal'].values, dtype=torch.float32)
            else:
                is_abnormal = torch.zeros(len(observations), dtype=torch.float32)

            self.data.append((observations, is_abnormal, file_name))

    def __len__(self):
        return len(self.data)

    def isScaled(self):
      if self.scaler is not None:
        return True
      else:
        return False
    
    def save_anomaly_to_csv(self, idx, output_dir, anom_ind, anom_size=3):
            observations, abnormal, file_name = self.data[idx]
            if self.scaler is not None:
              observations = self.scaler.inverse_transform(observations.reshape(-1, 1)).squeeze()
            data = []
			
            for obs_index, obs_value in enumerate(observations):
                is_abnormal = 1 if any(abs(obs_index - a) <= anom_size//2 for a in anom_ind) else 0
                data.append([obs_value, is_abnormal])
            current_directory = os.getcwd()
            output_directory = os.path.join(current_directory, output_dir)
            os.makedirs(output_directory, exist_ok=True)
            output_filename = os.path.join(output_directory, f'anomalies_{file_name}')
            with open(output_filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['observation_value', 'is_abnormal'])
                csvwriter.writerows(data)

    def __getitem__(self, idx):
        return self.data[idx]
		

def getDiffObservations(observations):
    diff_observations = np.diff((observations.reshape(-1, 1)).squeeze())
    return diff_observations


def filter_anomaly_indices(found_indices, true_indices, max_difference=3):
    filtered_indices = []
    used_true_indices = set()  

    for i, found_index in enumerate(found_indices):
        flag = False
        for true_index in true_indices:
            if abs(found_index - true_index) <= max_difference:
                if true_index not in used_true_indices:
                    filtered_indices.append(found_index)
                    used_true_indices.add(true_index)
                    flag = True
                    break 
                else:
                    flag = True
        if not flag:
            filtered_indices.append(found_index)

    return filtered_indices


def replace_with_neighbor_mean(data, cut_start, cut_end, exclude_centers, window_size):
    window_size+=1
    mask_cut = np.ones(len(data), dtype=bool)
    mask_cut[cut_start:cut_end] = False
    mask_exclude_centers = np.ones(len(data), dtype=bool)
    for center in exclude_centers[:-1]:
        mask_exclude_centers[center-window_size:center+window_size] = False 
    final_mask = mask_cut & mask_exclude_centers
    mean_value = np.mean(data[final_mask])
    mask_cut = np.zeros(len(data), dtype=bool)
    mask_cut[cut_start:cut_end] = True
    mask_exclude_centers = np.zeros(len(data), dtype=bool)
    for center in exclude_centers[:-1]:
        mask_exclude_centers[center-window_size:center+window_size] = True
    final_mask = mask_cut | mask_exclude_centers
    data[final_mask] = mean_value

    return data




def recursive_partitioning(time_series, threshold=1.5):

  anomaly_indices = []

  def partition_and_detect(data, anomaly_indices, start_index, end_index):
    if end_index - start_index <= 3:
      return anomaly_indices

    mid_index = (start_index + end_index) // 2
    left_std = np.std(data[start_index:mid_index])
    right_std = np.std(data[mid_index:end_index])

    if left_std > right_std:
      if left_std > threshold:
        anomaly_indices = np.array(range(start_index, mid_index))
      return partition_and_detect(data, anomaly_indices, start_index, mid_index)
    else:
      if right_std > threshold:
        anomaly_indices = np.array((range(mid_index, end_index)))
      return partition_and_detect(data, anomaly_indices, mid_index, end_index)

  anomaly_indices = partition_and_detect(time_series, anomaly_indices, 0, len(time_series))
  if len(anomaly_indices)>3:
    anomaly_indices = []
  return np.array(anomaly_indices)