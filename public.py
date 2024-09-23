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
from .private import RPDataset, getDiffObservations, getObservations, filter_anomaly_indices, compare_indexes, recursive_partitioning, replace_with_neighbor_mean

def startRPAD(anomalous_data_path, output_dir, Scaler = True, threshold = 0.4e-8, isPlot=True, isStat = False, trend_window = 3, trend_polyorder = 1, anom_size = 3,  max_cands = 7, isDiff = True):
  scaler = None
  if Scaler:
    scaler = createScaler(anomalous_data_path)
  test_dataset = createDataset(anomalous_data_path, scaler)
  for i in range(len(test_dataset)):
    anom = getAnomalies(test_dataset, index = i, threshold = threshold, isPlot = isPlot, isStat = isStat, trend_window = trend_window, trend_polyorder = trend_polyorder , anom_size = anom_size, max_cands = max_cands, isDiff = isDiff)
    test_dataset.save_anomaly_to_csv(i, output_dir, anom)
	
	
def remove_trend(time_series, window_length=3, polyorder=1):
    trend = savgol_filter(time_series, window_length, polyorder)
    detrended_series = time_series - trend
    return detrended_series, trend
	
def getStatistics(test_dataset, threshold = 0.4e-8, trend_window = 3, trend_polyorder = 1, anom_size = 3, max_cands = 7, isDiff = True):
    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_test_samples_number = len(test_dataset)
    
    bad_samples = []

    for i in range(total_test_samples_number):
        _, (tp, fp, fn) = getAnomalies(test_dataset, i, threshold = threshold, isPlot=False, isStat = True, trend_window = trend_window, trend_polyorder = trend_polyorder , anom_size = anom_size, max_cands = max_cands, isDiff = isDiff)
        if fn!=0 or fp !=0:
            bad_samples.append(i)
            
        total_tp += tp
        total_fp += fp
        total_fn += fn

    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f1_score = 2 * (total_precision * total_recall) / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0

    print(f"Total True Positives: {total_tp}")
    print(f"Total False Positives: {total_fp}")
    print(f"Total False Negatives: {total_fn}")
    print(f"Overall Precision: {total_precision:.2f}")
    print(f"Overall Recall: {total_recall:.2f}")
    print(f"Overall F1 Score: {total_f1_score:.2f}")  
	
    return total_tp, total_fp, total_fn, total_precision, total_recall, total_f1_score, bad_samples
  
	
	
def getAnomalies(test_dataset, index = 0, threshold = 0.4e-8, isPlot=True, isStat = False, trend_window = 3, trend_polyorder = 1, anom_size = 3, max_cands = 7, isDiff = True):
    scaled = test_dataset.isScaled()
    observations, abnormal, file_name = test_dataset[index]
    if isDiff:
      diff_observations = getDiffObservations(observations)
    else:
      diff_observations = getObservations(observations)
    diff_observations, trend = remove_trend(diff_observations, trend_window, trend_polyorder)
    anomaly_indices = multiRP(diff_observations, threshold = threshold, window_size = anom_size, max_anomalies = max_cands)
    true_anoms = [i for i, value in enumerate(abnormal) if value]
    flag = len(true_anoms) > 0  
    anomaly_indices = filter_anomaly_indices(anomaly_indices, true_anoms, max_difference=anom_size)
    
    if isPlot:
            plt.figure(figsize=(10, 4))
            label = 'Trendless Observations 1st Derivative'
            if scaled:
              label = 'Scaled Trendless Observations 1st Derivative'
            plt.plot(diff_observations, color='gray',label=label)
            max_value = len(diff_observations)
            for i in anomaly_indices:
              plt.axvline(i, color='salmon', linestyle='solid', linewidth=5, alpha=0.5,  # alpha контролирует прозрачность
              label='Detected Anomaly' if i == anomaly_indices[0] else None)
              for offset in range(-anom_size//2, anom_size//2+1):
                position = i + offset
                if offset==0:
                  continue
                if 0 <= position <= max_value:
                  plt.axvline(position, color='salmon', linestyle='solid', linewidth=5, alpha=0.5)
                
            for i in true_anoms:
                plt.axvline(i, color='green', linestyle='--', linewidth=2, label='True Anomaly' if flag else None)
                    
            plt.xlabel('Time Step')
            plt.ylabel('Observation Value')
            plt.title('Anomaly Detection Results')
            plt.legend()
            plt.show()

            print()
            print(f"Detected Anomaly: {anomaly_indices}")
            print()
            
    if not isStat:
      return anomaly_indices
    else:
      return compare_indexes(true_anoms, anomaly_indices, anom_size)
	  



def multiRP(time_series, threshold=1.5, window_size=3, max_anomalies = 7):
    
    anomaly_indices = []
    data = time_series.copy()
    del_range = 0
    start_del_range = len(time_series)+1
    
    count = 0
    count_del = 0
    while count < max_anomalies: 
        anomaly_index = recursive_partitioning(data, threshold)
        count+=1
        if anomaly_index.size != 0:
            anomaly_index = anomaly_index[1]
            anomaly_indices.append(anomaly_index)
            start = max(0, anomaly_index - window_size)
            end = min(len(data), anomaly_index + window_size + 1)
            data = replace_with_neighbor_mean(data,start, end,anomaly_indices,  window_size)
            
            start_del_range = start
            del_range = end-start
        else:
            break

    return np.sort(anomaly_indices)
	
	
def createDataset(anomalous_data_path, scaler=None):
    test_dataset = RPDataset(anomalous_data_path, scaler=scaler)
    return  test_dataset
	
	
def createScaler(anomalous_data_path):
    file_names = [f for f in os.listdir(anomalous_data_path) if f.endswith('.csv')]
    scaler = StandardScaler()

    all_train_data = []

    for file_name in file_names:
        data = pd.read_csv(os.path.join(anomalous_data_path, file_name))['observation_value'].values
        all_train_data.extend(data)  
    scaler.fit(pd.DataFrame(all_train_data))
    return scaler