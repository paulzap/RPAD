# Anomaly Detection in Spacecraft Trajectory Data

## Overview

This repository provides a method for detecting anomalous measurements in spacecraft trajectory data using recursive partitioning of time series observations. The method leverages various data processing techniques to effectively identify and manage anomalous signals, making it suitable for handling noisy data streams common in space missions.

## Features

- **Automatic Detection**: The method automatically detects anomalies without requiring initial orbital approximations.
- **Flexible Application**: Works with various types of orbits and scales of observations.
- **No Pre-training Required**: The algorithm operates effectively without the need for prior training.
- **User-friendly**: The code is designed for ease of use, with clear instructions for implementation.
- **Open Source**: The algorithm's code is publicly available, allowing for modifications and improvements by the community.
- **Data Compatibility**: The method can detect anomalies in both labeled datasets (which may include a column `is_abnormal`, where 1 indicates anomalies and 0 indicates normal measurements) and unlabeled datasets (which may not contain this column).


## Functionality

### Available Functions

1. **startRPAD(anomalous_data_path, output_dir, Scaler, threshold, isPlot, isStat, trend_window, trend_polyorder, anom_size, max_cands, isDiff)**: 
   - Initiates the anomaly detection process on the provided dataset.
   - **Parameters**:
     - **anomalous_data_path**: The path to the directory containing CSV files with measurement intervals.
     - **output_dir**: The directory where the output CSV files documenting detected anomalies will be saved.
     - **Scaler**: Optional. Indicates whether to use a scaler for normalization (default: `True`).
     - **threshold**: The threshold value for anomaly detection (default: `0.4e-8`).
     - **isPlot**: Boolean indicating whether to plot the results (default: `True`).
     - **isStat**: Boolean indicating whether to compute statistics (default: `False`).
     - **trend_window**: Length of the window to remove the trend (default: `3`).
     - **trend_polyorder**: Polynomial order for the trend removal (default: `1`).
     - **anom_size**: The size of the anomaly window (default: `3`).
     - **max_cands**: Maximum number of candidate anomalies to detect (default: `7`).
     - **isDiff**: Boolean indicating whether to compute the first difference of observations before detection (default: `True`).

2. **remove_trend(time_series, window_length, polyorder)**: 
   - Removes trend from the time series data.
   - **Parameters**:
     - **time_series**: The input time series data as a list or array.
     - **window_length**: The length of the filter window (default: `3`).
     - **polyorder**: The order of the polynomial used to fit the trend (default: `1`).
   - **Returns**: The detrended time series and the estimated trend.

3. **getStatistics(test_dataset, threshold, trend_window, trend_polyorder, anom_size, max_cands, isDiff)**:
   - Computes statistical metrics for the anomaly detection results.
   - **Parameters**:
     - **test_dataset**: The dataset used for testing.
     - **threshold**: The threshold value for anomaly detection (default: `0.4e-8`).
     - **trend_window**: Length of the window for the trend removal (default: `3`).
     - **trend_polyorder**: Polynomial order for the trend removal (default: `1`).
     - **anom_size**: The size of the anomaly (default: `3`).
     - **max_cands**: Maximum number of candidate anomalies to detect (default: `7`).
     - **isDiff**: Boolean indicating whether to compute the first difference of observations (default: `True`).
   - **Returns**: Total true positives, false positives, false negatives, precision, recall, F1 score, and a list of bad samples.

4. **getAnomalies(test_dataset, index, threshold, isPlot, isStat, trend_window, trend_polyorder, anom_size, max_cands, isDiff)**:
   - Identifies anomalies in the specified file of the dataset.
   - **Parameters**:
     - **test_dataset**: The dataset from which anomalies will be detected.
     - **index**: The index of the specific file from dataset to be analyzed (default: `0`).
     - **threshold**: The threshold value for anomaly detection (default: `0.4e-8`).
     - **isPlot**: Boolean indicating whether to plot the results (default: `True`).
     - **isStat**: Boolean indicating whether to compute statistics (default: `False`).
     - **trend_window**: Length of the window size for the trend removal (default: `3`).
     - **trend_polyorder**: Polynomial order for the trend removal (default: `1`).
     - **anom_size**: The size of the anomaly (default: `3`).
     - **max_cands**: Maximum number of candidate anomalies to detect (default: `7`).
     - **isDiff**: Boolean indicating whether to compute the first difference of observations (default: `True`).
   - **Returns**: Indices of detected anomalies.

5. **multiRP(time_series, threshold, window_size, max_anomalies)**:
   - Performs recursive partitioning to detect anomalies in the time series, returning an array of anomaly indices.
   - **Parameters**:
     - **time_series**: The input time series data as an array.
     - **threshold**: The threshold for detecting anomalies (default: `1.5`).
     - **window_size**: Size of the window around detected anomalies (default: `3`).
     - **max_anomalies**: Maximum number of anomalies to detect (default: `7`).
   - **Returns**: Sorted array of detected anomaly indices.

6. **createDataset(anomalous_data_path, scaler)**:
   - Creates a dataset object from the specified path.
   - **Parameters**:
     - **anomalous_data_path**: The path to the directory containing the dataset files.
     - **scaler**: Optional. A scaler object for normalizing the data (default: `None`).
   - **Returns**: A dataset object.

7. **createScaler(anomalous_data_path)**:
   - Generates a scaler based on the data from input folder.
   - **Parameters**:
     - **anomalous_data_path**: The path to the directory containing the dataset files.
   - **Returns**: A scaler object.


## Input and Output Formats

### Input Data
- The input data should be structured in CSV files containing a column `observation_value`, which holds the measurement values to be analyzed. 
- For labeled datasets, there may also be a column `is_abnormal` where 1 indicates an anomaly and 0 indicates a normal measurement. Unlabeled datasets may not contain this column.
- The `anomalous_data_path` specifies the directory containing these CSV files of measurement intervals.

### Output Data
- The output will be saved in the specified `output_dir`, containing CSV files that document the detected anomalies for each input dataset. Each output file will include a column `is_abnormal`, where 1 denotes anomalous measurements and 0 denotes normal ones.

## Usage Instructions

1. **Installation**:
   - Install the package using:
     ```bash
     !pip install RPAD-0.1.0.tar.gz
     ```
   - Import the necessary modules:
     ```python
     import RPAD
     import RPAD as rp
     ```

2. **Running the Method**:

   You can run the method in several ways, depending on your requirements:

   - **Creating a Dataset and Scaler** (Scaler is optional):
     ```python
     scaler = rp.createScaler('path/to/anomalous/data')  # Optional
     dataset = rp.createDataset('path/to/anomalous/data', scaler=None)
     ```

   - **Getting Statistics on Labeled Data**:
     If your data is labeled and you want to obtain statistics on the accuracy of anomaly detection:
     ```python
     statistics = rp.getStatistics(dataset, threshold)
     ```

   - **Getting Anomaly Indices**:
     If you need to directly obtain the indices of anomalous measurements:
     ```python
     anomaly_indices = rp.getAnomalies(dataset, index, threshold)
     ```

   - **Using the Internal Recursive Partitioning Algorithm**:
     If you want to directly use the recursive partitioning method:
     ```python
     time_series_data = [your_time_series_array]
     anomaly_indices = rp.multiRP(time_series_data, threshold=1.5, window_size=3, max_anomalies=7)
     ```

## References

If you use this method in your research or applications, please cite the following paper:  
**[Insert Citation Here]**

## Contact

For any inquiries, please contact:  
**Pavel Zapevalin**  
Email: pav9981@yandex.ru
