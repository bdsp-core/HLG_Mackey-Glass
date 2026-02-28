import os, h5py, hdf5storage, datetime
import numpy as np
import pandas as pd


# Data saving functions
def write_to_hdf5_file(df: pd.DataFrame, output_h5_path: str, hdr: dict = {}, 
                       default_dtype: str = 'float32', overwrite: bool = False) -> None:
    """
    Saves data from a DataFrame into an HDF5 file, including metadata in the header.

    Args:
        df (pd.DataFrame): DataFrame containing signals to save.
        output_h5_path (str): Output path for the HDF5 file.
        hdr (dict, optional): Header information to save as metadata.
        default_dtype (str, optional): Default data type for numerical values.
        overwrite (bool, optional): Overwrite the existing file if True.
    """

    chunk_size = 64
    output_h5_path = output_h5_path if output_h5_path.endswith('.hf5') else output_h5_path + '.hf5'

    if overwrite and os.path.exists(output_h5_path):
        os.remove(output_h5_path)

    with h5py.File(output_h5_path, 'a') as f:
        # Save signals from DataFrame
        for signal in df.columns:
            if signal not in f:
                dtype1 = default_dtype
                if signal.lower() in ['annotation', 'test_type', 'rec_type', 'patient_tag', 'dataset']:
                    dtype1 = h5py.string_dtype(encoding='utf-8')  # Save string data
                    dset_signal = f.create_dataset(signal, shape=(df.shape[0],), maxshape=(None,), 
                                                   chunks=(chunk_size,), dtype=dtype1)
                    dset_signal[:] = df[signal].astype(str)
                elif signal.lower() in ['stage', 'apnea', 'Fs', 'newFs', 'cpap_start']:
                    dtype1 = 'int32'  # Handle numerical data like sleep stages
                    df.loc[pd.isna(df[signal]), signal] = -1
                else:
                    dset_signal = f.create_dataset(signal, shape=(df.shape[0],), maxshape=(None,),
                                                   chunks=(chunk_size,), dtype=dtype1)
                    dset_signal[:] = df[signal].astype(dtype1)
            else:
                raise ValueError(f'Signal "{signal}" already exists in file and overwrite is not allowed.')

        # Save header metadata
        if hdr:
            for key, value in hdr.items():
                if value is None:
                    value = str(value)
                if isinstance(value, (datetime.datetime, pd.Timestamp)):
                    value = np.array([value.year, value.month, value.day, value.hour, value.minute, value.second, value.microsecond])
                if isinstance(value, (int, np.int32)):
                    f.create_dataset(key, shape=(1,), maxshape=(1,), chunks=True, dtype=np.int32)[:] = np.int32(value)
                elif isinstance(value, np.ndarray):
                    f.create_dataset(key, shape=value.shape, maxshape=(value.shape[0]+10,), chunks=True, dtype=np.int32)[:] = value.astype(np.int32)
                elif isinstance(value, str):
                    dtype_str = np.array([value + ' ' * (44-len(value))]).astype('<S44').dtype
                    f.create_dataset(key, shape=(1,), maxshape=(None,), chunks=True, dtype=dtype_str)[:] = value.encode('utf8')
                else:
                    raise ValueError(f'Unexpected datatype for header entry "{key}".')

def write_to_mat_file(data: pd.DataFrame, output_file: str, version: str, test_type: str, 
                      hdr: dict = {}, spectrogram: dict = {}, full_spectrogram: dict = {}, 
                      default_dtype: str = 'float32', overwrite: bool = False) -> None:
    """
    Save signals and optional spectrograms in a .mat file for MATLAB.

    Args:
        data (pd.DataFrame): DataFrame containing signals.
        output_file (str): Output path for the .mat file.
        version (str): Version of the format ('Tool' or 'PhysioNet').
        test_type (str): Type of test (used for some versions).
        hdr (dict, optional): Header data.
        spectrogram (dict, optional): Spectrogram data to save.
        full_spectrogram (dict, optional): Full spectrogram data to save.
        default_dtype (str, optional): Default data type for numerical values.
        overwrite (bool, optional): Overwrite the existing file if True.
    """

    eeg_channels = ['F3_M2', 'F4_M1', 'C3_M2', 'C4_M1', 'O1_M2', 'O2_M1']
    tool_channels = ['Ventilation_combined', 'CHEST', 'ABD', 'SpO2', 'HR', 'ECG', 'Pleth', 'Stage']
    all_data_dict = {}

    if version == 'Tool':
        for s in data.columns:
            if s in eeg_channels + tool_channels:
                all_data_dict[s] = np.array(data[s])

        # Add spectrograms
        all_data_dict.update(spectrogram)
        all_data_dict.update(full_spectrogram)

    elif version == 'PhysioNet':
        data, _ = create_ventilation_combined(data, hdr, test_type)
        s_list = eeg_channels + ['E1_M2', 'CHIN1_CHIN2', 'ABD', 'CHEST', 'Ventilation_combined', 'SpO2']
        all_data = np.zeros((len(s_list), data.shape[0]))

        for n, s in enumerate(s_list):
            all_data[n, :] = np.array(data[s])

        all_data_dict['val'] = all_data

        if not os.path.exists(output_file):
            os.makedirs(output_file)
        output_file = os.path.join(output_file, 'signals.mat')

        if overwrite and os.path.exists(output_file):
            os.remove(output_file)
    else:
        raise ValueError(f'Unsupported version: {version}')

    if not output_file.endswith('.mat'):
        output_file += '.mat'

    hdf5storage.savemat(output_file, all_data_dict)

def append_to_hdf5_file(data: pd.DataFrame, output_h5_path: str, default_dtype: str = 'float32', run: int = 0) -> None:
    """
    Append new data to an existing HDF5 file or create a new file if it doesn't exist.

    Args:
        data (pd.DataFrame): Data to append.
        output_h5_path (str): Path to the HDF5 file.
        default_dtype (str, optional): Default data type for numerical values.
        run (int, optional): If run=0, it will overwrite the file if it exists.
    """

    chunk_size = 64

    if run == 0 and os.path.exists(output_h5_path):
        os.remove(output_h5_path)
    elif os.path.exists(output_h5_path):
        old_data, _ = load_sleep_data(output_h5_path)
        if not old_data.empty:
            data = old_data.join(data)
        os.remove(output_h5_path)

    with h5py.File(output_h5_path, 'a') as f:
        for signal in data.columns:
            if signal not in f:
                if signal.lower() in ['annotation', 'test_type', 'rec_type', 'patient_tag']:
                    dtype1 = h5py.string_dtype(encoding='utf-8')
                    f.create_dataset(signal, shape=(data.shape[0],), maxshape=(None,), chunks=(chunk_size,), dtype=dtype1)[:] = data[signal].astype('str')
                elif signal.lower() in ['stage', 'apnea', 'Fs', 'newFs', 'cpap_start']:
                    dtype1 = 'int8'
                    data.loc[pd.isna(data[signal]), signal] = -1
                else:
                    dtype1 = default_dtype
                    f.create_dataset(signal, shape=(data.shape[0],), maxshape=(None,), chunks=(chunk_size,), dtype=dtype1)[:] = data[signal].astype(dtype1)
            else:
                raise ValueError(f'Signal "{signal}" already exists in file and overwrite is not allowed.')


