"""
HDF5 and MATLAB file writers for the HLG analysis pipeline.

This module contains all data-writing functions extracted from the
original ``Data_writers.py``.  Three writers are provided:

* ``write_to_hdf5_file`` -- the primary output writer for the pipeline.
  Creates or appends to an HDF5 file with per-signal datasets and an
  optional scalar-metadata header section.
* ``write_to_mat_file`` -- exports a subset of signals to MATLAB ``.mat``
  format for the PhysioNet/Tool analysis streams.
* ``append_to_hdf5_file`` -- a variant of ``write_to_hdf5_file`` designed
  for incremental (multi-run) output where successive runs are joined
  column-wise into a single HDF5 file.

Design rationale
----------------
* **Chunk size of 64** -- HDF5 chunked storage is required for
  ``maxshape=(None,)`` (resizable) datasets.  64 samples is a
  reasonable default for 10 Hz respiratory data (~6.4 s per chunk).
* **String padding to 44 chars** -- the legacy pipeline pads header
  strings to a fixed width so that downstream MATLAB readers can
  treat them as fixed-length character arrays.
* **Header datetime handling** -- datetime objects are decomposed into
  a 7-element integer array ``[year, month, day, hour, minute, second,
  microsecond]`` because HDF5 has no native datetime type.
"""

from __future__ import annotations

import datetime
import os
from typing import Any

import h5py
import hdf5storage
import numpy as np
import pandas as pd


def write_to_hdf5_file(
    df: pd.DataFrame,
    output_h5_path: str,
    hdr: dict[str, Any] | None = None,
    default_dtype: str = "float32",
    overwrite: bool = False,
) -> None:
    """Write a DataFrame of time-series signals to an HDF5 file.

    Each column of ``df`` becomes a 1-D dataset in the HDF5 file.
    Column dtype is inferred from the column name:

    * **String columns** (annotation, test_type, rec_type, patient_tag,
      dataset) -> UTF-8 variable-length strings.
    * **Integer columns** (stage, apnea, Fs, newFs, cpap_start) -> int32,
      with NaN values replaced by -1 (HDF5 int arrays cannot hold NaN).
    * **All others** -> ``default_dtype`` (float32 by default).

    An optional ``hdr`` dict writes scalar metadata as additional
    length-1 datasets.

    Args:
        df: DataFrame whose columns are the signals to store.  Each
            column must be a 1-D array of length ``n_samples``.
        output_h5_path: Destination file path.  A ``.hf5`` extension is
            appended if not already present.
        hdr: Optional dict of scalar metadata to store alongside the
            signals.  Supported value types: ``int``, ``np.int32``,
            ``np.ndarray`` (int), ``str``, ``datetime.datetime``,
            ``pd.Timestamp``.
        default_dtype: NumPy dtype string for "generic" float signals.
        overwrite: If ``True``, delete any existing file before writing.
            If ``False`` and the file exists, new datasets are appended
            (but duplicate signal names will raise).

    Raises:
        ValueError: If a signal column already exists in the file and
            ``overwrite`` is ``False``, or if a header value has an
            unexpected type.
    """
    if hdr is None:
        hdr = {}

    # HDF5 chunked storage size -- must be set for resizable datasets.
    chunk_size = 64

    output_h5_path = output_h5_path if output_h5_path.endswith(".hf5") else output_h5_path + ".hf5"

    if overwrite and os.path.exists(output_h5_path):
        os.remove(output_h5_path)

    with h5py.File(output_h5_path, "a") as f:
        for signal in df.columns:
            if signal not in f:
                dtype1 = default_dtype

                # ── String signals ──────────────────────────────────
                # Annotations and categorical identifiers are stored as
                # variable-length UTF-8 strings so they survive HDF5
                # round-trips without encoding issues.
                if signal.lower() in [
                    "annotation",
                    "test_type",
                    "rec_type",
                    "patient_tag",
                    "dataset",
                ]:
                    dtype1 = h5py.string_dtype(encoding="utf-8")
                    dset_signal = f.create_dataset(
                        signal,
                        shape=(df.shape[0],),
                        maxshape=(None,),
                        chunks=(chunk_size,),
                        dtype=dtype1,
                    )
                    dset_signal[:] = df[signal].astype(str)

                # ── Integer signals ─────────────────────────────────
                # Sleep stages and apnea labels are categorical integers.
                # Sampling rate and CPAP onset are also integral.
                # NaN -> -1 because HDF5 integer arrays have no NaN.
                elif signal.lower() in ["stage", "apnea", "Fs", "newFs", "cpap_start"]:
                    dtype1 = "int32"
                    df.loc[pd.isna(df[signal]), signal] = -1

                # ── Float signals (default) ─────────────────────────
                else:
                    dset_signal = f.create_dataset(
                        signal,
                        shape=(df.shape[0],),
                        maxshape=(None,),
                        chunks=(chunk_size,),
                        dtype=dtype1,
                    )
                    dset_signal[:] = df[signal].astype(dtype1)
            else:
                raise ValueError(f'Signal "{signal}" already exists in file and overwrite is not allowed.')

        # ── Header metadata ─────────────────────────────────────────
        # Each header entry is stored as a separate length-1 dataset.
        # Type dispatching covers the legacy conventions used by the
        # original MATLAB and Python readers.
        if hdr:
            for key, value in hdr.items():
                if value is None:
                    value = str(value)

                # Datetime -> 7-element int32 array (HDF5 lacks a native
                # datetime type; this mirrors the MATLAB datenum convention).
                if isinstance(value, (datetime.datetime, pd.Timestamp)):
                    value = np.array(
                        [
                            value.year,
                            value.month,
                            value.day,
                            value.hour,
                            value.minute,
                            value.second,
                            value.microsecond,
                        ]
                    )

                if isinstance(value, (int, np.int32)):
                    f.create_dataset(
                        key,
                        shape=(1,),
                        maxshape=(1,),
                        chunks=True,
                        dtype=np.int32,
                    )[:] = np.int32(value)

                elif isinstance(value, np.ndarray):
                    f.create_dataset(
                        key,
                        shape=value.shape,
                        maxshape=(value.shape[0] + 10,),
                        chunks=True,
                        dtype=np.int32,
                    )[:] = value.astype(np.int32)

                elif isinstance(value, str):
                    # Pad strings to 44 chars -- legacy convention so that
                    # downstream MATLAB readers see fixed-width char arrays.
                    dtype_str = np.array([value + " " * (44 - len(value))]).astype("<S44").dtype
                    f.create_dataset(
                        key,
                        shape=(1,),
                        maxshape=(None,),
                        chunks=True,
                        dtype=dtype_str,
                    )[:] = value.encode("utf8")

                else:
                    raise ValueError(f'Unexpected datatype for header entry "{key}".')


def write_to_mat_file(
    data: pd.DataFrame,
    output_file: str,
    version: str,
    test_type: str,
    hdr: dict[str, Any] | None = None,
    spectrogram: dict[str, Any] | None = None,
    full_spectrogram: dict[str, Any] | None = None,
    default_dtype: str = "float32",
    overwrite: bool = False,
) -> None:
    """Export signal data to a MATLAB ``.mat`` file.

    Two export modes are supported, selected by ``version``:

    * **"Tool"** -- writes only the EEG + tool channels plus optional
      spectrogram dicts.  This is consumed by the in-house scoring tool.
    * **"PhysioNet"** -- writes a dense ``val`` matrix following the
      PhysioNet WFDB convention and creates the output directory if
      needed.  Calls ``create_ventilation_combined`` (from
      ``hlg.core.ventilation``) to derive the composite ventilation
      trace before export.

    Args:
        data: DataFrame of time-series signals.
        output_file: Destination ``.mat`` file path (or directory path
            for PhysioNet mode -- ``signals.mat`` is appended).
        version: Either ``"Tool"`` or ``"PhysioNet"``.
        test_type: Recording type identifier (e.g. ``"Diagnostic"``,
            ``"Split-Night"``).  Passed to
            ``create_ventilation_combined``.
        hdr: Header metadata dict (used by PhysioNet mode).
        spectrogram: Optional spectrogram dict for Tool mode.
        full_spectrogram: Optional full-spectrogram dict for Tool mode.
        default_dtype: NumPy dtype for float signals.
        overwrite: If ``True``, remove any existing output file first.

    Raises:
        ValueError: If ``version`` is not ``"Tool"`` or ``"PhysioNet"``.

    Note:
        The ``create_ventilation_combined`` function is imported at call
        time from ``hlg.core.ventilation``.  That module has not yet
        been migrated; when it is, update the import path here.
    """
    if hdr is None:
        hdr = {}
    if spectrogram is None:
        spectrogram = {}
    if full_spectrogram is None:
        full_spectrogram = {}

    # Standard EEG montage channels (International 10-20, contralateral
    # mastoid reference).
    eeg_channels = ["F3_M2", "F4_M1", "C3_M2", "C4_M1", "O1_M2", "O2_M1"]

    # Physiological channels used by the HLG analysis tool UI.
    tool_channels = [
        "Ventilation_combined",
        "CHEST",
        "ABD",
        "SpO2",
        "HR",
        "ECG",
        "Pleth",
        "Stage",
    ]

    all_data_dict: dict[str, Any] = {}

    if version == "Tool":
        for s in data.columns:
            if s in eeg_channels + tool_channels:
                all_data_dict[s] = np.array(data[s])
        all_data_dict.update(spectrogram)
        all_data_dict.update(full_spectrogram)

    elif version == "PhysioNet":
        # NOTE: create_ventilation_combined is defined in the ventilation
        # module; import here to avoid a hard top-level dependency until
        # that module is fully migrated.
        from hlg.core.ventilation import create_ventilation_combined  # type: ignore[import-not-found]

        data, _ = create_ventilation_combined(data, hdr, test_type)

        # PhysioNet channel ordering: 6 EEG + EOG + chin EMG + belts +
        # derived ventilation + SpO2 -> packed into a 2-D ``val`` matrix.
        s_list = eeg_channels + [
            "E1_M2",
            "CHIN1_CHIN2",
            "ABD",
            "CHEST",
            "Ventilation_combined",
            "SpO2",
        ]
        all_data = np.zeros((len(s_list), data.shape[0]))
        for n, s in enumerate(s_list):
            all_data[n, :] = np.array(data[s])
        all_data_dict["val"] = all_data

        if not os.path.exists(output_file):
            os.makedirs(output_file)
        output_file = os.path.join(output_file, "signals.mat")

        if overwrite and os.path.exists(output_file):
            os.remove(output_file)
    else:
        raise ValueError(f"Unsupported version: {version}")

    if not output_file.endswith(".mat"):
        output_file += ".mat"

    hdf5storage.savemat(output_file, all_data_dict)


def append_to_hdf5_file(
    data: pd.DataFrame,
    output_h5_path: str,
    default_dtype: str = "float32",
    run: int = 0,
) -> None:
    """Incrementally write signal data across multiple pipeline runs.

    On the first run (``run == 0``), any existing file is deleted and a
    fresh HDF5 is created.  On subsequent runs the existing file is
    loaded, the new columns are joined (column-wise), and the merged
    result is written back.

    This is used by pipelines that compute different signal groups in
    separate passes (e.g. respiratory features in run 0, EEG features in
    run 1) and want them consolidated into a single HDF5 file.

    Args:
        data: DataFrame of signals to write or append.
        output_h5_path: Destination ``.hf5`` file path.
        default_dtype: NumPy dtype for generic float signals.
        run: Zero-indexed pipeline run counter.  ``0`` means "start
            fresh"; any other value triggers a load-merge-rewrite cycle.

    Raises:
        ValueError: If a signal already exists in the file.

    Note:
        ``load_sleep_data`` is referenced but not yet migrated into the
        ``hlg`` package.  It should be imported from ``hlg.io.readers``
        once that function is available.  Currently this will raise a
        ``NameError`` if ``run > 0`` and an existing file is found.

    .. todo::
        Replace the bare ``load_sleep_data`` reference with the proper
        package import once the full reader suite is migrated.
    """
    chunk_size = 64

    if run == 0 and os.path.exists(output_h5_path):
        os.remove(output_h5_path)
    elif os.path.exists(output_h5_path):
        # TODO: Replace with ``from hlg.io.readers import load_sleep_data``
        # once that function is migrated.  The original ``Data_writers.py``
        # called a top-level ``load_sleep_data`` that read the full HDF5
        # into a DataFrame -- it is a different function from
        # ``load_sim_output`` (which reads only SS output columns).
        old_data, _ = load_sleep_data(output_h5_path)  # noqa: F821
        if not old_data.empty:
            data = old_data.join(data)
        os.remove(output_h5_path)

    with h5py.File(output_h5_path, "a") as f:
        for signal in data.columns:
            if signal not in f:
                # ── String signals ──────────────────────────────────
                if signal.lower() in [
                    "annotation",
                    "test_type",
                    "rec_type",
                    "patient_tag",
                ]:
                    dtype1 = h5py.string_dtype(encoding="utf-8")
                    f.create_dataset(
                        signal,
                        shape=(data.shape[0],),
                        maxshape=(None,),
                        chunks=(chunk_size,),
                        dtype=dtype1,
                    )[:] = data[signal].astype("str")

                # ── Integer signals ─────────────────────────────────
                # int8 is used here (vs int32 in write_to_hdf5_file)
                # because the append path was originally designed for
                # compact intermediate files where the stage/apnea
                # values fit in a byte.
                elif signal.lower() in [
                    "stage",
                    "apnea",
                    "Fs",
                    "newFs",
                    "cpap_start",
                ]:
                    dtype1 = "int8"
                    data.loc[pd.isna(data[signal]), signal] = -1

                # ── Float signals (default) ─────────────────────────
                else:
                    dtype1 = default_dtype
                    f.create_dataset(
                        signal,
                        shape=(data.shape[0],),
                        maxshape=(None,),
                        chunks=(chunk_size,),
                        dtype=dtype1,
                    )[:] = data[signal].astype(dtype1)
            else:
                raise ValueError(f'Signal "{signal}" already exists in file and overwrite is not allowed.')
