"""
Event detection and manipulation utilities for sleep signal arrays.

This is the most widely-used utility module in the codebase. It converts
binary or labelled 1-D arrays (e.g. apnea masks, arousal masks, sleep-stage
labels) into discrete event representations ``[(start, end), ...]`` and back,
and provides helpers for merging, windowing, and correcting those events.

The core abstraction is simple: a contiguous run of non-zero values in a
signal array is an **event**. ``find_events`` extracts them, ``events_to_array``
reconstructs the array, and the remaining functions post-process the event
list (gap-filling, window expansion, label repair).

No local package dependencies -- only numpy and pandas are used.
"""

import numpy as np
import pandas as pd


def find_events(sig: np.ndarray, Fs: int = 10) -> list[tuple[int, int]]:
    """Detect contiguous non-zero regions in a 1-D signal.

    Scans a signal array for transitions between zero and non-zero values and
    returns a list of ``(start_index, end_index)`` tuples.  This is the
    canonical way the pipeline converts a binary/labelled mask into a list of
    discrete events (apneas, arousals, desaturations, etc.).

    The algorithm works by computing the first-order difference of the signal
    and classifying each transition as a rising edge (event start) or a falling
    edge (event end).  Boundary conditions (signal starts or ends in a non-zero
    state) are handled explicitly.

    Args:
        sig: 1-D numeric array where non-zero values indicate active events
            and zero indicates no event.
        Fs: Sampling frequency in Hz. Passed through to ``label_correction``
            when start/end counts are mismatched.

    Returns:
        A list of ``(start, end)`` tuples. Each tuple marks the inclusive
        start index and exclusive end index of a contiguous non-zero region.
        Returns an empty list if no events are found.

    Raises:
        AssertionError: If start/end pairing cannot be resolved even after
            automatic label correction.
    """
    signal = np.array(sig)
    starts: list[int] = []
    ends: list[int] = []

    # Handle leading edge: if the signal is already active at index 0,
    # that's the start of the first event.
    if signal[0] > 0:
        starts.append(0)

    # First-order difference highlights every transition point.
    diff_drops = pd.DataFrame(signal).diff()
    starts, ends = define_events_start_ends(signal, diff_drops, starts, ends)

    # Handle trailing edge: if the signal is still active at the last sample,
    # close the event at the array boundary.
    if signal[-1] > 0:
        ends.append(len(signal))
        signal[-1] = 0

    if len(ends) == len(starts) == 0:
        return []

    # Mismatched counts indicate overlapping/merged labels in the source data.
    # Attempt automatic repair before failing.
    if len(ends) != len(starts):
        print("Attempting label correction...")
        starts, ends = label_correction(starts, ends, signal, Fs)

    assert len(ends) == len(starts), "Mismatched starts and ends after correction."

    grouped_events = list(zip(starts, ends))
    return grouped_events


def define_events_start_ends(
    signal: np.ndarray,
    diff_drops: pd.DataFrame,
    starts: list[int],
    ends: list[int],
) -> tuple[list[int], list[int]]:
    """Classify each non-zero difference as a rising or falling edge.

    Iterates over every sample where the first-order difference is non-zero
    (i.e. a transition occurred).  A positive step (signal increases) is
    classified as an event start; a negative step (signal decreases) is an
    event end.  Equal non-zero values would be logically impossible for a
    valid binary/labelled mask and raise a ``ValueError``.

    Args:
        signal: The original 1-D signal array.
        diff_drops: First-order difference of ``signal`` as a single-column
            DataFrame (produced by ``pd.DataFrame(signal).diff()``).
        starts: Accumulator list of event start indices (may already contain
            the index 0 if the signal starts non-zero).
        ends: Accumulator list of event end indices.

    Returns:
        Updated ``(starts, ends)`` lists with all interior transitions added.

    Raises:
        ValueError: If a transition is detected but the step direction is
            ambiguous (should never happen with well-formed input).
    """
    for v in np.where(diff_drops[1:])[0]:
        # diff_drops is offset by 1 because diff() produces NaN at index 0,
        # so we shift the location index back to the original signal frame.
        loc = v + 1
        step = signal[loc]
        step_min_one = signal[loc - 1]

        if step > step_min_one:
            starts.append(loc)
        elif step < step_min_one:
            ends.append(loc)
        else:
            raise ValueError("Unexpected event error in find_events.")
    return starts, ends


def label_correction(
    starts: list[int],
    ends: list[int],
    signal: np.ndarray,
    Fs: int,
) -> tuple[list[int], list[int]]:
    """Repair start/end mismatches caused by adjacent merged labels.

    In polysomnography scoring, two events of different types can be annotated
    back-to-back (e.g. an obstructive apnea immediately followed by a central
    apnea).  In the labelled array this appears as a direct transition from
    one non-zero label to another without an intervening zero -- creating an
    extra start without a matching end (or vice-versa).

    This function finds such "merged" locations and resolves the conflict by
    keeping the longer event and absorbing the shorter one.

    Args:
        starts: Event start indices (may be longer or shorter than ``ends``).
        ends: Event end indices.
        signal: The original labelled signal array.
        Fs: Sampling frequency in Hz (reserved for future duration checks).

    Returns:
        Corrected ``(starts, ends)`` lists of equal length.
    """
    merged_locs = search_for_merged_labels(signal)

    for p, loc, n in merged_locs:
        # Compare the label values on either side of the merge point to
        # decide which event has priority (lower label value wins, keeping
        # the standard clinical ordering: obstructive < central < mixed).
        event1 = signal[loc - 1]
        event2 = signal[loc + 1]
        priority_loc = np.argmin([event1, event2])

        if priority_loc == 0:
            # First event wins -- remove the second event's start and extend
            # the first event's end to cover the merge region.
            starts = [s for s in starts if p != s]
            if n in ends:
                ends[ends.index(n)] = loc
        else:
            # Second event wins -- remove the first event's end and extend
            # the second event's start backward.
            ends = [e for e in ends if loc != e]
            if p in starts:
                starts[starts.index(p)] = loc

    return starts, ends


def search_for_merged_labels(signal: np.ndarray) -> list[tuple[int, int, int]]:
    """Find locations where two labelled events are directly adjacent.

    Scans the first-order difference for consecutive same-sign transitions
    (two rises or two falls in a row), which indicate that one event label
    transitions directly into another without returning to zero.

    Args:
        signal: 1-D labelled signal array.

    Returns:
        A list of ``(prev_index, merge_index, next_index)`` tuples, where
        ``merge_index`` is the boundary between the two merged events and
        the surrounding indices provide context for the correction logic.
    """
    locs: list[tuple[int, int, int]] = []
    diff_drops = pd.DataFrame(signal).diff().fillna(0)
    diff_drops = diff_drops[diff_drops != 0].dropna()

    for i in range(len(diff_drops) - 1):
        current_val = diff_drops.iloc[i].values[0]
        next_val = diff_drops.iloc[i + 1].values[0]

        # Two consecutive positive diffs -> two starts in a row (missing end).
        if current_val > 0 and next_val > 0:
            locs.append((diff_drops.index[i], diff_drops.index[i + 1], diff_drops.index[i + 2]))
        # Two consecutive negative diffs -> two ends in a row (missing start).
        elif current_val < 0 and next_val < 0:
            locs.append((diff_drops.index[i - 1], diff_drops.index[i], diff_drops.index[i + 1]))

    return locs


def events_to_array(
    events: list[tuple[int, int]],
    len_array: int,
    labels: list[int] | None = None,
) -> np.ndarray:
    """Convert a list of events back into a labelled signal array.

    This is the inverse of ``find_events``: given ``(start, end)`` pairs and
    optional per-event labels, reconstruct the full-length 1-D array.

    Args:
        events: List of ``(start, end)`` index tuples.
        len_array: Length of the output array (must cover all event indices).
        labels: Per-event label values. Defaults to all-ones if not provided.

    Returns:
        A 1-D numpy array of length ``len_array`` with labelled event regions
        and zeros elsewhere.
    """
    array = np.zeros(len_array)
    if labels is None:
        labels = [1] * len(events)
    for i, (st, end) in enumerate(events):
        array[st:end] = labels[i]
    return array


def window_correction(array: np.ndarray, window_size: int) -> np.ndarray:
    """Expand each event symmetrically by half a window.

    Useful for adding temporal tolerance around scored events.  For example,
    an arousal annotation might be slightly offset from the actual EEG
    activation -- expanding it by a few seconds ensures downstream processing
    captures the full physiological response.

    Args:
        array: 1-D labelled signal array.
        window_size: Total expansion window in samples.  Each event is
            extended by ``window_size // 2`` on each side.

    Returns:
        A copy of the array with every event expanded by the half-window,
        cast to integer dtype.
    """
    half_window = int(window_size // 2)
    events = find_events(array)
    corr_array = np.array(array)

    for st, end in events:
        label = array[st]
        corr_array[max(0, st - half_window) : min(len(array), end + half_window)] = label

    return corr_array.astype(int)


def connect_events(
    events: list[tuple[int, int]],
    win: int,
    Fs: int,
    max_dur: bool = False,
    labels: list[int] | None = None,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Merge events that are separated by less than a given gap.

    In respiratory event scoring it is common for a single apnea or hypopnea
    to be split into two annotations if the scorer briefly sees a partial
    recovery.  This function stitches such fragments back together when the
    inter-event gap is smaller than ``win`` seconds.

    When two events are merged, the label of the longer constituent event is
    kept, preserving the dominant event type.

    Args:
        events: List of ``(start, end)`` tuples.
        win: Maximum gap (in **seconds**) between two events for them to be
            merged.  Internally converted to samples via ``win * Fs``.
        Fs: Sampling frequency in Hz.
        max_dur: If not ``False``, the maximum allowed duration (in seconds)
            for a merged event.  Prevents runaway merging of many small events
            into one giant event.
        labels: Per-event label values.  Defaults to all-ones.

    Returns:
        A tuple ``(new_events, new_labels)`` with the merged event list and
        corresponding labels.
    """
    new_events: list[tuple[int, int]] = []
    new_labels: list[int] = []

    if len(events) > 0:
        cnt = 0
        # Convert the gap threshold from seconds to samples.
        win = win * Fs

        if labels is None:
            labels = [1] * len(events)

        while cnt < len(events) - 1:
            st = events[cnt][0]
            end = events[cnt][1]
            dist_to_next_event = events[cnt + 1][0] - end

            if (dist_to_next_event < win) and (not max_dur or (events[cnt + 1][1] - st < max_dur * Fs)):
                # Merge the two events into one spanning from the first start
                # to the second end.
                new_events.append((st, events[cnt + 1][1]))
                # The label of the longer event wins.
                label = labels[cnt] if (end - st) > (events[cnt + 1][1] - events[cnt + 1][0]) else labels[cnt + 1]
                new_labels.append(label)
                cnt += 2
            else:
                new_events.append((st, end))
                new_labels.append(labels[cnt])
                cnt += 1

        # The while loop exits when cnt reaches len(events) - 1 (last
        # event not yet visited) OR len(events) (last event was already
        # consumed by a merge on the previous iteration).  Only append
        # when the last event was NOT already consumed.
        if cnt == len(events) - 1:
            new_events.append(events[-1])
            new_labels.append(labels[-1])

    return new_events, new_labels
