"""
Tests for hlg.core.events -- the most widely used utility module.

These tests verify the event detection, conversion, and merging primitives
that underpin every stage of the HLG pipeline. Since downstream modules
(sleep_metrics, ventilation, segmentation, etc.) all depend on correct
event detection, these tests serve as the first line of defense.
"""

from __future__ import annotations

import numpy as np

from hlg.core.events import (
    connect_events,
    events_to_array,
    find_events,
    window_correction,
)


# ── find_events ──────────────────────────────────────────────────────────


class TestFindEvents:
    """Tests for the core event-detection function."""

    def test_empty_signal_returns_no_events(self):
        """A signal of all zeros should yield no events."""
        sig = np.zeros(100)
        assert find_events(sig) == []

    def test_single_event_in_middle(self):
        """A single contiguous non-zero region should be detected."""
        sig = np.zeros(100)
        sig[20:40] = 1
        events = find_events(sig)
        assert len(events) == 1
        assert events[0] == (20, 40)

    def test_multiple_events(self):
        """Multiple separated non-zero regions should each be detected."""
        sig = np.zeros(200)
        sig[10:30] = 1
        sig[50:70] = 1
        sig[100:120] = 1
        events = find_events(sig)
        assert len(events) == 3
        assert events[0] == (10, 30)
        assert events[1] == (50, 70)
        assert events[2] == (100, 120)

    def test_event_at_start(self):
        """An event starting at index 0 should be captured."""
        sig = np.zeros(100)
        sig[0:20] = 1
        events = find_events(sig)
        assert len(events) == 1
        assert events[0][0] == 0

    def test_event_at_end(self):
        """An event running to the last sample should be captured."""
        sig = np.zeros(100)
        sig[80:] = 1
        events = find_events(sig)
        assert len(events) == 1
        assert events[0][0] == 80
        assert events[0][1] == 100

    def test_full_signal_active(self):
        """An entirely active signal should yield one event spanning all."""
        sig = np.ones(50)
        events = find_events(sig)
        assert len(events) == 1
        assert events[0] == (0, 50)

    def test_different_label_values(self):
        """Non-zero values other than 1 should still be detected as events."""
        sig = np.zeros(100)
        sig[10:30] = 3
        sig[50:70] = 7
        events = find_events(sig)
        assert len(events) == 2


# ── events_to_array ──────────────────────────────────────────────────────


class TestEventsToArray:
    """Tests for converting event lists back to label arrays."""

    def test_roundtrip_with_find_events(self):
        """Converting events back to an array should reconstruct the original."""
        sig = np.zeros(100)
        sig[20:40] = 1
        sig[60:80] = 1
        events = find_events(sig)
        reconstructed = events_to_array(events, len(sig))
        np.testing.assert_array_equal(sig, reconstructed)

    def test_custom_labels(self):
        """Labels should be applied correctly to each event region."""
        events = [(10, 30), (50, 70)]
        labels = [2, 5]
        arr = events_to_array(events, 100, labels=labels)
        assert arr[20] == 2
        assert arr[60] == 5
        assert arr[0] == 0
        assert arr[40] == 0

    def test_empty_events(self):
        """An empty event list should produce an all-zero array."""
        arr = events_to_array([], 50)
        np.testing.assert_array_equal(arr, np.zeros(50))


# ── window_correction ────────────────────────────────────────────────────


class TestWindowCorrection:
    """Tests for the half-window event extension."""

    def test_extends_events(self):
        """Events should be widened by half_window on each side."""
        sig = np.zeros(100)
        sig[40:60] = 1
        corrected = window_correction(sig, window_size=10)
        # Original event: [40, 60). After correction: [35, 65)
        assert corrected[35] == 1
        assert corrected[64] == 1
        assert corrected[34] == 0

    def test_boundary_clipping(self):
        """Extension near signal boundaries should be clipped, not error."""
        sig = np.zeros(50)
        sig[0:5] = 1
        corrected = window_correction(sig, window_size=20)
        # Should extend right but clip left at 0
        assert corrected[0] == 1
        assert corrected[14] == 1


# ── connect_events ───────────────────────────────────────────────────────


class TestConnectEvents:
    """Tests for merging nearby events."""

    def test_close_events_merged(self):
        """Events within the window should be merged into one."""
        events = [(10, 20), (22, 35)]
        merged, labels = connect_events(events, win=1, Fs=10)
        # Gap is 2 samples, window is 1*10=10, so they merge into one.
        assert len(merged) == 1
        assert merged[0] == (10, 35)

    def test_far_events_not_merged(self):
        """Events far apart should remain separate."""
        events = [(10, 20), (50, 60)]
        merged, labels = connect_events(events, win=1, Fs=10)
        assert len(merged) == 2

    def test_empty_events(self):
        """An empty event list should return empty."""
        merged, labels = connect_events([], win=1, Fs=10)
        assert merged == []
        assert labels == []

    def test_labels_follow_longer_event(self):
        """When merging, the label of the longer sub-event should win."""
        events = [(10, 30), (32, 40)]  # first is 20 samples, second is 8
        labels = [1, 2]
        merged, new_labels = connect_events(events, win=1, Fs=10, labels=labels)
        assert len(merged) == 1
        assert merged[0] == (10, 40)
        assert new_labels[0] == 1  # first event was longer (20 > 8 samples)

    def test_three_events_first_two_merge(self):
        """When first two merge but third is far, result has 2 events."""
        events = [(10, 20), (22, 35), (100, 110)]
        merged, labels = connect_events(events, win=1, Fs=10)
        assert len(merged) == 2
        assert merged[0] == (10, 35)
        assert merged[1] == (100, 110)

    def test_three_events_last_two_merge(self):
        """When last two merge but first is far, result has 2 events."""
        events = [(10, 20), (80, 90), (92, 110)]
        merged, labels = connect_events(events, win=1, Fs=10)
        assert len(merged) == 2
        assert merged[0] == (10, 20)
        assert merged[1] == (80, 110)

    def test_single_event_unchanged(self):
        """A single event should be returned as-is."""
        events = [(10, 50)]
        merged, labels = connect_events(events, win=1, Fs=10)
        assert len(merged) == 1
        assert merged[0] == (10, 50)
