#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


class _CornerBuilderHelper:

    def __init__(self):
        self.corners = []
        self.ids = []
        self.sizes = []
        self.max_id = 0

        self.WINDOW_SIZE = (10, 10)
        self.MAX_LEVEL = 1
        self.MAX_CORNERS = 50000
        self.MAX_CORNERS_PER_FRAME = 500
        self.MIN_DIST = 10
        self.BLOCK_SIZE = 10

    # Creates mask that helps to avoid too close corners
    def create_mask(self, h, w):
        mask = np.full(shape=(h, w), fill_value=255, dtype=np.uint8)
        for (x, y), radius in zip(self.corners, self.sizes):
            mask = cv2.circle(mask, (np.round(x).astype(int), np.round(y).astype(int)), 4 * radius, color=0)
        return mask

    # Creates an optical flow to find corners from the previous frame on the next one
    def add_prev_corners_tracks(self, prev_frame, cur_frame):
        new_corners, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, cur_frame,
                                                          np.asarray(self.corners, dtype=np.float32), None,
                                                          winSize=self.WINDOW_SIZE,
                                                          criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 0.01),
                                                          maxLevel=self.MAX_LEVEL, minEigThreshold=1e-4, flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS)

        status = status.ravel()
        self.corners = new_corners[status == 1].tolist()
        self.sizes = np.asarray(self.sizes)[status == 1].tolist()
        self.ids = np.asarray(self.ids)[status == 1].tolist()

    # Find new corners on the current frame. Use mask to avoid corners too close to existing ones
    def add_new_corners(self, frame):

        mask = self.create_mask(frame.shape[0], frame.shape[1])
        _, pyramid = cv2.buildOpticalFlowPyramid(frame, self.WINDOW_SIZE, self.MAX_LEVEL, None, False)
        _, mask_pyramid = cv2.buildOpticalFlowPyramid(mask, self.WINDOW_SIZE, self.MAX_LEVEL, None, False)

        for level, level_frame in enumerate(pyramid):
            if len(self.corners) >= self.MAX_CORNERS:
                return
            num = min(self.MAX_CORNERS - len(self.corners), self.MAX_CORNERS_PER_FRAME)
            new_corners = cv2.goodFeaturesToTrack(image=level_frame, maxCorners=num,
                                                  qualityLevel=0.01, minDistance=self.MIN_DIST,
                                                  blockSize=self.BLOCK_SIZE, mask=mask_pyramid[level])
            if new_corners is None:
                continue

            new_corners = new_corners.reshape(-1, 2).astype(np.float32)
            for (x, y) in new_corners:
                self.corners.append((x * 2 ** level, y * 2 ** level))
                self.sizes.append(self.BLOCK_SIZE * 2 ** level)
                self.ids.append(self.max_id)
                self.max_id += 1

    def track_corners(self, prev_frame, cur_frame):
        if len(self.corners) > 0:
            self.add_prev_corners_tracks(prev_frame, cur_frame)

        self.add_new_corners(cur_frame)
        return FrameCorners(np.array(self.ids), np.array(self.corners), np.array(self.sizes))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    corners_helper = _CornerBuilderHelper()

    prev_frame = None
    for i, frame in enumerate(frame_sequence):
        frame = (frame * 256).astype(np.uint8)
        corners = corners_helper.track_corners(prev_frame, frame)
        builder.set_corners_at_frame(i, corners)
        prev_frame = frame


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
