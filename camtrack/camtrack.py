#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    TriangulationParameters,
    rodrigues_and_translation_to_view_mat3x4,
    eye3x4,
    Correspondences
)


class CloudPointInfo:
    def __init__(self, pos, num_inliers):
        self.pos = pos
        self.num_inliers = num_inliers


class TrackedPoseInfo:
    def __init__(self, pos, num_inliers):
        self.pos = pos
        self.num_inliers = num_inliers


class CameraTracker:

    def __init__(self, int_cam_params, corner_storage, known_view_1, known_view_2, num_frames):
        self.num_defined_poses = 2

        self.int_cam_params = int_cam_params
        self.corner_storage = corner_storage
        self.num_frames = num_frames
        self.point_cloud = {}
        self.tracked_poses = [None] * self.num_frames

        if known_view_1 is None or known_view_2 is None:
            known_view_frame_1, known_view_frame_2 = self._initialize_camtrack_()
        else:
            known_view_frame_1 = known_view_1[0]
            known_view_frame_2 = known_view_2[0]
            self.tracked_poses[known_view_1[0]] = TrackedPoseInfo(pose_to_view_mat3x4(known_view_1[1]), np.inf)
            self.tracked_poses[known_view_2[0]] = TrackedPoseInfo(pose_to_view_mat3x4(known_view_2[1]), np.inf)

        pts_cloud, ids, _ = self._triangulate(known_view_frame_1, known_view_frame_2, True)
        self._update_points_cloud(pts_cloud, ids)

    def _initialize_camtrack_(self,
                              rec_level = 1,
                              max_reproj_error=1.0,
                              min_triang_angle_degree=5.0,
                              min_triang_depth=10,
                              max_rec_level=5):
        print(f'Trying to find initial positions, attempt {rec_level}')
        best_frame_1, best_frame_2 = None, None
        m1 = eye3x4()
        best_m2 = None
        best_num_3d_points = 0

        for frame_1 in range(0, len(self.corner_storage), int(self.num_frames / 10)):
            frame_1_corners = self.corner_storage[frame_1]
            for frame_2 in range(frame_1 + 1, len(self.corner_storage), int(self.num_frames / 10)):
                frame_2_corners = self.corner_storage[frame_2]
                corresp = build_correspondences(frame_1_corners, frame_2_corners)
                essential_mat, inliers_mask = cv2.findEssentialMat(corresp.points_1, corresp.points_2,
                                                                   self.int_cam_params, method=cv2.RANSAC,
                                                                   threshold=max_reproj_error)
                if essential_mat is None or inliers_mask.sum() == 0:
                    continue

                _, R, t, inliers_mask_recovered = cv2.recoverPose(essential_mat, corresp.points_1, corresp.points_2,
                                                                  self.int_cam_params, mask=inliers_mask)

                if inliers_mask_recovered.sum() == 0:
                    continue
                #print(corresp.ids.shape, inliers_mask_recovered.shape)

                m2 = np.hstack((R, t))
                points_3d, _, _ = triangulate_correspondences(
                    Correspondences(
                        corresp.ids[inliers_mask_recovered.flatten() == 1],
                        corresp.points_1[inliers_mask_recovered.flatten() == 1],
                        corresp.points_2[inliers_mask_recovered.flatten() == 1]),
                    view_mat_1=m1, view_mat_2=m2,
                    intrinsic_mat=self.int_cam_params,
                    parameters=TriangulationParameters(max_reproj_error, min_triang_angle_degree, min_triang_depth)
                )
                if best_frame_1 is None or best_frame_2 is None or len(points_3d) > best_num_3d_points:
                    best_frame_1 = frame_1
                    best_frame_2 = frame_2
                    best_m2 = m2.copy()
                    best_num_3d_points = len(points_3d)

        if best_frame_1 is None or best_frame_2 is None:
            print("Fail to find positions good enough")
            if rec_level > max_rec_level:
                raise("Can't find initial camera positions")
            return self._initialize_camtrack_(rec_level+1, max_reproj_error * 1.2,
                                              min_triang_angle_degree=min_triang_angle_degree * 0.8)

        self.tracked_poses[best_frame_1] = TrackedPoseInfo(m1, best_num_3d_points)
        self.tracked_poses[best_frame_2] = TrackedPoseInfo(best_m2, best_num_3d_points)
        return best_frame_1, best_frame_2


    def _update_points_cloud(self, pts_cloud, ids):
        inl = np.ones_like(ids)
        for pt, pt_id in zip(pts_cloud, ids):
            if pt_id not in self.point_cloud.keys():
                self.point_cloud[pt_id] = CloudPointInfo(pt, inl)

    def _triangulate(self, frame_num_1, frame_num_2, initial_triangulation=False):
        corr = build_correspondences(self.corner_storage[frame_num_1], self.corner_storage[frame_num_2],
                                     ids_to_remove=np.array(list(map(int, self.point_cloud.keys())), dtype=int))
        if len(corr) == 0:
            return None, None

        max_reproj_error = 1.0
        min_angle = 5.0

        while True:
            pts_3d, triangulation_ids, med_cos = triangulate_correspondences(corr, self.tracked_poses[frame_num_1].pos,
                                                                             self.tracked_poses[frame_num_2].pos,
                                                                             self.int_cam_params,
                                                                             TriangulationParameters(max_reproj_error,
                                                                                                     min_angle, 0))
            if not initial_triangulation or len(pts_3d) > 20:
                break

            max_reproj_error *= 1.5
            min_angle /= 1.5

        return pts_3d, triangulation_ids, med_cos

    def _get_pos(self, num_frame):
        all_corners = self.corner_storage[num_frame]

        known_corners, known_3d_pts = [], []

        for corner_id, corner in zip(all_corners.ids.flatten(), all_corners.points):
            if corner_id in self.point_cloud.keys():
                known_corners.append(corner)
                known_3d_pts.append(self.point_cloud[corner_id].pos)

        if len(known_corners) < 4:
            return None

        known_corners, known_3d_pts = np.array(known_corners), np.array(known_3d_pts)
        repr_error = 5

        for _ in range(5):
            is_success, r_vec, t_vec, inl = cv2.solvePnPRansac(known_3d_pts, known_corners, self.int_cam_params, None,
                                                               flags=cv2.SOLVEPNP_EPNP, reprojectionError=repr_error)
            if is_success:
                break
            repr_error *= 2

        if not is_success:
            return None

        is_success, new_r_vec, new_t_vec, new_inl = cv2.solvePnPRansac(known_3d_pts[inl], known_corners[inl], self.int_cam_params,
                                                                       None, r_vec, t_vec, useExtrinsicGuess=True)

        if is_success:
            return new_r_vec, new_t_vec, len(new_inl)

        return r_vec, t_vec, len(inl)

    def _find_best_frame(self, undef_frames):
        new_poses_info = []
        for frame, found_pos_info in zip(undef_frames, map(self._get_pos, undef_frames)):
            if found_pos_info is not None:
                new_poses_info.append((frame, found_pos_info))

        if len(new_poses_info) == 0:
            raise NotImplementedError("Cannot get more camera positions")

        best_frame = new_poses_info[0][0]
        best_frame_pose_info = new_poses_info[0][1]

        for frame, pos_info in new_poses_info:
            if best_frame_pose_info[2] < pos_info[2]:
                best_frame = frame
                best_frame_pose_info = pos_info

        return best_frame, best_frame_pose_info

    def track(self):
        while self.num_defined_poses != self.num_frames:
            print(f'{self.num_defined_poses} out of {self.num_frames} frames detected')
            undef_frames = [i for i in range(self.num_frames) if self.tracked_poses[i] is None]

            best_frame, best_frame_pose_info = self._find_best_frame(undef_frames)

            self.tracked_poses[best_frame] = TrackedPoseInfo(
                rodrigues_and_translation_to_view_mat3x4(best_frame_pose_info[0], best_frame_pose_info[1]),
                best_frame_pose_info[2])

            print(f'Pose for {best_frame} frame detected')

            best_med_cos = 1
            best_pts_cloud = None
            best_ids = None

            for i in range(max(0, best_frame - 100), min(self.num_frames, best_frame + 100), 10):
                if i in undef_frames:
                    continue
                pts_cloud, ids, med_cos = self._triangulate(best_frame, i)
                if med_cos < best_med_cos:
                    best_med_cos = med_cos
                    best_ids = ids
                    best_pts_cloud = pts_cloud

            if best_pts_cloud is not None:
                self._update_points_cloud(best_pts_cloud, best_ids)

            self.num_defined_poses += 1
            print(f'Current points cloud size: {len(self.point_cloud)}')

        ids, cloud_points = [], []
        for pt_id, cloud_pt_info in self.point_cloud.items():
            ids.append(pt_id)
            cloud_points.append(cloud_pt_info.pos)

        return list(map(lambda tracked_pos_info: tracked_pos_info.pos, self.tracked_poses)), \
               PointCloudBuilder(np.array(ids), np.array(cloud_points))


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    view_mats, point_cloud_builder = CameraTracker(intrinsic_mat, corner_storage, known_view_1, known_view_2,
                                                   len(rgb_sequence)).track()

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
