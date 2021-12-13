# 3D Camera Tracker
Given the video the following tracker does the following:

1. Detect 2D features using Shi-Tomasi corners detector.
2. Build 2D correspondences using Lucas-Kanade algorithm.
3. Given 2 camera positions and correspondences find other camera positions and cloud of 3D points.
4. Find good enough initial camera positions.

