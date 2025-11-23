# CSC647 Final Project - Hough Transform Plane Detection

ROS package for detecting planes and lines in LiDAR data using **Hough Transform**.

## 🎯 Overview

This package demonstrates **Hough Transform** for geometric feature detection in robotic perception. It processes 3D point cloud data from the Toyota HSR robot's RGB-D camera and 2D LiDAR to detect lines and planes in the environment.

## 🔧 Key Algorithm: Hough Transform

**Hough Transform** is a classical computer vision algorithm for detecting geometric shapes:
- Converts points from Cartesian space to parameter space
- Detects lines, circles, and other parametric shapes
- Robust to noise and partial occlusions
- Used in lane detection, object recognition, and SLAM

### How it Works:
1. Create occupancy grid from point cloud
2. Apply edge detection
3. Transform to Hough space (ρ-θ parameterization)
4. Find peaks in accumulator array
5. Convert back to detected lines in world coordinates

## 📦 Package Structure

```
csc647-final-project/
├── scripts/
│   ├── lidar_activator.py              # Converts LaserScan to PointCloud2
│   ├── hough_transform_processor.py    # Hough Transform implementation
│   └── point_cloud_to_mesh.py          # Simple point cloud visualization
├── launch/
│   └── lidar_mesh_with_isaac.launch    # Main launch file
├── rviz/
│   └── hough_transform_visualization.rviz  # RViz configuration
└── README.md
```

## 🚀 Usage

### Launch with Isaac Sim:
```bash
roslaunch csc647-final-project lidar_mesh_with_isaac.launch
```

### Launch without RViz:
```bash
roslaunch csc647-final-project lidar_mesh_with_isaac.launch rviz:=false
```

## 📊 Topics

### Subscribed Topics:
- `/hsrb/base_scan` (sensor_msgs/LaserScan) - 2D LiDAR scan
- `/hsrb/head_rgbd_sensor/depth_registered/rectified_points` (sensor_msgs/PointCloud2) - RGB-D camera point cloud

### Published Topics:
- `/lidar/point_cloud` (sensor_msgs/PointCloud2) - Converted point cloud from LaserScan
- `/lidar/detected_lines` (visualization_msgs/MarkerArray) - Detected lines via Hough Transform
- `/lidar/hough_grid` (nav_msgs/OccupancyGrid) - Occupancy grid for Hough space
- `/lidar/point_cloud_viz` (visualization_msgs/Marker) - Simple point cloud visualization
- `/lidar/stats` (std_msgs/String) - Processing statistics

## 🎓 Computational Geometry Algorithms

### Hough Transform
- **File**: `hough_transform_processor.py`
- **Purpose**: Detect lines and planes in point cloud data
- **Complexity**: O(n × m) where n = points, m = angle discretization
- **Applications**:
  - Wall detection
  - Corridor recognition
  - Plane segmentation
  - Feature extraction for SLAM

**Parameters:**
- `grid_resolution`: 0.05m (grid cell size)
- `grid_size`: 300 cells (occupancy grid dimensions)
- `hough_threshold`: 30 votes (minimum for line detection)
- `min_line_length`: 0.3m (minimum line length)
- `max_line_gap`: 0.5m (maximum gap in line segments)

## 📈 Visualization

The RViz configuration shows:
- **DetectedLines**: Lines found by Hough Transform (colored markers)
- **HoughGrid**: Occupancy grid representation
- **PointCloud2**: Raw RGB-D point cloud
- **LaserScan**: 2D LiDAR visualization
- **Camera**: RGB camera feed overlay

## 🔬 Technical Details

### Hough Transform Algorithm:
```
1. Create occupancy grid from point cloud
   - Project 3D points to 2D grid
   - Mark occupied cells

2. Apply probabilistic Hough line detection
   - Use OpenCV's HoughLinesP
   - Parameters: rho=1, theta=π/180, threshold

3. Filter and extend line segments
   - Merge nearby collinear segments
   - Remove short lines

4. Convert to world coordinates
   - Transform from grid to robot frame
   - Publish as MarkerArray
```

### Performance:
- Real-time processing at 10-30 Hz
- Handles 10,000+ points per frame
- Low computational overhead

## 🏗️ Dependencies

- ROS Noetic
- Python 3
- NumPy
- SciPy
- OpenCV (cv2)
- sensor_msgs
- visualization_msgs
- nav_msgs

## 📝 Notes

- Focused implementation using only **Hough Transform** for plane detection
- Simplified from previous versions that included Delaunay, RANSAC, and Convex Hull
- Designed for Toyota HSR robot with RGB-D camera and 2D LiDAR
- Compatible with Isaac Sim simulation

## 🎯 Applications

1. **Navigation**: Detect walls and corridors for path planning
2. **Mapping**: Extract geometric features for SLAM
3. **Object Recognition**: Identify planar surfaces (doors, tables, walls)
4. **Scene Understanding**: Semantic segmentation of indoor environments

## 👥 Author

CSC647 Final Project
Focus: Hough Transform for Plane Detection in Robotic Perception
