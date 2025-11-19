# HSR LiDAR Mesh Processing Package

A ROS Noetic package for activating LiDAR sensors on the Toyota HSR robot, generating meshes from point cloud data, and applying Hough transforms for feature detection.

## 📋 Overview

This package provides three main capabilities:
1. **LiDAR Activation**: Subscribes to HSR LiDAR sensor and processes scan data
2. **Mesh Generation**: Converts point cloud data into 3D meshes using Delaunay triangulation
3. **Hough Transform**: Detects lines and geometric features in LiDAR data

## 📦 Package Contents

```
hsr_lidar_mesh/
├── CMakeLists.txt              # Build configuration
├── package.xml                 # Package metadata and dependencies
├── README.md                   # This file
├── INSTRUCTIONS.md             # Complete ROS project guide
├── requirements.txt            # Python dependencies
├── csc647-final-project-start.py  # Main startup script
├── launch/                     # Launch files
│   ├── lidar_mesh_processing.launch      # Main launch file
│   ├── lidar_mesh_with_isaac.launch      # For Isaac Sim integration
│   └── visualization_only.launch         # RViz only
├── scripts/                    # Python nodes (executable)
│   ├── lidar_activator.py                # LiDAR activation node
│   ├── point_cloud_to_mesh.py            # Mesh generation node
│   └── hough_transform_processor.py      # Hough transform node
├── config/                     # Configuration files
│   └── lidar_params.yaml                 # Node parameters
└── rviz/                       # RViz configurations
    └── lidar_mesh_visualization.rviz     # Visualization config
```

## 🔧 Dependencies

### ROS Packages
- `rospy` - Python ROS interface
- `roscpp` - C++ ROS interface
- `sensor_msgs` - Sensor message types
- `geometry_msgs` - Geometry message types
- `visualization_msgs` - Visualization markers
- `nav_msgs` - Navigation messages
- `pcl_ros` - Point Cloud Library ROS interface
- `pcl_conversions` - PCL conversions
- `tf` - Transform library
- `tf2_ros` - TF2 library
- `laser_geometry` - Laser scan to point cloud conversion

### Python Packages
- `numpy` - Numerical computing
- `scipy` - Scientific computing (Delaunay triangulation)
- `opencv-python` - Computer vision (Hough transform)

## 📥 Installation

### For CSC 647 Students (Standard Setup)

This package is located at: `~/hsr_robocanes_omniverse/src/csc647-final-project`

Your catkin workspace is already set up at `~/hsr_robocanes_omniverse`, so the package is in the right place!

### 1. Install Dependencies

```bash
# Navigate to package directory
cd ~/hsr_robocanes_omniverse/src/csc647-final-project

# Install ROS dependencies
sudo apt-get update
sudo apt-get install -y \
    ros-noetic-pcl-ros \
    ros-noetic-pcl-conversions \
    ros-noetic-laser-geometry \
    ros-noetic-visualization-msgs \
    python3-numpy \
    python3-scipy

# Install Python dependencies
pip3 install -r requirements.txt
```

### 2. Build the Package

```bash
# If you have aliases set up, just use:
c    # Compiles and sources

# Or manually:
cd ~/hsr_robocanes_omniverse
catkin_make
source devel/setup.bash

# Verify installation
rospack find csc647-final-project
```

## 🚀 Usage

### Quick Start

#### Option 1: Standalone Launch (No Isaac Sim)
```bash
# Terminal 1: Start ROS core
roscore

# Terminal 2: Launch the package
s    # Source the workspace first
roslaunch csc647-final-project lidar_mesh_processing.launch
```

#### Option 2: With Isaac Sim
```bash
# Terminal 1: Start ROS core
roscore

# Terminal 2: Start Isaac Sim
cd ~/hsr_robocanes_omniverse
python3 csc647-final-project-start.py

# Terminal 3: Launch LiDAR mesh processing
s    # Source the workspace
roslaunch csc647-final-project lidar_mesh_with_isaac.launch
```

### Running Individual Nodes

```bash
# Make sure workspace is sourced first
s

# LiDAR Activator only
rosrun csc647-final-project lidar_activator.py

# Mesh Generator only
rosrun csc647-final-project point_cloud_to_mesh.py

# Hough Transform only
rosrun csc647-final-project hough_transform_processor.py

# RViz visualization
roslaunch csc647-final-project visualization_only.launch
```

### Launch File Arguments

```bash
# Customize scan topic
roslaunch csc647-final-project lidar_mesh_processing.launch scan_topic:=/custom/scan

# Disable visualization
roslaunch csc647-final-project lidar_mesh_processing.launch enable_visualization:=false

# Disable Hough transform
roslaunch csc647-final-project lidar_mesh_processing.launch enable_hough:=false
```

## 📡 Topics

### Subscribed Topics
- `/hsrb/base_scan` (sensor_msgs/LaserScan) - Raw LiDAR data from HSR robot

### Published Topics
- `/lidar/scan_processed` (sensor_msgs/LaserScan) - Filtered LiDAR scan data
- `/lidar/point_cloud` (sensor_msgs/PointCloud2) - Converted point cloud
- `/lidar/mesh` (visualization_msgs/Marker) - Generated mesh visualization
- `/lidar/mesh_triangles` (visualization_msgs/MarkerArray) - Triangle mesh edges
- `/lidar/detected_lines` (visualization_msgs/MarkerArray) - Lines detected by Hough transform
- `/lidar/hough_grid` (nav_msgs/OccupancyGrid) - Occupancy grid for Hough space

## 🔧 Configuration

Edit `config/lidar_params.yaml` to customize parameters:

```yaml
# LiDAR range filtering
lidar_activator:
  min_range: 0.1      # meters
  max_range: 30.0     # meters

# Mesh generation
point_cloud_to_mesh:
  voxel_size: 0.05          # meters (downsampling)
  max_distance: 5.0         # meters (max range for meshing)
  mesh_alpha: 0.7           # transparency (0.0 to 1.0)

# Hough transform
hough_transform_processor:
  grid_resolution: 0.05     # meters
  hough_threshold: 50       # minimum votes
  min_line_length: 0.5      # meters
  max_line_gap: 0.3         # meters
```

## 📊 Visualization in RViz

The package includes a pre-configured RViz setup. When launched, you'll see:

1. **Grid** - Reference coordinate frame
2. **LaserScan** - Raw LiDAR scan (white points)
3. **PointCloud2** - Processed point cloud (colored points)
4. **Mesh** - Generated 3D mesh (cyan surface)
5. **MeshTriangles** - Triangle edges (red lines)
6. **DetectedLines** - Lines from Hough transform (green lines)
7. **HoughGrid** - Occupancy grid
8. **TF** - Coordinate transforms

### RViz Tips
- Use the mouse to rotate, pan, and zoom the view
- Toggle displays on/off in the Displays panel
- Adjust transparency and colors in each display's properties
- Save your custom configuration: File → Save Config As

## 🧪 Testing

### Check Topics
```bash
# List all topics
rostopic list

# Echo point cloud data
rostopic echo /lidar/point_cloud -n 1

# Check publishing rate
rostopic hz /lidar/mesh
```

### Monitor Nodes
```bash
# List active nodes
rosnode list

# Get node info
rosnode info /lidar_activator
rosnode info /point_cloud_to_mesh
rosnode info /hough_transform_processor
```

### Verify Data Flow
```bash
# Check computation graph
rqt_graph

# View logs
rqt_console
```

## 🐛 Troubleshooting

### Issue: "Package not found"
**Solution:**
```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
rospack find hsr_lidar_mesh
```

### Issue: "No messages on /hsrb/base_scan"
**Solution:**
- Ensure Isaac Sim is running and robot is loaded
- Check topic name: `rostopic list | grep scan`
- Verify topic is publishing: `rostopic hz /hsrb/base_scan`

### Issue: "ImportError: No module named laser_geometry"
**Solution:**
```bash
sudo apt-get install ros-noetic-laser-geometry
```

### Issue: "ImportError: No module named cv2"
**Solution:**
```bash
pip3 install opencv-python
```

### Issue: "Mesh not visible in RViz"
**Solution:**
- Check if topic is publishing: `rostopic echo /lidar/mesh -n 1`
- Verify Fixed Frame in RViz is set to `base_link`
- Ensure mesh transparency (alpha) is not 0.0
- Check if points are within viewing range

### Issue: "Hough transform not detecting lines"
**Solution:**
- Adjust `hough_threshold` parameter (lower value = more sensitive)
- Increase `grid_size` for better resolution
- Check if point cloud has sufficient data: `rostopic echo /lidar/point_cloud -n 1`

## 📚 Node Details

### lidar_activator.py
Activates and processes LiDAR sensor data.
- Filters invalid range readings
- Converts LaserScan to PointCloud2
- Publishes processed data for downstream nodes

### point_cloud_to_mesh.py
Generates 3D meshes from point cloud data.
- Applies voxel grid downsampling for efficiency
- Uses Delaunay triangulation for mesh generation
- Publishes mesh markers for RViz visualization

### hough_transform_processor.py
Detects geometric features using Hough transform.
- Creates occupancy grid from point cloud
- Applies edge detection and Hough line detection
- Publishes detected lines as visualization markers

## 🎓 Learning Resources

### ROS Documentation
- [ROS Noetic Tutorials](http://wiki.ros.org/ROS/Tutorials)
- [sensor_msgs](http://docs.ros.org/en/noetic/api/sensor_msgs/html/index-msg.html)
- [visualization_msgs](http://docs.ros.org/en/noetic/api/visualization_msgs/html/index-msg.html)
- [RViz User Guide](http://wiki.ros.org/rviz/UserGuide)

### Algorithms
- [Delaunay Triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation)
- [Hough Transform](https://en.wikipedia.org/wiki/Hough_transform)
- [Point Cloud Processing](http://pointclouds.org/)

## 📝 Development

### Adding New Features

1. Create new Python script in `scripts/`:
```bash
cd ~/catkin_ws/src/hsr_lidar_mesh/scripts
touch my_new_node.py
chmod +x my_new_node.py
```

2. Add to CMakeLists.txt:
```cmake
catkin_install_python(PROGRAMS
  scripts/my_new_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
```

3. Rebuild:
```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

### Modifying Parameters

Edit `config/lidar_params.yaml` and restart nodes:
```bash
rosnode kill -a
roslaunch hsr_lidar_mesh lidar_mesh_processing.launch
```

## 📄 License

MIT License - See LICENSE file for details

## 👥 Contributors

CSC 647 - Machine Intelligence
Florida International University

## 🔗 Related Packages

- [hsr_isaac_localization](https://github.com/...) - HSR localization with Isaac Sim
- [robot_localization](http://docs.ros.org/kinetic/api/robot_localization/html/) - EKF/UKF for sensor fusion

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review INSTRUCTIONS.md for detailed ROS commands
3. Consult ROS documentation
4. Contact course instructor or TA

---

**Last Updated:** November 2025
**ROS Version:** Noetic
**Platform:** Ubuntu 20.04
