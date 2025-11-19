# Computational Geometry Documentation
## LiDAR Mesh Processing Package

**Course**: Computational Geometry
**Algorithms Used**: Delaunay Triangulation, Hough Transform
**Application**: 3D Environment Reconstruction from Point Clouds

---

## 📐 Computational Geometry Algorithms

### 1. **Delaunay Triangulation** (`point_cloud_to_mesh.py`)

**Location**: Line 226
```python
tri = Delaunay(points_2d)  # 2D Delaunay on XY projection
```

#### Theory:
- **Definition**: Creates a triangulation where no point is inside the circumcircle of any triangle
- **Optimality**: Maximizes the minimum angle of all triangles (avoids "skinny" triangles)
- **Dual**: Voronoi diagram (each Delaunay edge crosses a Voronoi edge perpendicularly)

#### Implementation:
1. Project 3D points (x, y, z) onto 2D plane (x, y)
2. Compute 2D Delaunay triangulation on XY projection
3. Lift triangles back to 3D using original Z-coordinates
4. Apply quality filters:
   - Edge length: max 0.35m (prevents spanning gaps)
   - Triangle area: min 0.0005m² (removes degenerates)
   - Z-variation: max 0.4m (avoids connecting different heights)

#### Complexity:
- **Time**: O(n log n) for 2D Delaunay
- **Space**: O(n) for storing triangles

#### Why Delaunay?
- **Quality**: Produces well-shaped triangles (no thin/degenerate triangles)
- **Speed**: Much faster than k-NN approach (O(n log n) vs O(n²))
- **Coverage**: Automatically connects all points without gaps

---

### 2. **Hough Transform** (`hough_transform_processor.py`)

**Location**: Line 85
```python
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                        threshold=self.hough_threshold, ...)
```

#### Theory:
- **Definition**: Parameter space transformation for line detection
- **Concept**: Each point (x, y) in image space maps to a sinusoid in parameter space (ρ, θ)
- **Detection**: Lines appear as intersection peaks in Hough space

#### Implementation:
1. Convert point cloud to 2D occupancy grid
2. Apply Canny edge detection to find boundaries
3. Probabilistic Hough Line Transform detects line segments
4. Publish detected lines as visualization markers

#### Parameters:
- **ρ (rho)**: Distance resolution = 1 pixel
- **θ (theta)**: Angle resolution = 1 degree (π/180)
- **Threshold**: 30 votes minimum
- **Min line length**: 0.3m
- **Max line gap**: 0.5m

#### Applications:
- **Wall detection**: Identifies planar surfaces in environment
- **Line primitives**: Extracts geometric features for mapping
- **Navigation**: Detects obstacles and boundaries

---

## 🎨 Color Coding Scheme

### Surface Classification (Normal Vector Analysis)

The mesh uses surface normals to classify and color geometric surfaces:

#### **Normal Vector Formula:**
For triangle with vertices p₀, p₁, p₂:
```
v₁ = p₁ - p₀
v₂ = p₂ - p₀
normal = v₁ × v₂  (cross product)
normal = normal / ||normal||  (normalize)
```

#### **Horizontal Surfaces** (|normal.z| > 0.7)

| Color | RGB | Surface Type | Normal Direction | Height Range | Meaning |
|-------|-----|--------------|------------------|--------------|---------|
| **🟢 GREEN** | (0.1, 0.7-1.0, 0.1) | Floors, tables, platforms | Upward (nz > 0) | Variable | Walkable surfaces |
| **🔵 BLUE** | (0.3, 0.5, 1.0) | Ceilings, overhangs | Downward (nz < 0) | Top level | Overhead obstacles |

**Green Intensity Gradient**: Darker green = lower elevation, Brighter green = higher elevation

#### **Vertical Surfaces** (|normal.z| ≤ 0.7) - Walls

4-tier height-based rainbow encoding:

| Color | RGB | Height Ratio | Typical Height | Level |
|-------|-----|--------------|----------------|-------|
| **🔴 RED** | (1.0, 0.1, 0.1) | 0% - 25% | 0.0m - 0.7m | Floor level, baseboards |
| **🟠 ORANGE** | (1.0, 0.65, 0.1) | 25% - 50% | 0.7m - 1.5m | Waist/table height |
| **🟡 YELLOW** | (1.0, 1.0, 0.1) | 50% - 75% | 1.5m - 2.2m | Eye/door height |
| **🔵 CYAN** | (0.1, 0.9, 0.9) | 75% - 100% | 2.2m - 3.0m | Upper walls, ceiling level |

**Purpose**: Rainbow gradient provides intuitive depth perception of wall heights

---

## 🌍 Real-World Applications

### 1. **Autonomous Navigation**
**Use Case**: Mobile robots, self-driving cars, drones
- **Floor Detection** (green surfaces): Identifies drivable/walkable areas
- **Obstacle Detection** (walls): Red/orange/yellow walls indicate vertical obstacles
- **Ceiling Clearance** (blue surfaces): Ensures safe height for flying robots
- **Path Planning**: Uses mesh geometry to compute collision-free paths

**Example**: Warehouse robot navigating between shelves, avoiding obstacles

---

### 2. **3D Mapping & SLAM**
**Use Case**: Indoor mapping, construction site scanning
- **Real-time 3D reconstruction**: Creates textured 3D models of environments
- **Loop closure detection**: Uses Hough-detected lines for geometric matching
- **Semantic mapping**: Color-coded surfaces label floor/wall/ceiling
- **Change detection**: Compare meshes over time to detect modifications

**Example**: Building inspection robot creating as-built models

---

### 3. **Assistive Technology**
**Use Case**: Navigation aids for visually impaired, wheelchair navigation
- **Obstacle warning**: Red walls indicate low obstacles (trip hazards)
- **Terrain analysis**: Green gradient shows slope/elevation changes
- **Door detection**: Hough lines identify doorways and passages
- **Audio feedback**: Convert colors to audible tones (red=low, cyan=high)

**Example**: Smart cane detecting stairs and obstacles ahead

---

### 4. **Virtual/Augmented Reality**
**Use Case**: AR overlays, VR environment creation
- **Real-time scene reconstruction**: Mesh provides geometry for AR object placement
- **Occlusion handling**: Mesh surfaces determine if virtual objects are hidden
- **Physics simulation**: Use mesh as collision geometry for virtual objects
- **Mixed reality**: Align virtual and real worlds using detected planes

**Example**: AR furniture placement app showing sofa on detected floor

---

### 5. **Construction & Architecture**
**Use Case**: As-built verification, volume measurement
- **Floor area calculation**: Integrate green surface areas
- **Wall surface estimation**: Sum vertical surface areas for painting/materials
- **Volume computation**: Use mesh to calculate room volumes
- **Defect detection**: Compare mesh to CAD models (BIM)

**Example**: Verify room dimensions match architectural plans

---

### 6. **Agricultural Robotics**
**Use Case**: Crop monitoring, autonomous harvesting
- **Terrain mapping**: Green surfaces identify ground level for navigation
- **Plant height estimation**: Color gradient shows crop height distribution
- **Row detection**: Hough lines identify crop rows for guidance
- **Obstacle avoidance**: Detect trees, posts, equipment

**Example**: Autonomous tractor navigating field rows

---

### 7. **Security & Surveillance**
**Use Case**: Intrusion detection, crowd monitoring
- **3D scene understanding**: Mesh provides geometric context
- **Height estimation**: Wall colors estimate person height
- **Change detection**: Alert when new objects appear in mesh
- **Coverage planning**: Ensure cameras cover all surfaces

**Example**: Security robot detecting unauthorized objects in restricted areas

---

### 8. **Search & Rescue**
**Use Case**: Disaster response, cave exploration
- **Safe path identification**: Green surfaces show stable ground
- **Structural assessment**: Mesh reveals damaged walls/ceilings
- **Victim location**: Map environment to coordinate rescue efforts
- **3D visualization**: Remote operators see environment mesh

**Example**: Robot exploring collapsed building, mapping safe routes

---

## 🔧 Technical Specifications

### Performance Metrics
- **Input**: RGB-D point cloud (~10,000-50,000 points)
- **Sampling**: 3000 points for triangulation
- **Output**: Up to 10,000 triangles
- **Processing Time**: ~50-100ms per frame
- **Quality Filters**: 3 criteria (edge, area, Z-variation)

### Parameters (Tunable)
```python
# Mesh Generation
voxel_size = 0.03          # Downsampling resolution (meters)
max_distance = 3.5         # Maximum point distance (meters)
max_edge_length = 0.35     # Max triangle edge (meters)
max_z_variation = 0.4      # Max height difference per triangle (meters)
min_triangle_area = 0.0005 # Min triangle area (m²)

# Hough Transform
grid_resolution = 0.05     # Occupancy grid cell size (meters)
hough_threshold = 30       # Minimum votes for line
min_line_length = 0.3      # Minimum line length (meters)
```

### ROS Topics
| Topic | Type | Description |
|-------|------|-------------|
| `/hsrb/head_rgbd_sensor/depth_registered/rectified_points` | PointCloud2 | Input RGB-D data |
| `/lidar/mesh` | Marker | Triangular mesh output |
| `/lidar/detected_lines` | MarkerArray | Hough-detected lines |
| `/lidar/hough_grid` | OccupancyGrid | 2D occupancy grid |

---

## 📚 Academic Relevance

### Computational Geometry Concepts Demonstrated:

1. **Delaunay Triangulation**
   - Voronoi duals
   - Optimal triangulation
   - Convex hull relationships

2. **Hough Transform**
   - Parameter space mapping
   - Line detection
   - Geometric feature extraction

3. **Surface Normal Computation**
   - Cross product for plane normals
   - Normal vector analysis
   - Surface orientation classification

4. **Point Cloud Processing**
   - Voxel grid downsampling
   - Distance filtering
   - Spatial data structures (KD-trees)

5. **Quality Metrics**
   - Triangle aspect ratio
   - Edge length constraints
   - Degenerate case handling

---

## 🚀 Running the Package

```bash
# 1. Build
cd ~/hsr_robocanes_omniverse
catkin_make install
source devel/setup.bash

# 2. Launch with Isaac Sim
roslaunch csc647-final-project lidar_mesh_with_isaac.launch

# 3. Visualize in RViz
rviz -d ~/hsr_robocanes_omniverse/src/csc647-final-project/rviz/lidar_mesh_visualization.rviz
```

---

## 📖 References

1. **Delaunay Triangulation**
   - de Berg et al., "Computational Geometry: Algorithms and Applications"
   - Shewchuk, J.R., "Delaunay refinement algorithms for triangular mesh generation"

2. **Hough Transform**
   - Duda, R.O., Hart, P.E., "Use of the Hough transformation to detect lines and curves in pictures"
   - Illingworth, J., Kittler, J., "A survey of the Hough transform"

3. **Point Cloud Processing**
   - Rusu, R.B., "Semantic 3D Object Maps for Everyday Manipulation in Human Living Environments"
   - Schnabel, R., et al., "Efficient RANSAC for Point-Cloud Shape Detection"

---

**Author**: Computational Geometry Project
**Date**: 2025
**Framework**: ROS Noetic, Python 3, SciPy, OpenCV
