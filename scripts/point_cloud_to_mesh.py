#!/usr/bin/env python3
"""
Point Cloud to Mesh Converter Node
===================================
Converts LiDAR point cloud data into mesh representations for visualization and analysis.
Uses triangulation and surface reconstruction techniques.

Topics:
    Subscribed:
        /lidar/point_cloud (sensor_msgs/PointCloud2): Input point cloud from LiDAR

    Published:
        /lidar/mesh (visualization_msgs/Marker): Mesh visualization
        /lidar/mesh_triangles (visualization_msgs/MarkerArray): Triangle mesh
"""

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from scipy.spatial import Delaunay


class PointCloudToMesh:
    """Converts point cloud data to mesh representations."""

    def __init__(self):
        """Initialize the mesh converter node."""
        rospy.init_node('point_cloud_to_mesh', anonymous=False)

        # Parameters
        self.voxel_size = rospy.get_param('~voxel_size', 0.05)  # Voxel grid size for downsampling
        self.max_distance = rospy.get_param('~max_distance', 5.0)  # Max distance for mesh generation
        self.mesh_alpha = rospy.get_param('~mesh_alpha', 0.7)  # Mesh transparency
        self.use_3d_camera = rospy.get_param('~use_3d_camera', True)  # Use RGB-D camera point cloud

        # Publishers
        self.mesh_pub = rospy.Publisher('/lidar/mesh', Marker, queue_size=10)
        self.triangles_pub = rospy.Publisher('/lidar/mesh_triangles', MarkerArray, queue_size=10)

        # Subscribers - support both 2D LiDAR and 3D camera point clouds
        if self.use_3d_camera:
            # Subscribe to RGB-D camera point cloud (TRUE 3D!)
            self.cloud_sub = rospy.Subscriber(
                '/hsrb/head_rgbd_sensor/depth_registered/rectified_points',
                PointCloud2,
                self.cloud_callback
            )
            rospy.loginfo("Using RGB-D camera 3D point cloud for mesh generation")
        else:
            # Subscribe to converted 2D LiDAR point cloud
            self.cloud_sub = rospy.Subscriber('/lidar/point_cloud', PointCloud2, self.cloud_callback)
            rospy.loginfo("Using 2D LiDAR point cloud for mesh generation")

        # Mesh color
        self.mesh_color = ColorRGBA(0.0, 0.7, 1.0, self.mesh_alpha)  # Cyan

        rospy.loginfo("Point Cloud to Mesh converter initialized")
        rospy.loginfo(f"Voxel size: {self.voxel_size}m")
        rospy.loginfo(f"Max distance: {self.max_distance}m")

    def cloud_callback(self, cloud_msg):
        """
        Process incoming point cloud and generate mesh.

        Args:
            cloud_msg (sensor_msgs/PointCloud2): Input point cloud
        """
        try:
            # Convert PointCloud2 to numpy array
            points = self.pointcloud2_to_array(cloud_msg)

            if len(points) < 3:
                rospy.logwarn("Not enough points for mesh generation")
                return

            # Filter points by distance
            points = self.filter_by_distance(points, self.max_distance)

            if len(points) < 3:
                return

            # Downsample using voxel grid
            points = self.voxel_downsample(points, self.voxel_size)

            if len(points) < 3:
                return

            # Detect if this is 3D or 2D planar data
            z_range = np.max(points[:, 2]) - np.min(points[:, 2])
            is_3d = z_range > 0.1  # If Z variation > 10cm, it's 3D data

            if is_3d:
                rospy.loginfo_throttle(5.0, f"Processing 3D point cloud (Z range: {z_range:.2f}m)")
                # For 3D data, use proper 3D mesh generation
                mesh_marker = self.generate_3d_mesh_marker(points, cloud_msg.header)
            else:
                rospy.loginfo_throttle(5.0, f"Processing 2D planar data (Z range: {z_range:.2f}m)")
                # For 2D data, use vertical extrusion
                mesh_marker = self.generate_mesh_marker(points, cloud_msg.header)

            # Publish mesh
            self.mesh_pub.publish(mesh_marker)

        except Exception as e:
            rospy.logerr(f"Error generating mesh: {e}")

    def pointcloud2_to_array(self, cloud_msg):
        """
        Convert PointCloud2 message to numpy array.

        Args:
            cloud_msg (sensor_msgs/PointCloud2): Input point cloud

        Returns:
            numpy.ndarray: Nx3 array of points
        """
        points_list = []
        for point in point_cloud2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z")):
            points_list.append([point[0], point[1], point[2]])

        return np.array(points_list)

    def filter_by_distance(self, points, max_dist):
        """
        Filter points by distance from origin.

        Args:
            points (numpy.ndarray): Input points
            max_dist (float): Maximum distance

        Returns:
            numpy.ndarray: Filtered points
        """
        distances = np.linalg.norm(points, axis=1)
        mask = distances <= max_dist
        return points[mask]

    def voxel_downsample(self, points, voxel_size):
        """
        Downsample point cloud using voxel grid.

        Args:
            points (numpy.ndarray): Input points
            voxel_size (float): Voxel size

        Returns:
            numpy.ndarray: Downsampled points
        """
        if len(points) == 0:
            return points

        # Compute voxel indices
        voxel_indices = np.floor(points / voxel_size).astype(int)

        # Get unique voxels
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)

        return points[unique_indices]

    def generate_3d_mesh_marker(self, points, header):
        """
        Generate OPTIMIZED 3D mesh from RGB-D camera point cloud.

        Optimizations:
        - Delaunay triangulation on 2D projection (XY plane) for speed
        - Multi-criteria quality filtering (edge length, area, Z-variation)
        - Smart sampling for optimal triangle density (3000 points)
        - Adaptive coloring based on surface normals and height

        Quality improvements:
        - Removes degenerate triangles (min area threshold)
        - Prevents spanning gaps (max edge length 0.35m)
        - Avoids connecting different height levels (max Z-diff 0.4m)
        - 4-tier color gradient for walls (red→orange→yellow→cyan)

        Args:
            points (numpy.ndarray): Input 3D points
            header (std_msgs/Header): Message header

        Returns:
            visualization_msgs/Marker: Optimized mesh marker with up to 10k triangles
        """
        marker = Marker()
        marker.header = header
        # Override frame to use head_rgbd_sensor_link which exists in TF
        # The point cloud says rgb_frame but that doesn't exist in TF tree
        marker.header.frame_id = "head_rgbd_sensor_link"
        marker.ns = "lidar_mesh_3d"
        marker.id = 0
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD

        # Initialize orientation
        marker.pose.orientation.w = 1.0

        # Scale
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        # Don't set marker.color - we'll use per-vertex colors instead!

        rospy.loginfo(f"Generating 3D mesh from {len(points)} points")

        if len(points) < 4:
            rospy.logwarn("Not enough points for 3D mesh")
            return marker

        # OPTIMIZATION: Use Delaunay triangulation for MUCH better quality
        # Calculate height range for coloring
        z_min = np.min(points[:, 2])
        z_max = np.max(points[:, 2])
        z_range = z_max - z_min if z_max > z_min else 1.0
        rospy.loginfo(f"Height range: {z_min:.2f}m to {z_max:.2f}m (range: {z_range:.2f}m)")

        # OPTIMIZATION 1: Smart sampling - keep more points for better quality
        target_points = 3000  # Increased from 2000
        if len(points) > target_points:
            step = len(points) // target_points
            sampled_points = points[::step]
        else:
            sampled_points = points

        rospy.loginfo(f"Using {len(sampled_points)} points for triangulation")

        # OPTIMIZATION 2: Use 2D Delaunay (MUCH faster than k-NN)
        # Project to XY plane for triangulation
        points_2d = sampled_points[:, :2]

        try:
            tri = Delaunay(points_2d)
            rospy.loginfo(f"Delaunay created {len(tri.simplices)} candidate triangles")
        except Exception as e:
            rospy.logerr(f"Delaunay triangulation failed: {e}")
            return marker

        # QUALITY IMPROVEMENT: Filter triangles with multiple criteria
        max_edge_length = 0.35  # Optimized - tighter for better quality
        max_triangles = 10000  # Increased capacity
        min_triangle_area = 0.0005  # Remove tiny degenerate triangles
        max_z_variation = 0.4  # Don't span different height levels
        triangle_count = 0

        for simplex in tri.simplices:
            if triangle_count >= max_triangles:
                break

            # Get 3D coordinates
            p0 = sampled_points[simplex[0]]
            p1 = sampled_points[simplex[1]]
            p2 = sampled_points[simplex[2]]

            # QUALITY CHECK 1: Edge length (avoid spanning gaps)
            edge1 = np.linalg.norm(p1 - p0)
            edge2 = np.linalg.norm(p2 - p1)
            edge3 = np.linalg.norm(p0 - p2)

            if max(edge1, edge2, edge3) > max_edge_length:
                continue

            # QUALITY CHECK 2: Triangle area (avoid degenerate triangles)
            v1 = p1 - p0
            v2 = p2 - p0
            cross = np.cross(v1, v2)
            area = 0.5 * np.linalg.norm(cross)

            if area < min_triangle_area:
                continue

            # QUALITY CHECK 3: Z-variation (avoid spanning floors/walls)
            z_diff = max(abs(p0[2] - p1[2]), abs(p1[2] - p2[2]), abs(p0[2] - p2[2]))
            if z_diff > max_z_variation:
                continue

            # Calculate normal (already have cross product)
            normal = cross / (2.0 * area) if area > 0 else np.array([0, 0, 1])

            # Determine surface type
            is_horizontal = abs(normal[2]) > 0.7

            # IMPROVED COLORING: 4-tier gradient for walls
            avg_z = (p0[2] + p1[2] + p2[2]) / 3.0
            height_ratio = (avg_z - z_min) / z_range if z_range > 0 else 0.5

            if is_horizontal:
                if normal[2] > 0:  # Floor
                    # Green with height variation
                    intensity = 0.7 + 0.3 * height_ratio
                    color = ColorRGBA(0.1, intensity, 0.1, 1.0)
                else:  # Ceiling
                    color = ColorRGBA(0.3, 0.5, 1.0, 1.0)
            else:  # Walls - 4-tier color gradient
                if height_ratio < 0.25:
                    color = ColorRGBA(1.0, 0.1, 0.1, 1.0)  # Bright Red (bottom)
                elif height_ratio < 0.5:
                    color = ColorRGBA(1.0, 0.65, 0.1, 1.0)  # Orange
                elif height_ratio < 0.75:
                    color = ColorRGBA(1.0, 1.0, 0.1, 1.0)  # Yellow
                else:
                    color = ColorRGBA(0.1, 0.9, 0.9, 1.0)  # Cyan (top)

            # Add triangle
            marker.points.append(Point(p0[0], p0[1], p0[2]))
            marker.points.append(Point(p1[0], p1[1], p1[2]))
            marker.points.append(Point(p2[0], p2[1], p2[2]))
            marker.colors.append(color)
            marker.colors.append(color)
            marker.colors.append(color)

            triangle_count += 1

        rospy.loginfo(f"Created {triangle_count} OPTIMIZED triangles (Delaunay)")

        return marker

    def generate_mesh_marker(self, points, header):
        """
        Generate mesh marker from 2D LiDAR scan by creating vertical wall planes.

        Since HSR LiDAR is 2D planar (horizontal scan), we extrude the scan
        points vertically to create wall-like surfaces.

        Args:
            points (numpy.ndarray): Input points
            header (std_msgs/Header): Message header

        Returns:
            visualization_msgs/Marker: Mesh marker
        """
        marker = Marker()
        marker.header = header
        marker.ns = "lidar_mesh"
        marker.id = 0
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD

        # Initialize orientation (fix quaternion error!)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Scale
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        # Color
        marker.color = self.mesh_color

        rospy.loginfo(f"Generating vertical wall mesh from {len(points)} 2D scan points")

        if len(points) < 2:
            rospy.logwarn("Not enough points for mesh generation")
            return marker

        # For 2D LiDAR, we create vertical wall segments
        # Each consecutive pair of scan points becomes a vertical rectangle

        wall_height = 2.0  # Height of wall planes (meters)
        min_segment_length = 0.05  # Minimum distance between points
        max_segment_length = 0.5   # Maximum gap to connect points

        triangle_count = 0
        max_triangles = 500

        # Sort points by angle around robot for proper ordering
        angles = np.arctan2(points[:, 1], points[:, 0])
        sorted_indices = np.argsort(angles)
        points_sorted = points[sorted_indices]

        # Create vertical wall segments between consecutive points
        for i in range(len(points_sorted) - 1):
            if triangle_count >= max_triangles:
                break

            p1 = points_sorted[i]
            p2 = points_sorted[i + 1]

            # Distance between points
            segment_length = np.linalg.norm(p2[:2] - p1[:2])

            # Skip if points are too close or too far apart
            if segment_length < min_segment_length or segment_length > max_segment_length:
                continue

            # Create 4 vertices for a vertical rectangular wall segment
            # Bottom two points (at LiDAR height, z=0)
            v1_bottom = Point(p1[0], p1[1], 0.0)
            v2_bottom = Point(p2[0], p2[1], 0.0)

            # Top two points (extruded upward)
            v1_top = Point(p1[0], p1[1], wall_height)
            v2_top = Point(p2[0], p2[1], wall_height)

            # Create two triangles to form the rectangle
            # Triangle 1: bottom-left, bottom-right, top-right
            marker.points.append(v1_bottom)
            marker.points.append(v2_bottom)
            marker.points.append(v2_top)

            marker.colors.append(self.mesh_color)
            marker.colors.append(self.mesh_color)
            marker.colors.append(self.mesh_color)

            # Triangle 2: bottom-left, top-right, top-left
            marker.points.append(v1_bottom)
            marker.points.append(v2_top)
            marker.points.append(v1_top)

            marker.colors.append(self.mesh_color)
            marker.colors.append(self.mesh_color)
            marker.colors.append(self.mesh_color)

            triangle_count += 2

        rospy.loginfo(f"Created {triangle_count} triangles forming vertical wall segments")

        return marker

    def generate_triangle_markers(self, points, header):
        """
        Generate individual triangle markers for better visualization.

        Args:
            points (numpy.ndarray): Input points
            header (std_msgs/Header): Message header

        Returns:
            visualization_msgs/MarkerArray: Array of triangle markers
        """
        marker_array = MarkerArray()

        # Project to 2D for triangulation
        points_2d = points[:, :2]

        try:
            tri = Delaunay(points_2d)

            for i, simplex in enumerate(tri.simplices[:50]):  # Limit to 50 triangles for performance
                marker = Marker()
                marker.header = header
                marker.ns = "lidar_triangles"
                marker.id = i
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD

                # Initialize orientation
                marker.pose.orientation.w = 1.0

                marker.scale.x = 0.01  # Line width
                marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)  # Red edges

                # Add triangle edges
                p1 = points[simplex[0]]
                p2 = points[simplex[1]]
                p3 = points[simplex[2]]

                marker.points.append(Point(p1[0], p1[1], p1[2]))
                marker.points.append(Point(p2[0], p2[1], p2[2]))
                marker.points.append(Point(p3[0], p3[1], p3[2]))
                marker.points.append(Point(p1[0], p1[1], p1[2]))  # Close the triangle

                marker_array.markers.append(marker)

        except Exception as e:
            rospy.logwarn(f"Triangle marker generation failed: {e}")

        return marker_array

    def run(self):
        """Run the node."""
        rospy.spin()


if __name__ == '__main__':
    try:
        converter = PointCloudToMesh()
        converter.run()
    except rospy.ROSInterruptException:
        pass
