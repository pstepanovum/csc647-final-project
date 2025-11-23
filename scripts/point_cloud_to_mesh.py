#!/usr/bin/env python3
"""
Delaunay Mesh Generation Node
==============================
Generates 3D mesh from point cloud using Delaunay triangulation.
Pairs with Hough Transform for plane detection.

Topics:
    Subscribed:
        /hsrb/head_rgbd_sensor/depth_registered/rectified_points (sensor_msgs/PointCloud2): RGB-D camera point cloud

    Published:
        /lidar/mesh (visualization_msgs/Marker): Delaunay triangulated mesh
        /lidar/stats (std_msgs/String): Processing statistics
"""

import rospy
import numpy as np
import time
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, String
from scipy.spatial import Delaunay


class DelaunayMeshGenerator:
    """Generates mesh from point cloud using Delaunay triangulation."""

    def __init__(self):
        """Initialize the mesh generator node."""
        rospy.init_node('delaunay_mesh_generator', anonymous=False)

        # Parameters
        self.voxel_size = rospy.get_param('~voxel_size', 0.05)  # Voxel grid size for downsampling
        self.max_distance = rospy.get_param('~max_distance', 5.0)  # Max distance for mesh
        self.use_3d_camera = rospy.get_param('~use_3d_camera', True)  # Use RGB-D camera point cloud

        # Publishers
        self.mesh_pub = rospy.Publisher('/lidar/mesh', Marker, queue_size=10)
        self.stats_pub = rospy.Publisher('/lidar/stats', String, queue_size=1)

        # Performance tracking
        self.frame_counter = 0

        # Subscribers
        if self.use_3d_camera:
            # Subscribe to RGB-D camera point cloud
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

        rospy.loginfo("Delaunay Mesh Generator initialized")
        rospy.loginfo(f"Voxel size: {self.voxel_size}m")
        rospy.loginfo(f"Max distance: {self.max_distance}m")
        rospy.loginfo("Mesh generation: Delaunay Triangulation")
        rospy.loginfo("Plane detection: Hough Transform (separate node)")

    def cloud_callback(self, cloud_msg):
        """
        Process incoming point cloud and generate mesh.

        Args:
            cloud_msg (sensor_msgs/PointCloud2): Input point cloud
        """
        try:
            # Start timing
            start_time = time.time()
            self.frame_counter += 1

            # Convert PointCloud2 to numpy array
            points = self.pointcloud2_to_array(cloud_msg)

            if len(points) < 4:
                rospy.logwarn("Not enough points for mesh generation")
                return

            # Filter points by distance
            points = self.filter_by_distance(points, self.max_distance)

            if len(points) < 4:
                return

            # Downsample using voxel grid
            points = self.voxel_downsample(points, self.voxel_size)

            if len(points) < 4:
                return

            # Generate Delaunay mesh
            mesh_marker = self.generate_delaunay_mesh(points, cloud_msg.header)
            self.mesh_pub.publish(mesh_marker)

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Publish statistics
            stats = f"Frame {self.frame_counter}: {len(points)} points, {processing_time:.1f}ms"
            self.stats_pub.publish(String(data=stats))

        except Exception as e:
            rospy.logerr(f"Error in mesh generation: {e}")

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

        if len(points_list) == 0:
            return np.array([])

        return np.array(points_list)

    def filter_by_distance(self, points, max_distance):
        """
        Filter points by distance from origin.

        Args:
            points (numpy.ndarray): Input points
            max_distance (float): Maximum distance

        Returns:
            numpy.ndarray: Filtered points
        """
        distances = np.linalg.norm(points, axis=1)
        mask = distances <= max_distance
        return points[mask]

    def voxel_downsample(self, points, voxel_size):
        """
        Downsample point cloud using voxel grid filtering.

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

    def generate_delaunay_mesh(self, points, header):
        """
        Generate mesh using Delaunay triangulation.

        Simple approach for CSC647:
        1. Project 3D points to 2D (XY plane)
        2. Run Delaunay triangulation
        3. Lift triangles back to 3D with original Z values
        4. Filter bad triangles (too long edges)
        5. Color by height

        Args:
            points (numpy.ndarray): Input 3D points
            header (std_msgs/Header): Message header

        Returns:
            visualization_msgs/Marker: Triangle mesh
        """
        marker = Marker()
        marker.header = header
        marker.header.frame_id = "head_rgbd_sensor_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "delaunay_mesh"
        marker.id = 0
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime = rospy.Duration(0)

        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        rospy.loginfo(f"Generating Delaunay mesh from {len(points)} points")

        # Sample points if too many (for performance)
        max_points = 2000
        if len(points) > max_points:
            step = len(points) // max_points
            sampled_points = points[::step]
        else:
            sampled_points = points

        # Project to 2D (XY plane) for Delaunay
        points_2d = sampled_points[:, :2]

        try:
            # Run Delaunay triangulation
            tri = Delaunay(points_2d)
            rospy.loginfo(f"Delaunay created {len(tri.simplices)} triangles")

            # Filter triangles by edge length
            max_edge_length = 0.3  # Maximum edge length in meters
            triangle_count = 0

            # Get height range for coloring
            z_min = np.min(sampled_points[:, 2])
            z_max = np.max(sampled_points[:, 2])
            z_range = z_max - z_min if z_max > z_min else 1.0

            for simplex in tri.simplices:
                # Get triangle vertices
                p0 = sampled_points[simplex[0]]
                p1 = sampled_points[simplex[1]]
                p2 = sampled_points[simplex[2]]

                # Check edge lengths
                edge1 = np.linalg.norm(p1 - p0)
                edge2 = np.linalg.norm(p2 - p1)
                edge3 = np.linalg.norm(p0 - p2)

                # Skip if any edge is too long
                if max(edge1, edge2, edge3) > max_edge_length:
                    continue

                # Add triangle vertices
                marker.points.append(Point(p0[0], p0[1], p0[2]))
                marker.points.append(Point(p1[0], p1[1], p1[2]))
                marker.points.append(Point(p2[0], p2[1], p2[2]))

                # Color by average height of triangle
                avg_z = (p0[2] + p1[2] + p2[2]) / 3.0
                height_ratio = (avg_z - z_min) / z_range if z_range > 0 else 0.5

                # BOLD, VIBRANT height-based gradient
                if height_ratio < 0.33:
                    color = ColorRGBA(0.0, 0.5, 1.0, 0.9)  # Bright cyan-blue - low
                elif height_ratio < 0.66:
                    color = ColorRGBA(0.0, 1.0, 0.5, 0.9)  # Bright green - mid
                else:
                    color = ColorRGBA(1.0, 0.5, 0.0, 0.9)  # Bright orange-red - high

                # Same color for all 3 vertices of the triangle
                marker.colors.append(color)
                marker.colors.append(color)
                marker.colors.append(color)

                triangle_count += 1

            rospy.loginfo(f"Created {triangle_count} valid triangles in mesh")

        except Exception as e:
            rospy.logerr(f"Delaunay triangulation failed: {e}")

        return marker

    def run(self):
        """Run the node."""
        rospy.spin()


if __name__ == '__main__':
    try:
        generator = DelaunayMeshGenerator()
        generator.run()
    except rospy.ROSInterruptException:
        pass
