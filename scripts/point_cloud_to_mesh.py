#!/usr/bin/env python3
"""
Point Cloud Visualization Node
===============================
Simple point cloud visualization for LiDAR data.
Focus on Hough Transform for plane detection (see hough_transform_processor.py).

Topics:
    Subscribed:
        /hsrb/head_rgbd_sensor/depth_registered/rectified_points (sensor_msgs/PointCloud2): RGB-D camera point cloud

    Published:
        /lidar/point_cloud_viz (visualization_msgs/Marker): Point cloud visualization
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


class PointCloudVisualizer:
    """Simple point cloud visualizer - pairs with Hough Transform for plane detection."""

    def __init__(self):
        """Initialize the point cloud visualizer node."""
        rospy.init_node('point_cloud_visualizer', anonymous=False)

        # Parameters
        self.voxel_size = rospy.get_param('~voxel_size', 0.05)  # Voxel grid size for downsampling
        self.max_distance = rospy.get_param('~max_distance', 5.0)  # Max distance for visualization
        self.use_3d_camera = rospy.get_param('~use_3d_camera', True)  # Use RGB-D camera point cloud

        # Publishers
        self.viz_pub = rospy.Publisher('/lidar/point_cloud_viz', Marker, queue_size=10)
        self.stats_pub = rospy.Publisher('/lidar/stats', String, queue_size=1)

        # Performance tracking
        self.frame_counter = 0

        # Subscribers - support both 2D LiDAR and 3D camera point clouds
        if self.use_3d_camera:
            # Subscribe to RGB-D camera point cloud (TRUE 3D!)
            self.cloud_sub = rospy.Subscriber(
                '/hsrb/head_rgbd_sensor/depth_registered/rectified_points',
                PointCloud2,
                self.cloud_callback
            )
            rospy.loginfo("Using RGB-D camera 3D point cloud for visualization")
        else:
            # Subscribe to converted 2D LiDAR point cloud
            self.cloud_sub = rospy.Subscriber('/lidar/point_cloud', PointCloud2, self.cloud_callback)
            rospy.loginfo("Using 2D LiDAR point cloud for visualization")

        rospy.loginfo("Point Cloud Visualizer initialized")
        rospy.loginfo(f"Voxel size: {self.voxel_size}m")
        rospy.loginfo(f"Max distance: {self.max_distance}m")
        rospy.loginfo("For plane detection, see Hough Transform node (hough_transform_processor.py)")

    def cloud_callback(self, cloud_msg):
        """
        Process incoming point cloud and visualize.

        Args:
            cloud_msg (sensor_msgs/PointCloud2): Input point cloud
        """
        try:
            # Start timing
            start_time = time.time()
            self.frame_counter += 1

            # Convert PointCloud2 to numpy array
            points = self.pointcloud2_to_array(cloud_msg)

            if len(points) < 3:
                rospy.logwarn("Not enough points for visualization")
                return

            # Filter points by distance
            points = self.filter_by_distance(points, self.max_distance)

            if len(points) < 3:
                return

            # Downsample using voxel grid
            points = self.voxel_downsample(points, self.voxel_size)

            if len(points) < 3:
                return

            # Create simple point cloud visualization
            viz_marker = self.create_point_cloud_marker(points, cloud_msg.header)
            self.viz_pub.publish(viz_marker)

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Publish statistics
            stats = f"Frame {self.frame_counter}: {len(points)} points, {processing_time:.1f}ms"
            self.stats_pub.publish(String(data=stats))

        except Exception as e:
            rospy.logerr(f"Error in point cloud visualization: {e}")

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

    def create_point_cloud_marker(self, points, header):
        """
        Create a simple point cloud visualization marker.

        Args:
            points (numpy.ndarray): Input 3D points
            header (std_msgs/Header): Message header

        Returns:
            visualization_msgs/Marker: Point cloud visualization
        """
        marker = Marker()
        marker.header = header
        marker.header.frame_id = "head_rgbd_sensor_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "point_cloud"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime = rospy.Duration(0)

        # Point size
        marker.scale.x = 0.02
        marker.scale.y = 0.02

        # Height-based coloring
        if len(points) > 0:
            z_min = np.min(points[:, 2])
            z_max = np.max(points[:, 2])
            z_range = z_max - z_min if z_max > z_min else 1.0

            for point in points:
                # Add point
                marker.points.append(Point(point[0], point[1], point[2]))

                # Color based on height (blue=low, green=mid, red=high)
                height_ratio = (point[2] - z_min) / z_range if z_range > 0 else 0.5

                if height_ratio < 0.33:
                    color = ColorRGBA(0.0, 0.0, 1.0, 1.0)  # Blue - low
                elif height_ratio < 0.66:
                    color = ColorRGBA(0.0, 1.0, 0.0, 1.0)  # Green - mid
                else:
                    color = ColorRGBA(1.0, 0.0, 0.0, 1.0)  # Red - high

                marker.colors.append(color)

        rospy.loginfo_throttle(5.0, f"Visualizing {len(points)} points")
        return marker

    def run(self):
        """Run the node."""
        rospy.spin()


if __name__ == '__main__':
    try:
        visualizer = PointCloudVisualizer()
        visualizer.run()
    except rospy.ROSInterruptException:
        pass
