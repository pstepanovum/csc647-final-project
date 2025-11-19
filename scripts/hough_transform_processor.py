#!/usr/bin/env python3
"""
Hough Transform Processor Node
===============================
Applies Hough transform to detect lines and shapes in LiDAR data.
Useful for detecting walls, corridors, and geometric features in the environment.

Topics:
    Subscribed:
        /lidar/point_cloud (sensor_msgs/PointCloud2): Input point cloud from LiDAR

    Published:
        /lidar/detected_lines (visualization_msgs/MarkerArray): Detected lines
        /lidar/hough_grid (nav_msgs/OccupancyGrid): Occupancy grid for Hough space
"""

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import OccupancyGrid
from scipy import ndimage
import cv2


class HoughTransformProcessor:
    """Applies Hough transform to detect lines in LiDAR data."""

    def __init__(self):
        """Initialize the Hough transform processor node."""
        rospy.init_node('hough_transform_processor', anonymous=False)

        # Parameters
        self.grid_resolution = rospy.get_param('~grid_resolution', 0.05)  # Grid resolution (m)
        self.grid_size = rospy.get_param('~grid_size', 200)  # Grid size (cells)
        self.hough_threshold = rospy.get_param('~hough_threshold', 50)  # Minimum votes for line detection
        self.min_line_length = rospy.get_param('~min_line_length', 0.5)  # Minimum line length (m)
        self.max_line_gap = rospy.get_param('~max_line_gap', 0.3)  # Maximum gap between line segments (m)

        # Publishers
        self.lines_pub = rospy.Publisher('/lidar/detected_lines', MarkerArray, queue_size=10)
        self.grid_pub = rospy.Publisher('/lidar/hough_grid', OccupancyGrid, queue_size=10)

        # Subscriber
        self.cloud_sub = rospy.Subscriber('/lidar/point_cloud', PointCloud2, self.cloud_callback)

        rospy.loginfo("Hough Transform Processor initialized")
        rospy.loginfo(f"Grid resolution: {self.grid_resolution}m")
        rospy.loginfo(f"Hough threshold: {self.hough_threshold}")

    def cloud_callback(self, cloud_msg):
        """
        Process incoming point cloud with Hough transform.

        Args:
            cloud_msg (sensor_msgs/PointCloud2): Input point cloud
        """
        try:
            # Convert PointCloud2 to numpy array
            points = self.pointcloud2_to_array(cloud_msg)

            if len(points) < 10:
                rospy.logwarn("Not enough points for Hough transform")
                return

            # Create occupancy grid from points
            grid = self.create_occupancy_grid(points)

            # Publish occupancy grid
            grid_msg = self.create_grid_message(grid, cloud_msg.header)
            self.grid_pub.publish(grid_msg)

            # Detect lines using Hough transform
            lines = self.detect_lines(grid)

            # Convert grid coordinates back to world coordinates
            world_lines = self.grid_to_world(lines)

            # Publish detected lines
            line_markers = self.create_line_markers(world_lines, cloud_msg.header)
            self.lines_pub.publish(line_markers)

        except Exception as e:
            rospy.logerr(f"Error in Hough transform processing: {e}")

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

    def create_occupancy_grid(self, points):
        """
        Create occupancy grid from point cloud.

        Args:
            points (numpy.ndarray): Input points

        Returns:
            numpy.ndarray: Binary occupancy grid
        """
        # Initialize grid
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        # Get grid center
        center = self.grid_size // 2

        # Convert points to grid coordinates
        for point in points:
            x_idx = int(point[0] / self.grid_resolution) + center
            y_idx = int(point[1] / self.grid_resolution) + center

            # Check bounds
            if 0 <= x_idx < self.grid_size and 0 <= y_idx < self.grid_size:
                grid[y_idx, x_idx] = 255  # Occupied cell

        return grid

    def detect_lines(self, grid):
        """
        Detect lines in occupancy grid using Hough transform.

        Args:
            grid (numpy.ndarray): Binary occupancy grid

        Returns:
            list: List of detected lines as (x1, y1, x2, y2) tuples
        """
        # Apply edge detection
        edges = cv2.Canny(grid, 50, 150, apertureSize=3)

        # Detect lines using Probabilistic Hough Transform
        min_line_length_px = int(self.min_line_length / self.grid_resolution)
        max_line_gap_px = int(self.max_line_gap / self.grid_resolution)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=min_line_length_px,
            maxLineGap=max_line_gap_px
        )

        if lines is None:
            return []

        # Convert to list of tuples
        detected_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            detected_lines.append((x1, y1, x2, y2))

        rospy.loginfo(f"Detected {len(detected_lines)} lines")
        return detected_lines

    def grid_to_world(self, lines):
        """
        Convert grid coordinates to world coordinates.

        Args:
            lines (list): Lines in grid coordinates

        Returns:
            list: Lines in world coordinates
        """
        center = self.grid_size // 2
        world_lines = []

        for x1, y1, x2, y2 in lines:
            # Convert to world coordinates
            wx1 = (x1 - center) * self.grid_resolution
            wy1 = (y1 - center) * self.grid_resolution
            wx2 = (x2 - center) * self.grid_resolution
            wy2 = (y2 - center) * self.grid_resolution

            world_lines.append((wx1, wy1, wx2, wy2))

        return world_lines

    def create_grid_message(self, grid, header):
        """
        Create OccupancyGrid message.

        Args:
            grid (numpy.ndarray): Occupancy grid
            header (std_msgs/Header): Message header

        Returns:
            nav_msgs/OccupancyGrid: Grid message
        """
        grid_msg = OccupancyGrid()
        grid_msg.header = header
        grid_msg.info.resolution = self.grid_resolution
        grid_msg.info.width = self.grid_size
        grid_msg.info.height = self.grid_size

        # Set origin (center of grid)
        grid_msg.info.origin.position.x = -(self.grid_size * self.grid_resolution) / 2.0
        grid_msg.info.origin.position.y = -(self.grid_size * self.grid_resolution) / 2.0
        grid_msg.info.origin.position.z = 0.0

        # Convert grid to occupancy values (0-100)
        occupancy_data = (grid / 255.0 * 100).astype(np.int8)
        grid_msg.data = occupancy_data.flatten().tolist()

        return grid_msg

    def create_line_markers(self, lines, header):
        """
        Create visualization markers for detected lines.

        Args:
            lines (list): Detected lines
            header (std_msgs/Header): Message header

        Returns:
            visualization_msgs/MarkerArray: Line markers
        """
        marker_array = MarkerArray()

        for i, (x1, y1, x2, y2) in enumerate(lines):
            marker = Marker()
            marker.header = header
            marker.ns = "detected_lines"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            # Initialize orientation (fix quaternion warning)
            marker.pose.orientation.w = 1.0

            # Line properties
            marker.scale.x = 0.05  # Line width

            # Color - bright green for detected lines
            marker.color = ColorRGBA(0.0, 1.0, 0.0, 1.0)

            # Add line endpoints
            p1 = Point()
            p1.x = x1
            p1.y = y1
            p1.z = 0.0

            p2 = Point()
            p2.x = x2
            p2.y = y2
            p2.z = 0.0

            marker.points.append(p1)
            marker.points.append(p2)

            marker_array.markers.append(marker)

        return marker_array

    def run(self):
        """Run the node."""
        rospy.spin()


if __name__ == '__main__':
    try:
        processor = HoughTransformProcessor()
        processor.run()
    except rospy.ROSInterruptException:
        pass
