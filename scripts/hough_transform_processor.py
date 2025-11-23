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

        # Subscriber - Subscribe to RGB-D camera for 3D plane detection
        self.cloud_sub = rospy.Subscriber(
            '/hsrb/head_rgbd_sensor/depth_registered/rectified_points',
            PointCloud2,
            self.cloud_callback
        )

        rospy.loginfo("Hough Transform Processor initialized")
        rospy.loginfo("Subscribing to RGB-D camera for 3D plane detection")
        rospy.loginfo(f"Grid resolution: {self.grid_resolution}m")
        rospy.loginfo(f"Hough threshold: {self.hough_threshold}")

    def cloud_callback(self, cloud_msg):
        """
        Process incoming point cloud with Hough transform.
        Detect planes by projecting to XY (floor/ceiling), XZ (walls), and YZ (walls).

        Args:
            cloud_msg (sensor_msgs/PointCloud2): Input point cloud
        """
        try:
            # Convert PointCloud2 to numpy array
            points = self.pointcloud2_to_array(cloud_msg)

            if len(points) < 10:
                rospy.logwarn("Not enough points for Hough transform")
                return

            # Filter points by distance
            distances = np.linalg.norm(points, axis=1)
            points = points[distances <= 5.0]  # Keep points within 5m

            if len(points) < 10:
                return

            all_lines = []

            # Detect horizontal planes (floor/ceiling) - XY projection
            xy_grid = self.create_grid_from_projection(points, 'xy')
            xy_lines = self.detect_lines(xy_grid)
            for x1, y1, x2, y2 in xy_lines:
                # Convert to 3D lines on floor (z=0)
                all_lines.append(('xy', x1, y1, 0, x2, y2, 0))

            # Detect vertical planes (walls) - XZ projection
            xz_grid = self.create_grid_from_projection(points, 'xz')
            xz_lines = self.detect_lines(xz_grid)
            for x1, z1, x2, z2 in xz_lines:
                # Convert to 3D lines on wall (y varies)
                all_lines.append(('xz', x1, 0, z1, x2, 0, z2))

            # Detect vertical planes (walls) - YZ projection
            yz_grid = self.create_grid_from_projection(points, 'yz')
            yz_lines = self.detect_lines(yz_grid)
            for y1, z1, y2, z2 in yz_lines:
                # Convert to 3D lines on wall (x varies)
                all_lines.append(('yz', 0, y1, z1, 0, y2, z2))

            # Publish occupancy grid (XY projection for visualization)
            grid_msg = self.create_grid_message(xy_grid, cloud_msg.header)
            self.grid_pub.publish(grid_msg)

            # Create 3D line markers
            line_markers = self.create_3d_line_markers(all_lines, cloud_msg.header)
            self.lines_pub.publish(line_markers)

            rospy.loginfo_throttle(5.0, f"Detected {len(all_lines)} plane boundaries")

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

    def create_grid_from_projection(self, points, projection='xy'):
        """
        Create occupancy grid from point cloud using specified projection.

        Args:
            points (numpy.ndarray): Input 3D points
            projection (str): 'xy', 'xz', or 'yz'

        Returns:
            numpy.ndarray: Binary occupancy grid
        """
        # Initialize grid
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        # Get grid center
        center = self.grid_size // 2

        # Select projection axes
        if projection == 'xy':
            idx1, idx2 = 0, 1  # X and Y
        elif projection == 'xz':
            idx1, idx2 = 0, 2  # X and Z
        elif projection == 'yz':
            idx1, idx2 = 1, 2  # Y and Z
        else:
            raise ValueError(f"Unknown projection: {projection}")

        # Convert points to grid coordinates
        for point in points:
            coord1 = int(point[idx1] / self.grid_resolution) + center
            coord2 = int(point[idx2] / self.grid_resolution) + center

            # Check bounds
            if 0 <= coord1 < self.grid_size and 0 <= coord2 < self.grid_size:
                grid[coord2, coord1] = 255  # Occupied cell

        return grid

    def create_occupancy_grid(self, points):
        """
        Create occupancy grid from point cloud (XY projection).

        Args:
            points (numpy.ndarray): Input points

        Returns:
            numpy.ndarray: Binary occupancy grid
        """
        return self.create_grid_from_projection(points, 'xy')

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

    def create_3d_line_markers(self, lines, header):
        """
        Create 3D visualization markers for detected plane boundaries.

        Args:
            lines (list): List of (projection, x1, y1, z1, x2, y2, z2) tuples
            header (std_msgs/Header): Message header

        Returns:
            visualization_msgs/MarkerArray: 3D line markers
        """
        marker_array = MarkerArray()

        # Convert grid coordinates to world coordinates and assign colors
        center = self.grid_size // 2

        for i, line_data in enumerate(lines):
            projection = line_data[0]
            x1, y1, z1, x2, y2, z2 = line_data[1:]

            marker = Marker()
            marker.header = header
            marker.header.frame_id = "head_rgbd_sensor_link"  # Use camera frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = f"plane_{projection}"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.lifetime = rospy.Duration(0)

            # Initialize orientation
            marker.pose.orientation.w = 1.0

            # Line properties
            marker.scale.x = 0.03  # Line width

            # Color based on projection type
            if projection == 'xy':
                # Floor/ceiling planes - Blue
                marker.color = ColorRGBA(0.0, 0.5, 1.0, 1.0)
            elif projection == 'xz':
                # Walls (XZ plane) - Red
                marker.color = ColorRGBA(1.0, 0.2, 0.2, 1.0)
            elif projection == 'yz':
                # Walls (YZ plane) - Green
                marker.color = ColorRGBA(0.2, 1.0, 0.2, 1.0)

            # Convert grid coordinates to world coordinates
            if projection == 'xy':
                wx1 = (x1 - center) * self.grid_resolution
                wy1 = (y1 - center) * self.grid_resolution
                wz1 = z1
                wx2 = (x2 - center) * self.grid_resolution
                wy2 = (y2 - center) * self.grid_resolution
                wz2 = z2
            elif projection == 'xz':
                wx1 = (x1 - center) * self.grid_resolution
                wy1 = y1
                wz1 = (z1 - center) * self.grid_resolution
                wx2 = (x2 - center) * self.grid_resolution
                wy2 = y2
                wz2 = (z2 - center) * self.grid_resolution
            elif projection == 'yz':
                wx1 = x1
                wy1 = (y1 - center) * self.grid_resolution
                wz1 = (z1 - center) * self.grid_resolution
                wx2 = x2
                wy2 = (y2 - center) * self.grid_resolution
                wz2 = (z2 - center) * self.grid_resolution

            # Add line endpoints
            p1 = Point()
            p1.x = wx1
            p1.y = wy1
            p1.z = wz1

            p2 = Point()
            p2.x = wx2
            p2.y = wy2
            p2.z = wz2

            marker.points.append(p1)
            marker.points.append(p2)

            marker_array.markers.append(marker)

        return marker_array

    def create_line_markers(self, lines, header):
        """
        Create visualization markers for detected 2D lines (legacy function).

        Args:
            lines (list): Detected lines as (x1, y1, x2, y2)
            header (std_msgs/Header): Message header

        Returns:
            visualization_msgs/MarkerArray: Line markers
        """
        # Convert to 3D format and call 3D function
        lines_3d = [('xy', x1, y1, 0, x2, y2, 0) for x1, y1, x2, y2 in lines]
        return self.create_3d_line_markers(lines_3d, header)

    def run(self):
        """Run the node."""
        rospy.spin()


if __name__ == '__main__':
    try:
        processor = HoughTransformProcessor()
        processor.run()
    except rospy.ROSInterruptException:
        pass
