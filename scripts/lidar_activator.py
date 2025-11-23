#!/usr/bin/env python3
"""
LiDAR Activator Node
====================
Subscribes to the HSR robot's LiDAR sensor and processes the laser scan data.
Publishes the processed scan data for visualization and further processing.

Topics:
    Subscribed:
        /hsrb/base_scan (sensor_msgs/LaserScan): Raw LiDAR data from HSR

    Published:
        /lidar/scan_processed (sensor_msgs/LaserScan): Processed LiDAR data
        /lidar/point_cloud (sensor_msgs/PointCloud2): Converted point cloud
"""

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Point32
from std_msgs.msg import Header
import laser_geometry.laser_geometry as lg


class LidarActivator:
    """Activates and processes LiDAR data from the HSR robot."""

    def __init__(self):
        """Initialize the LiDAR activator node."""
        rospy.init_node('lidar_activator', anonymous=False)

        # Parameters
        self.scan_topic = rospy.get_param('~scan_topic', '/hsrb/base_scan')
        self.min_range = rospy.get_param('~min_range', 0.1)  # Minimum valid range (m)
        self.max_range = rospy.get_param('~max_range', 30.0)  # Maximum valid range (m)

        # Laser geometry projector for converting LaserScan to PointCloud2
        self.laser_projector = lg.LaserProjection()

        # Publishers
        self.scan_pub = rospy.Publisher('/lidar/scan_processed', LaserScan, queue_size=10)
        self.cloud_pub = rospy.Publisher('/lidar/point_cloud', PointCloud2, queue_size=10)

        # Subscriber
        self.scan_sub = rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback)

        rospy.loginfo("LiDAR Activator initialized")
        rospy.loginfo(f"Subscribing to: {self.scan_topic}")
        rospy.loginfo(f"Range filter: {self.min_range}m to {self.max_range}m")

    def scan_callback(self, scan_msg):
        """
        Process incoming laser scan data.

        Args:
            scan_msg (sensor_msgs/LaserScan): Raw laser scan message
        """
        try:
            # Filter the scan data
            filtered_scan = self.filter_scan(scan_msg)

            # Publish filtered scan
            self.scan_pub.publish(filtered_scan)

            # Convert to point cloud
            point_cloud = self.laser_projector.projectLaser(filtered_scan)

            # Publish point cloud
            self.cloud_pub.publish(point_cloud)

        except Exception as e:
            rospy.logerr(f"Error processing scan: {e}")

    def filter_scan(self, scan):
        """
        Filter laser scan to remove invalid readings.

        Args:
            scan (sensor_msgs/LaserScan): Input scan message

        Returns:
            sensor_msgs/LaserScan: Filtered scan message
        """
        filtered_scan = LaserScan()
        filtered_scan.header = scan.header
        filtered_scan.angle_min = scan.angle_min
        filtered_scan.angle_max = scan.angle_max
        filtered_scan.angle_increment = scan.angle_increment
        filtered_scan.time_increment = scan.time_increment
        filtered_scan.scan_time = scan.scan_time
        filtered_scan.range_min = self.min_range
        filtered_scan.range_max = self.max_range

        # Filter ranges
        filtered_ranges = []
        for r in scan.ranges:
            if self.min_range <= r <= self.max_range:
                filtered_ranges.append(r)
            else:
                filtered_ranges.append(float('inf'))  # Invalid reading

        filtered_scan.ranges = filtered_ranges
        filtered_scan.intensities = scan.intensities

        return filtered_scan

    def run(self):
        """Run the node."""
        rospy.spin()


if __name__ == '__main__':
    try:
        activator = LidarActivator()
        activator.run()
    except rospy.ROSInterruptException:
        pass
