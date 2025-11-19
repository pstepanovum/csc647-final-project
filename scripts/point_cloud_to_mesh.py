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

        # Publishers
        self.mesh_pub = rospy.Publisher('/lidar/mesh', Marker, queue_size=10)
        self.triangles_pub = rospy.Publisher('/lidar/mesh_triangles', MarkerArray, queue_size=10)

        # Subscriber
        self.cloud_sub = rospy.Subscriber('/lidar/point_cloud', PointCloud2, self.cloud_callback)

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

            # Generate mesh using Delaunay triangulation
            mesh_marker = self.generate_mesh_marker(points, cloud_msg.header)

            # Publish mesh
            self.mesh_pub.publish(mesh_marker)

            # Generate and publish triangle markers
            triangle_markers = self.generate_triangle_markers(points, cloud_msg.header)
            self.triangles_pub.publish(triangle_markers)

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

    def generate_mesh_marker(self, points, header):
        """
        Generate mesh marker using Delaunay triangulation.

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

        # Scale
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        # Color
        marker.color = self.mesh_color

        # Project to 2D (x, y) for triangulation
        points_2d = points[:, :2]

        try:
            # Perform Delaunay triangulation
            tri = Delaunay(points_2d)

            # Add triangles to marker
            for simplex in tri.simplices:
                # Get the three vertices of the triangle
                p1 = points[simplex[0]]
                p2 = points[simplex[1]]
                p3 = points[simplex[2]]

                # Add triangle vertices
                marker.points.append(Point(p1[0], p1[1], p1[2]))
                marker.points.append(Point(p2[0], p2[1], p2[2]))
                marker.points.append(Point(p3[0], p3[1], p3[2]))

                # Add colors for each vertex
                marker.colors.append(self.mesh_color)
                marker.colors.append(self.mesh_color)
                marker.colors.append(self.mesh_color)

        except Exception as e:
            rospy.logwarn(f"Delaunay triangulation failed: {e}")

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
