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
        /lidar/mesh_edges (visualization_msgs/Marker): Black edges for mesh wireframe
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
        self.edge_width = rospy.get_param('~edge_width', 0.005)  # Width of black edges

        # Publishers
        self.mesh_pub = rospy.Publisher('/lidar/mesh', Marker, queue_size=10)
        self.mesh_edges_pub = rospy.Publisher('/lidar/mesh_edges', Marker, queue_size=10)
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
        rospy.loginfo(f"Edge width: {self.edge_width}m")
        rospy.loginfo("Mesh generation: Delaunay Triangulation with black edge outlines")
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

            rospy.loginfo_throttle(2.0, f"Received {len(points)} raw points from PointCloud2")

            if len(points) < 4:
                rospy.logwarn("Not enough points for mesh generation")
                return

            # Filter points by distance
            points = self.filter_by_distance(points, self.max_distance)

            rospy.loginfo_throttle(2.0, f"After distance filter: {len(points)} points")

            if len(points) < 4:
                return

            # Downsample using voxel grid
            points_downsampled = self.voxel_downsample(points, self.voxel_size)

            rospy.loginfo_throttle(2.0, f"After voxel downsampling: {len(points_downsampled)} points")

            if len(points_downsampled) < 4:
                return

            # Generate Delaunay mesh with edges
            mesh_marker, edges_marker = self.generate_delaunay_mesh_with_edges(
                points_downsampled, cloud_msg.header
            )
            
            self.mesh_pub.publish(mesh_marker)
            self.mesh_edges_pub.publish(edges_marker)

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Publish statistics
            stats = f"Frame {self.frame_counter}: {len(points_downsampled)} points, {processing_time:.1f}ms"
            self.stats_pub.publish(String(data=stats))

        except Exception as e:
            rospy.logerr(f"Error in mesh generation: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())

    def pointcloud2_to_array(self, cloud_msg):
        """
        Convert PointCloud2 message to numpy array.
        Extracts ALL valid points from the point cloud.

        Args:
            cloud_msg (sensor_msgs/PointCloud2): Input point cloud

        Returns:
            numpy.ndarray: Nx3 array of points
        """
        points_list = []
        
        # Read all points, skip NaNs to get full spectrum
        for point in point_cloud2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z")):
            # Only add valid points (not NaN or Inf)
            if np.isfinite(point[0]) and np.isfinite(point[1]) and np.isfinite(point[2]):
                points_list.append([point[0], point[1], point[2]])

        if len(points_list) == 0:
            rospy.logwarn("No valid points found in PointCloud2")
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
        if len(points) == 0:
            return points
            
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

    def generate_delaunay_mesh_with_edges(self, points, header):
        """
        Generate mesh using Delaunay triangulation with black edge outlines.

        Args:
            points (numpy.ndarray): Input 3D points
            header (std_msgs/Header): Message header

        Returns:
            tuple: (mesh_marker, edges_marker) - Triangle mesh and edge wireframe
        """
        # Create mesh marker
        mesh_marker = Marker()
        mesh_marker.header = header
        mesh_marker.header.frame_id = "head_rgbd_sensor_link"
        mesh_marker.header.stamp = rospy.Time.now()
        mesh_marker.ns = "delaunay_mesh"
        mesh_marker.id = 0
        mesh_marker.type = Marker.TRIANGLE_LIST
        mesh_marker.action = Marker.ADD
        mesh_marker.pose.orientation.w = 1.0
        mesh_marker.lifetime = rospy.Duration(0)
        mesh_marker.scale.x = 1.0
        mesh_marker.scale.y = 1.0
        mesh_marker.scale.z = 1.0

        # Create edges marker
        edges_marker = Marker()
        edges_marker.header = header
        edges_marker.header.frame_id = "head_rgbd_sensor_link"
        edges_marker.header.stamp = rospy.Time.now()
        edges_marker.ns = "mesh_edges"
        edges_marker.id = 0
        edges_marker.type = Marker.LINE_LIST
        edges_marker.action = Marker.ADD
        edges_marker.pose.orientation.w = 1.0
        edges_marker.lifetime = rospy.Duration(0)
        edges_marker.scale.x = self.edge_width  # Line width
        edges_marker.color = ColorRGBA(0.0, 0.0, 0.0, 1.0)  # Black edges

        rospy.loginfo(f"Generating Delaunay mesh from {len(points)} points")

        # Sample points if too many (for performance)
        max_points = 2000
        if len(points) > max_points:
            step = len(points) // max_points
            sampled_points = points[::step]
            rospy.loginfo(f"Sampled down to {len(sampled_points)} points for performance")
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

            # Track unique edges to avoid duplicates
            edge_set = set()

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

                # Add triangle vertices to mesh
                mesh_marker.points.append(Point(p0[0], p0[1], p0[2]))
                mesh_marker.points.append(Point(p1[0], p1[1], p1[2]))
                mesh_marker.points.append(Point(p2[0], p2[1], p2[2]))

                # Color by average height of triangle
                avg_z = (p0[2] + p1[2] + p2[2]) / 3.0
                height_ratio = (avg_z - z_min) / z_range if z_range > 0 else 0.5

                # BOLD, VIBRANT, FULLY OPAQUE height-based gradient
                if height_ratio < 0.33:
                    color = ColorRGBA(0.0, 0.5, 1.0, 1.0)  # Bright cyan-blue - low (FULLY OPAQUE)
                elif height_ratio < 0.66:
                    color = ColorRGBA(0.0, 1.0, 0.5, 1.0)  # Bright green - mid (FULLY OPAQUE)
                else:
                    color = ColorRGBA(1.0, 0.5, 0.0, 1.0)  # Bright orange-red - high (FULLY OPAQUE)

                # Same color for all 3 vertices of the triangle
                mesh_marker.colors.append(color)
                mesh_marker.colors.append(color)
                mesh_marker.colors.append(color)

                # Add edges (avoid duplicates using sorted tuple of indices)
                edges = [
                    (simplex[0], simplex[1]),
                    (simplex[1], simplex[2]),
                    (simplex[2], simplex[0])
                ]

                for edge in edges:
                    edge_key = tuple(sorted(edge))
                    if edge_key not in edge_set:
                        edge_set.add(edge_key)
                        # Add edge to edges marker
                        idx0, idx1 = edge
                        pt0 = sampled_points[idx0]
                        pt1 = sampled_points[idx1]
                        edges_marker.points.append(Point(pt0[0], pt0[1], pt0[2]))
                        edges_marker.points.append(Point(pt1[0], pt1[1], pt1[2]))

                triangle_count += 1

            rospy.loginfo(f"Created {triangle_count} valid triangles in mesh")
            rospy.loginfo(f"Created {len(edge_set)} unique edges")

        except Exception as e:
            rospy.logerr(f"Delaunay triangulation failed: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())

        return mesh_marker, edges_marker

    def run(self):
        """Run the node."""
        rospy.spin()


if __name__ == '__main__':
    try:
        generator = DelaunayMeshGenerator()
        generator.run()
    except rospy.ROSInterruptException:
        pass