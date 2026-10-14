// Added by YANDEX LLC on 2026-06-08 for local map construction.

#include <ros/ros.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/TransformStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include <pcl/common/centroid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Dense>

#include <ikd-Tree/ikd_Tree.h>
#include <common_lib.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>


using KdTreePtr = KD_TREE<PointType>::Ptr;


PointCloudXYZI::Ptr load_cloud(const std::string& path) {
    PointCloudXYZI::Ptr cloud(new PointCloudXYZI);
    if (pcl::io::loadPCDFile(path, *cloud) < 0) {
        throw std::runtime_error("Failed to load point cloud: " + path);
    }
    return cloud;
}


void save_cloud(const std::string& path, const PointCloudXYZI& cloud, bool binary) {
    if (pcl::io::savePCDFile(path, cloud, binary) < 0) {
        throw std::runtime_error("Failed to save point cloud: " + path);
    }
}


PointCloudXYZI::Ptr voxelize_cloud(
    PointCloudXYZI::Ptr cloud_in, double leaf_size) {
    PointCloudXYZI::Ptr cloud_out(new PointCloudXYZI);

    pcl::VoxelGrid<PointType> voxel_grid;
    voxel_grid.setInputCloud(cloud_in);
    voxel_grid.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel_grid.filter(*cloud_out);

    return cloud_out;
}


bool estimate_normal(const PointVector& points, Eigen::Vector3d& centroid, Eigen::Vector3d& normal) {
    PointCloudXYZI::Ptr cloud(new PointCloudXYZI);
    cloud->points = points;

    Eigen::Matrix3d cov;
    Eigen::Vector4d mean;
    if (pcl::computeMeanAndCovarianceMatrix(*cloud, cov, mean) == 0) {
        return false;
    }

    centroid = Eigen::Vector3d(mean.x(), mean.y(), mean.z());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    if (solver.info() != Eigen::Success) {
        return false;
    }

    normal = solver.eigenvectors().col(0);
    normal.normalize();

    return true;
}


PointCloudXYZI::Ptr estimate_normals(const PointCloudXYZI::Ptr& cloud_in, const PointCloudXYZI::Ptr& dense_cloud) {
    PointCloudXYZI::Ptr cloud_out(new PointCloudXYZI);
    cloud_out->reserve(cloud_in->size());

    KdTreePtr ikdtree = std::make_shared<KD_TREE<PointType>>();
    ikdtree->set_downsample_param(0.01);
    ikdtree->Build(dense_cloud->points);

    for (const auto& pt : cloud_in->points) {
        PointType pn;
        pn.x = pt.x;
        pn.y = pt.y;
        pn.z = pt.z;
        pn.intensity = pt.intensity;
        pn.normal_x = 0.0;
        pn.normal_y = 0.0;
        pn.normal_z = 0.0;
        pn.curvature = 0.0;

        PointVector neighbors;
        ikdtree->Radius_Search(pt, 0.5, neighbors);

        Eigen::Vector3d centroid;
        Eigen::Vector3d normal;
        if (estimate_normal(neighbors, centroid, normal)) {
            pn.normal_x = normal.x();
            pn.normal_y = normal.y();
            pn.normal_z = normal.z();
        }

        cloud_out->push_back(pn);
    }

    return cloud_out;
}


visualization_msgs::Marker build_surface_marker(
    const PointCloudXYZI::Ptr& cloud,
    const std::string& frame_id,
    double radius,
    int circle_segments) {
    visualization_msgs::Marker marker;
    marker.header.stamp = ros::Time::now();
    marker.header.frame_id = frame_id;
    marker.ns = "surfaces";
    marker.id = 1;
    marker.type = visualization_msgs::Marker::TRIANGLE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = 1.0;
    marker.scale.y = 1.0;
    marker.scale.z = 1.0;
    marker.color.r = 0.5;
    marker.color.g = 0.5;
    marker.color.b = 0.5;
    marker.color.a = 0.5;
    marker.points.reserve(3 * circle_segments * cloud->size());

    for (const auto& pn : cloud->points) {
        if (pn.normal_x == 0.0 && pn.normal_y == 0.0 && pn.normal_z == 0.0) {
            continue;
        }
        Eigen::Vector3d n(pn.normal_x, pn.normal_y, pn.normal_z);
        Eigen::Vector3d center(pn.x, pn.y, pn.z);

        Eigen::Vector3d ref(0, 0, 1);
        if (std::abs(n.z()) > 0.9) {
            ref = Eigen::Vector3d(1, 0, 0);
        }
        Eigen::Vector3d u = ref.cross(n).normalized();
        Eigen::Vector3d v = n.cross(u);

        for (int i = 0; i < circle_segments; ++i) {
            double theta0 = 2.0 * M_PI * static_cast<double>(i) / circle_segments;
            double theta1 = 2.0 * M_PI * static_cast<double>(i + 1) / circle_segments;
            Eigen::Vector3d p0 = center + radius * (std::cos(theta0) * u + std::sin(theta0) * v);
            Eigen::Vector3d p1 = center + radius * (std::cos(theta1) * u + std::sin(theta1) * v);

            geometry_msgs::Point c, a, b;
            c.x = center.x();
            c.y = center.y();
            c.z = center.z();
            a.x = p0.x();
            a.y = p0.y();
            a.z = p0.z();
            b.x = p1.x();
            b.y = p1.y();
            b.z = p1.z();
            marker.points.push_back(c);
            marker.points.push_back(a);
            marker.points.push_back(b);
        }
    }
    return marker;
}


void publish_static_transform(tf2_ros::StaticTransformBroadcaster& broadcaster,
    const std::string& parent_frame_id,
    const std::string& child_frame_id) {
    geometry_msgs::TransformStamped static_tf;
    static_tf.header.stamp = ros::Time::now();
    static_tf.header.frame_id = parent_frame_id;
    static_tf.child_frame_id = child_frame_id;
    static_tf.transform.rotation.x = 1.0;
    broadcaster.sendTransform(static_tf);
}


void publish_point_cloud(ros::Publisher& pub,
                        const PointCloudXYZI::Ptr& cloud,
                        const std::string& frame_id) {
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(*cloud, msg);
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = frame_id;
    pub.publish(msg);
}


void publish_markers(
    ros::Publisher& pub,
    const PointCloudXYZI::Ptr& cloud,
    const std::string& frame_id) {
    visualization_msgs::MarkerArray arr;
    arr.markers.push_back(build_surface_marker(cloud, frame_id, 0.25, 16));
    pub.publish(arr);
}


void run(int argc, char** argv) {
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    std::string dense_cloud_path;
    std::string sparse_cloud_path;
    std::string output_cloud_path;

    const std::string frame_id = "map";
    const std::string parent_frame_id = "world";

    const double voxel_size = 0.10;

    pnh.param<std::string>("dense_cloud_path", dense_cloud_path, dense_cloud_path);
    pnh.param<std::string>("sparse_cloud_path", sparse_cloud_path, sparse_cloud_path);
    pnh.param<std::string>("output_cloud_path", output_cloud_path, output_cloud_path);

    PointCloudXYZI::Ptr dense_cloud = load_cloud(dense_cloud_path);
    ROS_INFO("Dense cloud size: %zu", dense_cloud->size());

    dense_cloud = voxelize_cloud(dense_cloud, voxel_size);
    if (!dense_cloud) {
        throw std::runtime_error("Failed to voxelize dense cloud");
    }
    ROS_INFO("Voxelized dense cloud size: %zu", dense_cloud->size());

    PointCloudXYZI::Ptr sparse_cloud = load_cloud(sparse_cloud_path);
    ROS_INFO("Sparse cloud size: %zu", sparse_cloud->size());

    PointCloudXYZI::Ptr sparse_cloud_with_normals = estimate_normals(sparse_cloud, dense_cloud);
    save_cloud(output_cloud_path, *sparse_cloud_with_normals, true);
    ROS_INFO("Saved sparse cloud with normals (%zu points) to %s",
             sparse_cloud_with_normals->size(), output_cloud_path.c_str());

    tf2_ros::StaticTransformBroadcaster static_broadcaster;
    publish_static_transform(static_broadcaster, parent_frame_id, frame_id);

    ros::Publisher dense_pub = nh.advertise<sensor_msgs::PointCloud2>("dense_cloud", 1, true);
    publish_point_cloud(dense_pub, dense_cloud, frame_id);

    ros::Publisher sparse_pub = nh.advertise<sensor_msgs::PointCloud2>("sparse_cloud", 1, true);
    publish_point_cloud(sparse_pub, sparse_cloud, frame_id);

    ros::Publisher markers_pub = nh.advertise<visualization_msgs::MarkerArray>("markers", 1, true);
    publish_markers(markers_pub, sparse_cloud_with_normals, frame_id);

    ROS_INFO("Spinning...");
    ros::spin();
}


int main(int argc, char** argv) {
    ros::init(argc, argv, "map_builder");
    try {
        run(argc, argv);
        return 0;
    } catch (const std::exception& e) {
        ROS_ERROR("%s", e.what());
        return 1;
    }
}
