#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <map>

#include <jsoncpp/json/json.h>

#include <pcl/point_cloud.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>

namespace fs = std::filesystem;

using PointT = pcl::PointXYZRGBL;
using VoxelIndex = Eigen::Vector3i;

constexpr float kVoxelGridLeafSize = 0.05f;

int main(int argc, char *argv[])
{

    if (argc < 1)
    {
        std::cerr << "Usage: cloud_to_segs <path_to_labeled_pointcloud>" << std::endl;
        return -1;
    }

    // Parse arguments
    std::vector<std::string> args(argv, argv + argc);
    fs::path cloud_file_path(args[1]);
    if (!fs::exists(cloud_file_path))
    {
        std::cerr << cloud_file_path << " not found!" << std::endl;
        return -1;
    }

    // Load pointcloud from ply file
    pcl::PointCloud<PointT>::Ptr cloud_ptr(new pcl::PointCloud<PointT>());
    if (pcl::io::loadPLYFile(cloud_file_path.string(), *cloud_ptr) == -1)
    {
        std::cerr << "Error: an error occurred while loading the pointcloud!" << std::endl;
        return -1;
    }

    // Voxelize the pointcloud
    pcl::VoxelGrid<PointT> voxel_grid;
    voxel_grid.setInputCloud(cloud_ptr);
    voxel_grid.setLeafSize(kVoxelGridLeafSize, kVoxelGridLeafSize, kVoxelGridLeafSize);
    pcl::PointCloud<PointT>::Ptr voxelized_cloud_ptr(new pcl::PointCloud<PointT>());
    voxel_grid.filter(*voxelized_cloud_ptr);

    // Collect segments
    std::map<int, std::vector<VoxelIndex>> segments;
    for (size_t i = 0; i < voxelized_cloud_ptr->size(); ++i)
    {
        auto const &point = voxelized_cloud_ptr->at(i);
        auto label = point.label;
        auto index = voxel_grid.getGridCoordinates(point.x, point.y, point.z);
        segments[label].push_back(index);
    }

    // Save the segments as JSON
    Json::Value segs(Json::arrayValue);
    for (auto const &[label, voxel_ids] : segments)
    {
        Json::Value segment_info;
        segment_info["id"] = label;
        Json::Value voxels(Json::arrayValue);
        for (size_t i = 0; i < voxel_ids.size(); ++i)
        {
            Json::Value voxel_id(Json::arrayValue);
            voxel_id.append(voxel_ids[i].x());
            voxel_id.append(voxel_ids[i].y());
            voxel_id.append(voxel_ids[i].z());
            voxels.append(voxel_id);
        }
        segment_info["voxels"] = voxels;
        segs.append(segment_info);
    }

    // Dump the segments info json to file
    fs::path segs_file_path = cloud_file_path.replace_extension(".voxel_segs.json");
    std::ofstream ofs(segs_file_path, std::ofstream::out);
    ofs << segs;

    return 0;
}
