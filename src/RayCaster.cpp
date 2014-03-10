/*
 * =====================================================================================
 *
 *       Filename:  RayCaster.cpp
 *
 *    Description:  Implimentation for Ray Caster
 *
 *        Version:  1.0
 *        Created:  01/25/2014 10:53:38 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  David Nilosek (), drn2369@cis.rit.edu
 *        Company:  Rochester Institute of Technology
 *
 * =====================================================================================
 */

#include <pcl/io/ply_io.h>

#include "RayCaster.h"

RayCaster::RayCaster(pcl::PointCloud<pType>::Ptr &cloud, float voxelSize, std::vector<double> bBox){

	//Create octree
	voxelGrid = new pcl::octree::OctreePointCloudSearch<pType>(voxelSize);


	voxelGrid->defineBoundingBox(bBox[0], bBox[2], bBox[4],
				     bBox[1], bBox[3], bBox[5]);

	voxelGrid->setInputCloud(cloud);
	voxelGrid->addPointsFromInputCloud();

	//Store pointer
	cloudPtr = cloud;

}

void RayCaster::castRay(cv::Point3f origin, cv::Point3f direction, std::vector<int> &intersections){

	//Set up Eigen vectors for PCL
	Eigen::Vector3f ori,dir;

	ori.x() = origin.x;
	ori.y() = origin.y;
	ori.z() = origin.z;

	dir.x() = direction.x;
	dir.y() = direction.y;
	dir.z() = direction.z;

	//Cast ray
	voxelGrid->getIntersectedVoxelIndices(ori,dir,intersections);
}

void RayCaster::castRay(cv::Point3f origin, cv::Point3f direction, std::vector<int> &intersections, int maxIntersections){

	//Set up Eigen vectors for PCL
	Eigen::Vector3f ori,dir;

	ori.x() = origin.x;
	ori.y() = origin.y;
	ori.z() = origin.z;

	dir.x() = direction.x;
	dir.y() = direction.y;
	dir.z() = direction.z;

	//Cast ray
	voxelGrid->getIntersectedVoxelIndices(ori,dir,intersections,maxIntersections);
}

void RayCaster::writeVoxelCenters(std::string path){
	//Definitions
	pcl::PointCloud<pType>::Ptr writeCloud(new pcl::PointCloud<pType>);
	pType pt;
	pcl::PLYWriter writer;
	
	//Extract the voxel centers
	std::vector<pType, Eigen::aligned_allocator<pType> > voxelList;
	voxelGrid->getOccupiedVoxelCenters(voxelList);
	
	//Fill the write cloud with the voxel centers
	for(int i = 0; i < voxelList.size(); i++){
		writeCloud->points.push_back(voxelList[i]);
	}
	
	//Write out the voxel centers
	writer.write<pType>(path, *writeCloud);	
}

void RayCaster::getVoxelCenters( pcl::PointCloud<pType>::Ptr &centers){

	//Extract the voxel centers
	std::vector<pType, Eigen::aligned_allocator<pType> > voxelList;
	voxelGrid->getOccupiedVoxelCenters(voxelList);
	
	//Fill the write cloud with the voxel centers
	for(int i = 0; i < voxelList.size(); i++){
		centers->points.push_back(voxelList[i]);
	}

}
