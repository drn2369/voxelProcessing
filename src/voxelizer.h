/*
 * =====================================================================================
 *
 *       Filename:  voxelizer.h
 *
 *    Description:  Header for Voxelizer class
 *
 *        Version:  1.0
 *        Created:  11/05/2013 08:45:18 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  David Nilosek (), drn2369@cis.rit.edu
 *        Company:  Rochester Institute of Technology
 *
 * =====================================================================================
 */
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/octree/octree.h>
#include <pcl/common/common.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <vector>
#include <string>

typedef pcl::PointXYZRGBNormal pType;

#ifndef VOXELIZER_H
#define VOXELIZER_H
/*
 * =====================================================================================
 *        Class:  Voxelizer
 *  Description:  Handles voxelization related processes
 * =====================================================================================
 */
class Voxelizer{
	public:
		//Structure for holding voxel info
		struct voxelInfo{
			float minX;
			float maxX;
			float minY;
			float maxY;
			float minZ;
			float maxZ;
			float voxelSize;
		};
		/* ====================  LIFECYCLE     ======================================= */
		Voxelizer (pcl::PointCloud<pType>::Ptr& , float);            /* constructor */
		Voxelizer (pcl::PointCloud<pType>::Ptr& , float, std::vector<double>);            /* constructor */

		/* ====================  ACCESSORS     ======================================= */
		pcl::octree::OctreePointCloud<pType>* getVoxelGrid(){return voxelGrid;}	
		voxelInfo getVoxelInfo(){return vInfo;}
		/* ====================  MUTATORS      ======================================= */

		/* ====================  OPERATORS     ======================================= */
		void writeVoxelCenters(std::string path);
		void depthClean();
		void estimateSurface(int dNorms);
		void getVoxels(pcl::PointCloud<pType>::Ptr& voxels);
		void getVoxels(pcl::PointCloud<pType>::Ptr& knownVoxels, pcl::PointCloud<pType>::Ptr& estimatedVoxels);
		void getVoxels(pcl::PointCloud<pType>::Ptr& knownVoxels, std::vector<int> &ptsInKnown,
			       pcl::PointCloud<pType>::Ptr& estimatedVoxels, std::vector<int> &ptsInEst);
		void findNearestK(pType point, int k, std::vector<int> &k_inds, std::vector<float> &k_dist);
		void findApproxNearest(pType point, int &inds, float &dist);
		bool isVoxelOccupied(pType point);
		void fillFromLevel(cv::Mat &levelImage, int level);
		//TEST FUNCTIONS
		void testRayCast();
		/* ====================  DATA MEMBERS  ======================================= */
	protected:
		void fillImageFromLevel(cv::Mat &levelImage, int level);
		void hitmiss(cv::Mat& src, cv::Mat& dst, cv::Mat& kernel);
		void updateVoxelGrid(cv::Mat& levelImage, cv::Mat& workingImage, int level);
		void updateVoxelGrid(cv::Mat& workingImage, int level);
		void cardinalNorm(cv::Point3f &norm, int dNorms);
	private:
		pcl::octree::OctreePointCloud<pType> *voxelGrid;
		pcl::PointCloud<pType>::Ptr cloudPtr;
		voxelInfo vInfo;
}; /* -----  end of class Voxelizer  ----- */
#endif
