/*
 * =====================================================================================
 *
 *       Filename:  RayCaster.h
 *
 *    Description:  Interface class for ray caster
 *
 *        Version:  1.0
 *        Created:  01/25/2014 10:46:25 PM
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
#include <pcl/octree/octree.h>

#include <opencv/cv.h>

#include <vector>

typedef pcl::PointXYZRGBNormal pType;

#ifndef RAYCASTER_H
#define RAYCASTER_H

/*
 * =====================================================================================
 *        Class:  RayCaster
 *  Description:  Defitions for ray caster
 * =====================================================================================
 */
class RayCaster
{
	public:
		/* ====================  LIFECYCLE     ======================================= */
		RayCaster (pcl::PointCloud<pType>::Ptr&, float, std::vector<double>);                /* constructor */

		/* ====================  ACCESSORS     ======================================= */

		/* ====================  MUTATORS      ======================================= */

		/* ====================  OPERATORS     ======================================= */
		void castRay(cv::Point3f origin, cv::Point3f direction, std::vector<int> &intersections);
		void castRay(cv::Point3f origin, cv::Point3f direction, std::vector<int> &intersections, int maxIntersections);
		void writeVoxelCenters(std::string path);
		void getVoxelCenters(pcl::PointCloud<pType>::Ptr &centers);
		/* ====================  DATA MEMBERS  ======================================= */
	protected:

	private:
		pcl::octree::OctreePointCloudSearch<pType> *voxelGrid;
		pcl::PointCloud<pType>::Ptr cloudPtr;
}; /* -----  end of class RayCaster  ----- */
#endif
