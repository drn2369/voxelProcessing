/*
 * =====================================================================================
 *
 *       Filename:  voxelizer.cpp
 *
 *    Description: Implimentation for Voxelizer class 
 *
 *        Version:  1.0
 *        Created:  11/05/2013 08:45:09 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  David Nilosek (), drn2369@cis.rit.edu
 *        Company:  Rochester Institute of Technology
 *
 * =====================================================================================
 */
#include <vector>

#include "voxelizer.h"
#include <iostream>

Voxelizer::Voxelizer(pcl::PointCloud<pType>::Ptr &cloud, float voxelSize){

	//Create new octree with voxel resolution 
	voxelGrid = new pcl::octree::OctreePointCloud<pType>(voxelSize);

	//Get bounding box
	Eigen::Vector4f min_pt,max_pt;
	pcl::getMinMax3D(*cloud, min_pt,max_pt);

	//Fill voxelInfo
	vInfo.minX = (double)min_pt.x();
	vInfo.minY = (double)min_pt.y();
	vInfo.minZ = (double)min_pt.z();
	
	vInfo.maxX = (double)max_pt.x();
	vInfo.maxY = (double)max_pt.y();
	vInfo.maxZ = (double)max_pt.z();
	
	vInfo.voxelSize = (double)voxelSize;

	//Set bounding box
	voxelGrid->defineBoundingBox(vInfo.minX, vInfo.minY, vInfo.minZ,
				     vInfo.maxX, vInfo.maxY, vInfo.maxZ);
	//Add points
	voxelGrid->setInputCloud(cloud);
	voxelGrid->addPointsFromInputCloud();

	//Store pointer
	cloudPtr = cloud;
}

Voxelizer::Voxelizer(pcl::PointCloud<pType>::Ptr &cloud, float voxelSize, std::vector<double> boundingBox){

	//Create new octree with voxel resolution 
	voxelGrid = new pcl::octree::OctreePointCloud<pType>(voxelSize);

	//Fill voxelInfo
	vInfo.minX = boundingBox[0];
	vInfo.minY = boundingBox[2];
	vInfo.minZ = boundingBox[4];
	
	vInfo.maxX = boundingBox[1];
	vInfo.maxY = boundingBox[3];
	vInfo.maxZ = boundingBox[5];
	
	vInfo.voxelSize = (double)voxelSize;

	//Set bounding box
	voxelGrid->defineBoundingBox(vInfo.minX, vInfo.minY, vInfo.minZ,
				     vInfo.maxX, vInfo.maxY, vInfo.maxZ);
	//Add points
	voxelGrid->setInputCloud(cloud);
	voxelGrid->addPointsFromInputCloud();

	//Store pointer
	cloudPtr = cloud;
}


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  writeVoxelCenters
 *  Description:  Writes the voxel centers to a ply file
 *
 *  @param path: Path to write
 *
 * =====================================================================================
 */
void Voxelizer::writeVoxelCenters(std::string path){
	
	//Create OctreePointCloudSearch so points in voxels can be found
	pcl::octree::OctreePointCloudSearch<pType> voxelSearch(vInfo.voxelSize);
	voxelSearch.setInputCloud(cloudPtr);
	voxelSearch.addPointsFromInputCloud();
	
	//Definitions
	pcl::PointCloud<pType>::Ptr writeCloud(new pcl::PointCloud<pType>);
	pType pt;
	pcl::PLYWriter writer;
	
	//Extract the voxel centers
	std::vector<pType, Eigen::aligned_allocator<pType> > voxelList;
	voxelGrid->getOccupiedVoxelCenters(voxelList);
	
	//Fill the write cloud with the voxel centers
	for(int i = 0; i < voxelList.size(); i++){

		//Estimate color
		double r = 0.0;
		double g = 0.0;
		double b = 0.0;
		double nInVox = 0;
		std::vector<int> pointsInVox;
		voxelSearch.voxelSearch(voxelList[i],pointsInVox);
	
		if(pointsInVox.size() == 0) continue;

		for(int j = 0; j < pointsInVox.size(); j++){
			r += cloudPtr->points[pointsInVox[j]].r;
			g += cloudPtr->points[pointsInVox[j]].g;
			b += cloudPtr->points[pointsInVox[j]].b;
			nInVox += 1.0;
		}

		r *= 1/nInVox;
		b *= 1/nInVox;
		g *= 1/nInVox;
		
		voxelList[i].r = (int)r;
		voxelList[i].g = (int)g;
		voxelList[i].b = (int)b;

		writeCloud->points.push_back(voxelList[i]);
	}
	
	//Write out the voxel centers
	writer.write<pType>(path, *writeCloud);	
}



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  depthClean
 *  Description:  Steps top-down through the voxel cloud and uses morphalogical
 *  			operators to clean the voxel cloud, find boundaries, and create
 *  			walls
 * =====================================================================================
 */

void Voxelizer::depthClean(){
	//Definitions
	int xRange,yRange,zLevels;

	//Determine ranges
	xRange = (int)((vInfo.maxX-vInfo.minX)/vInfo.voxelSize);
	yRange = (int)((vInfo.maxY-vInfo.minY)/vInfo.voxelSize);
	zLevels = (int)((vInfo.maxZ-vInfo.minZ)/vInfo.voxelSize);

	//Step from max to min z level
	for(int i = zLevels; i > 0; i--){
		//Definitions
		cv::Mat levelImage;
		cv::Mat workingImage;
		cv::Mat workingImage2;
		cv::Mat workingImage3;
		cv::Mat holes;
		cv::Mat strel;
		cv::Mat boundries;
		pcl::PointCloud<pType> pointFill;
		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;

		//Set size of CV image
		levelImage = cv::Mat::zeros(yRange,xRange,CV_8U);

		//Fill image with voxels at level
		fillImageFromLevel(levelImage,i);
	
		//Remove isolated pixels using hit or miss transform
		cv::Mat isolate = (cv::Mat_<char>(3,3) << -1, -1, -1,
							  -1, 1, -1,
							  -1, -1, -1);

		hitmiss(levelImage,workingImage,isolate);
 
		workingImage2 = levelImage - workingImage;


		//Perform closing to close holes
		//Using 5x5 rect structuing element
		strel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
		cv::morphologyEx(workingImage2,workingImage2,cv::MORPH_CLOSE,strel);

		//Save current working image
		workingImage2.copyTo(workingImage3);

		//Clean level by filling holes
		//Flood fill from all corners (just to be safe)
		cv::floodFill(workingImage2,cv::Point(0,0),cv::Scalar(255));
		cv::floodFill(workingImage2,cv::Point(0,levelImage.rows-1),cv::Scalar(255));
		cv::floodFill(workingImage2,cv::Point(levelImage.cols-1,0),cv::Scalar(255));
		cv::floodFill(workingImage2,cv::Point(levelImage.cols-1,levelImage.rows-1),cv::Scalar(255));

		//Invert to get locations of holes and add back in original
		//data
		holes = 255 - workingImage2;
		workingImage3 += holes;


		//Do an opening to clean out small isolated areas
		strel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
		cv::morphologyEx(workingImage3,workingImage3,cv::MORPH_OPEN,strel);
/*		
		//TEST: display window
		cout << "Testing Level: " << i << endl;
		cv::namedWindow("Voxels",CV_WINDOW_AUTOSIZE);
		cv::imshow("Voxels",levelImage);
		cv::waitKey(0);

		cv::namedWindow("Voxels Cleaned",CV_WINDOW_AUTOSIZE);
		cv::imshow("Voxels Cleaned",workingImage3);
		cv::waitKey(0);
*/
		//Update the voxel grid with the cleaned points
		updateVoxelGrid(levelImage,workingImage3,i);

		//Identify boundry pixels
		cv::findContours(workingImage3, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE,cv::Point(0,0));

		//Draw contours
		boundries = cv::Mat::zeros(levelImage.size(),CV_8U);
		for(int j = 0; j < contours.size(); j++){
		//Only draw contours that are larger than 10 pixels in length
		     if(contours[j].size() > 10){
			drawContours(boundries,contours,j,cv::Scalar(255),1,8,hierarchy,0, cv::Point());
		     }
		}
	
		//Drop the boundry down one level
		updateVoxelGrid(boundries,i-1);

/*
		cv::namedWindow("Voxels Bound",CV_WINDOW_AUTOSIZE);
		cv::imshow("Voxels Bound",boundries);
		cv::waitKey(0);
*/
	}
}

void Voxelizer::estimateSurface(int dNorms){

	//Create OctreePointCloudSearch so points in voxels can be found
	pcl::octree::OctreePointCloudSearch<pType> voxelSearch(vInfo.voxelSize);
	voxelSearch.setInputCloud(cloudPtr);
	voxelSearch.addPointsFromInputCloud();

	//Extract the voxel centers
	std::vector<pType, Eigen::aligned_allocator<pType> > voxelList;
	voxelGrid->getOccupiedVoxelCenters(voxelList);

	//Store which points to remove
	std::vector<pType> toRemove;
	//Iterate over each point
	for(int i = 0; i < voxelList.size(); i++){

		int neighborCount = 0;
		std::vector<cv::Point3f> normals;
		//Count the number of neighbors in the 26 surrounding voxels
		pType tmpPt; 

		for(int dx = -1; dx < 2; dx++){
			tmpPt.x = voxelList[i].x + vInfo.voxelSize*dx;
			for(int dy = -1; dy < 2; dy++){
				tmpPt.y = voxelList[i].y + vInfo.voxelSize*dy; 
				for(int dz = -1; dz < 2; dz++){
					tmpPt.z = voxelList[i].z + vInfo.voxelSize*dz;
					if(voxelGrid->isVoxelOccupiedAtPoint(tmpPt) && !(dx == 0 && dy == 0 && dz == 0)){
						//Add opposite direction
						normals.push_back(cv::Point3f(-dx,-dy,-dz));
						neighborCount++;
					}
				}
			}
		}
	
		//Calculate normal
		cv::Point3f norm;
		for(int j = 0; j < normals.size(); j++){
			norm.x += normals[j].x;
			norm.y += normals[j].y;
			norm.z += normals[j].z;
		
		}

		norm.x /= (float)normals.size();
		norm.y /= (float)normals.size();
		norm.z /= (float)normals.size();

		//Normalize normal
		double scale = std::sqrt(norm.x*norm.x + norm.y*norm.y + norm.z*norm.z);

		norm.x /= scale;
		norm.y /= scale;
		norm.z /= scale;
	
		cardinalNorm(norm, dNorms);

		//Set All points in voxel to have this normal
		std::vector<int> pointsInVox;
		voxelSearch.voxelSearch(voxelList[i],pointsInVox);
			
		for(int j = 0; j < pointsInVox.size(); j++){
			cloudPtr->points[pointsInVox[j]].normal_x = norm.x;
			cloudPtr->points[pointsInVox[j]].normal_y = norm.y;
			cloudPtr->points[pointsInVox[j]].normal_z = norm.z;
		}

		if(neighborCount == 26){
			toRemove.push_back(voxelList[i]);
		}
	}

	//Remove points
	for(int i = 0; i < toRemove.size(); i++){
		    std::vector<int> pointIdx;
		    //Find all points in point cloud
		    voxelSearch.voxelSearch(toRemove[i],pointIdx);
		    for(int j = 0; j < pointIdx.size(); j++){
		    	//Remove corresponding points from cloud
			voxelGrid->deleteVoxelAtPoint(pointIdx[j]);
		      }
		    //Delete point from voxel grid
		    voxelGrid->deleteVoxelAtPoint(toRemove[i]);
	}

	std::cout << "Removed " << toRemove.size() << " non surface voxels" << std::endl;
}

void Voxelizer::cardinalNorm(cv::Point3f &norm, int dNorms){

	//determine which cardinal norm the current norm is closest to
	float upXp = norm.dot(cv::Point3f(0.7,0,0.7));
	float upXn = norm.dot(cv::Point3f(-0.7,0,0.7));	
	float upYp = norm.dot(cv::Point3f(0,0.7,0.7));
	float upYn = norm.dot(cv::Point3f(0,-0.7,0.7));
	
	float downXp = norm.dot(cv::Point3f(0.7,0,-0.7));
	float downXn = norm.dot(cv::Point3f(-0.7,0,-0.7));	
	float downYp = norm.dot(cv::Point3f(0,0.7,-0.7));
	float downYn = norm.dot(cv::Point3f(0,-0.7,-0.7));

	float up = norm.dot(cv::Point3f(0,0,1));
	float down = norm.dot(cv::Point3f(0,0,-1));
	
	float xP = norm.dot(cv::Point3f(1,0,0));
	float xN = norm.dot(cv::Point3f(-1,0,0));
	
	float yP = norm.dot(cv::Point3f(0,1,0));
	float yN = norm.dot(cv::Point3f(0,-1,0));


	//Find max value and adjust norm (prefer up or down over sideways)

	cv::Point3f tmpN;
	float max = 0;
    if(dNorms){
	if( upXp > max ) {
		max = upXp;
		tmpN = cv::Point3f(0.7,0,0.7);
	}

	if( upXn > max ) {
		max = upXn;
		tmpN = cv::Point3f(-0.7,0,0.7);
	}

	if( upYp > max ) {
		max = upYp;
		tmpN = cv::Point3f(0,0.7,0.7);
	}

	if( upYn > max ) {
		max = upYn;
		tmpN = cv::Point3f(0,-0.7,0.7);
	}

	if( downXp > max ) {
		max = downXp;
		tmpN = cv::Point3f(0.7,0,-0.7);
	}

	if( downXn > max ) {
		max = downXn;
		tmpN = cv::Point3f(-0.7,0,-0.7);
	}

	if( downYp > max ) {
		max = downYp;
		tmpN = cv::Point3f(0,0.7,-0.7);
	}

	if( downYn > max ) {
		max = downYn;
		tmpN = cv::Point3f(0,-0.7,-0.7);
	}
     }

	if( yP > max ) {
		max = yP;
		tmpN = cv::Point3f(0,1,0);
	}
	if( yN > max ) {
		max = yN;
		tmpN = cv::Point3f(0,-1,0);
	}

	
	if( xP > max ) {
		max = xP;
		tmpN = cv::Point3f(1,0,0);
	}
	if( xN > max ) {
		max = xN;
		tmpN = cv::Point3f(-1,0,0);
	}
	
	if( up > max ) {
		max = up;
		tmpN = cv::Point3f(0,0,1);
	}
	if( down > max ) {
		max = down;
		tmpN = cv::Point3f(0,0,-1);
	}

	norm = tmpN;
}

void Voxelizer::getVoxels(pcl::PointCloud<pType>::Ptr& voxels){

	//Create OctreePointCloudSearch so points in voxels can be found
	pcl::octree::OctreePointCloudSearch<pType> voxelSearch(vInfo.voxelSize);
	voxelSearch.setInputCloud(cloudPtr);
	voxelSearch.addPointsFromInputCloud();

	//Definitions
	pType pt;

	//Extract the voxel centers
	std::vector<pType, Eigen::aligned_allocator<pType> > voxelList;
	voxelGrid->getOccupiedVoxelCenters(voxelList);

	//Fill the write cloud with the voxel centers
	for(int i = 0; i < voxelList.size(); i++){

		//Estimate color
		double r = 0.0;
		double g = 0.0;
		double b = 0.0;
		double nx = 0.0;
		double ny = 0.0;
		double nz = 0.0;
		double nInVox = 0;
		std::vector<int> pointsInVox;
		voxelSearch.voxelSearch(voxelList[i],pointsInVox);
	
		if(pointsInVox.size() == 0) continue;

		for(int j = 0; j < pointsInVox.size(); j++){
			r += cloudPtr->points[pointsInVox[j]].r;
			g += cloudPtr->points[pointsInVox[j]].g;
			b += cloudPtr->points[pointsInVox[j]].b;
			nx += cloudPtr->points[pointsInVox[j]].normal_x;
			ny += cloudPtr->points[pointsInVox[j]].normal_y;
			nz += cloudPtr->points[pointsInVox[j]].normal_z;
			nInVox += 1.0;
		}

		r *= 1/nInVox;
		b *= 1/nInVox;
		g *= 1/nInVox;
		nx *= 1/nInVox;
		ny *= 1/nInVox;
		nz *= 1/nInVox;


		voxelList[i].r = (int)r;
		voxelList[i].g = (int)g;
		voxelList[i].b = (int)b;
		voxelList[i].normal_x = nx;
		voxelList[i].normal_y = ny;
		voxelList[i].normal_z = nz;
		
		voxels->push_back( voxelList[i] );	

	}

}

void Voxelizer::getVoxels(pcl::PointCloud<pType>::Ptr& knownVoxels, std::vector<int> &ptsInKnown,
			  pcl::PointCloud<pType>::Ptr& estimatedVoxels, std::vector<int> &ptsInEst){

	//Create OctreePointCloudSearch so points in voxels can be found
	pcl::octree::OctreePointCloudSearch<pType> voxelSearch(vInfo.voxelSize);
	voxelSearch.setInputCloud(cloudPtr);
	voxelSearch.addPointsFromInputCloud();
	
	//Definitions
	pType pt;
	
	//Extract the voxel centers
	std::vector<pType, Eigen::aligned_allocator<pType> > voxelList;
	voxelGrid->getOccupiedVoxelCenters(voxelList);
	
	//Fill the write cloud with the voxel centers
	for(int i = 0; i < voxelList.size(); i++){

		//Estimate color
		double r = 0.0;
		double g = 0.0;
		double b = 0.0;
		double nx = 0.0;
		double ny = 0.0;
		double nz = 0.0;
		double nInVox = 0;
		std::vector<int> pointsInVox;
		voxelSearch.voxelSearch(voxelList[i],pointsInVox);
	
		if(pointsInVox.size() == 0) continue;

		for(int j = 0; j < pointsInVox.size(); j++){
			r += cloudPtr->points[pointsInVox[j]].r;
			g += cloudPtr->points[pointsInVox[j]].g;
			b += cloudPtr->points[pointsInVox[j]].b;
			nx += cloudPtr->points[pointsInVox[j]].normal_x;
			ny += cloudPtr->points[pointsInVox[j]].normal_y;
			nz += cloudPtr->points[pointsInVox[j]].normal_z;
			nInVox += 1.0;
		}

		r *= 1/nInVox;
		b *= 1/nInVox;
		g *= 1/nInVox;
		nx *= 1/nInVox;
		ny *= 1/nInVox;
		nz *= 1/nInVox;

		voxelList[i].r = (int)r;
		voxelList[i].g = (int)g;
		voxelList[i].b = (int)b;
		voxelList[i].normal_x = nx;
		voxelList[i].normal_y = ny;
		voxelList[i].normal_z = nz;
		
		if( (r != 0.0) && (b != 0.0) && (g != 0.0) ){
			knownVoxels->push_back( voxelList[i] );	
			ptsInKnown.push_back(pointsInVox.size());
		}else{

			estimatedVoxels->push_back( voxelList[i] );
			ptsInEst.push_back(pointsInVox.size());
		}

	}

}


void Voxelizer::fillFromLevel(cv::Mat &levelImage, int level){

	int xRange = (int)((vInfo.maxX-vInfo.minX)/vInfo.voxelSize);
	int yRange = (int)((vInfo.maxY-vInfo.minY)/vInfo.voxelSize);

	levelImage = cv::Mat::zeros(yRange,xRange,CV_8U);

	fillImageFromLevel(levelImage,level);
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  fillImageFromLevel
 *  Description:  Fills a cv::Mat that represents one level of the voxel space with 
 *  			binary information if the voxel is present or not
 *
 *  @param1 levelImage: Image that is the size of a full level of the voxel space
 *  				height = (maxY - minY)/voxelSize
 *  				width = (maxX - minX)/voxelSize
 *
 *  @param2 level: The level of the voxel space to extract
 * =====================================================================================
 */
void Voxelizer::fillImageFromLevel(cv::Mat &levelImage, int level){
   //There might be a faster way to do this, currently unsure how
	double X,Y,Z;

	//Use voxel center as point
	Z = vInfo.minZ + ((double)level) * vInfo.voxelSize - vInfo.voxelSize/2;
	
	std::cout << level << std::endl;
	std::cout << Z << std::endl;
	
	//Iterate over whole image
	for(int i = 0; i < levelImage.cols; i++){
		for(int j = 0; j < levelImage.rows; j++){
			//Use voxel center as point
			X = vInfo.minX + ((double)i) * vInfo.voxelSize	- vInfo.voxelSize/2;
			Y = vInfo.minY + ((double)j) * vInfo.voxelSize	- vInfo.voxelSize/2;

			if(voxelGrid->isVoxelOccupiedAtPoint(X,Y,Z)){
				levelImage.at<uchar>(j,i) = 255;
			}
		}
	}
}


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  hitmiss
 *  Description:  impliments the hit or miss transform
 *
 *  @param1 src:  Source image, 8 bit single-channel binary image
 *  @param2 dst:  Destination image
 *  @param3 kernel: The hit or miss kernal. 1 = foreground, -1 = background, 0 = ignore
 * =====================================================================================
 */
void Voxelizer::hitmiss(cv::Mat& src, cv::Mat& dst, cv::Mat& kernel){
  CV_Assert(src.type() == CV_8U && src.channels() == 1);

    cv::Mat k1 = (kernel == 1) / 255;
    cv::Mat k2 = (kernel == -1) / 255;

    cv::normalize(src, src, 0, 1, cv::NORM_MINMAX);

    cv::Mat e1, e2;
    cv::erode(src, e1, k1, cv::Point(-1,-1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::erode(1-src, e2, k2, cv::Point(-1,-1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));

    src *= 255;
    dst = (e1 & e2)*255;
}


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  updateVoxelGrid
 *  Description:  Updates the internal voxel grid by comparing the level image to the
 *  			working image
 *
 *  @param1 levelImage: The starting binary image
 *  @param2 workingImage: The cleaned binary image
 *  @param3 level: The level of the voxel grid
 * =====================================================================================
 */
void Voxelizer::updateVoxelGrid(cv::Mat& levelImage, cv::Mat& workingImage, int level){

	//Definitions
	cv::Mat xorImg;
	cv::Mat locs;
        pcl::PointCloud<pType>::Ptr input(new pcl::PointCloud<pType>);

	//Find differences using xor
	cv::bitwise_xor(levelImage,workingImage,xorImg);
	//Find nonzero elements
	cv::findNonZero(xorImg,locs);

//Iterate over elements to find change locations
//determine if the change location was an addition or subtraction from the
//voxel grid

	//Precalc Z level as that does not change
	double Z = vInfo.minZ + ((double)level) * vInfo.voxelSize - vInfo.voxelSize/2;
	int newPts  = 0;
	int rmPts = 0;

	//Create OctreePointCloudSearch so points in voxels can be found
	pcl::octree::OctreePointCloudSearch<pType> voxelSearch(vInfo.voxelSize);
	voxelSearch.setInputCloud(cloudPtr);
	voxelSearch.addPointsFromInputCloud();

	for(int i = 0; i < locs.total(); i++){
		cv::Point loc_P = locs.at<cv::Point>(i);
		std::vector<int> pointIdx;

		//Calculate point location in Voxel Grid
		pType newPt;
			//Use voxel center
			newPt.x =  vInfo.minX + ((double)loc_P.x)*vInfo.voxelSize - vInfo.voxelSize/2;
			newPt.y =  vInfo.minY + ((double)loc_P.y)*vInfo.voxelSize - vInfo.voxelSize/2; 
			newPt.z = Z;
		//If the value is 255, add to points
		if(workingImage.at<uchar>(loc_P)){
			//Update voxelgrid and input point cloud
			voxelGrid->addPointToCloud(newPt,cloudPtr);
			newPts++;
		//Otherwise remove from voxelGrid and input point cloud
		}else{
		    if(voxelGrid->isVoxelOccupiedAtPoint(newPt)){
			std::cout << "occupado" << std::endl;
		    }

		    //Find all points in point cloud
		    voxelSearch.voxelSearch(newPt,pointIdx);
		    for(int j = 0; j < pointIdx.size(); j++){
		    	//Remove corresponding points from cloud
			voxelGrid->deleteVoxelAtPoint(pointIdx[j]);
		      }
		    //Delete point from voxel grid
		    voxelGrid->deleteVoxelAtPoint(newPt);
			rmPts++;
		}
	}

	std::cout << "Adding " << newPts << " voxels" << std::endl;
	std::cout << "Removing " << rmPts << " voxels" << std::endl;
}


void Voxelizer::findNearestK(pType point, int k, std::vector<int> &k_inds, std::vector<float> &k_dist){


	//Init vectors
	k_inds.resize(k);
	k_dist.resize(k);

	//Create OctreeSearch so points can be found
	pcl::octree::OctreePointCloudSearch<pType> voxelSearch(vInfo.voxelSize);
	voxelSearch.setInputCloud(cloudPtr);
	voxelSearch.addPointsFromInputCloud();

	//Do search
	voxelSearch.nearestKSearch(point, k, k_inds, k_dist);
}


void Voxelizer::findApproxNearest(pType point, int &inds, float &dist){


	//Create OctreeSearch so points can be found
	pcl::octree::OctreePointCloudSearch<pType> voxelSearch(vInfo.voxelSize);
	voxelSearch.setInputCloud(cloudPtr);
	voxelSearch.addPointsFromInputCloud();

	//Do search
	voxelSearch.approxNearestSearch(point, inds, dist);
	
}
/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  updateVoxelGrid
 *  Description:  Updates the internal voxel grid by adding positive values from the
 *  			working image to the point cloud at the specified level
 *
 *  @param1 workingImage: The image containing points to be added
 *  @param3 level: The level of the voxel grid
 * =====================================================================================
 */
void Voxelizer::updateVoxelGrid(cv::Mat& workingImage, int level){

	//Definitions
	cv::Mat locs;
	//Find nonzero elements
	cv::findNonZero(workingImage,locs);

//Iterate over elements and add to voxel

	//Precalc Z level as that does not change
	double Z = vInfo.minZ + ((double)level) * vInfo.voxelSize - vInfo.voxelSize/2;
	int newPts = 0;

	for(int i = 0; i < locs.total(); i++){
		cv::Point loc_P = locs.at<cv::Point>(i);
		std::vector<int> pointIdx;

		//Calculate point location in Voxel Grid
		pType newPt;
			//Use voxel center
			newPt.x =  vInfo.minX + ((double)loc_P.x)*vInfo.voxelSize - vInfo.voxelSize/2;
			newPt.y =  vInfo.minY + ((double)loc_P.y)*vInfo.voxelSize - vInfo.voxelSize/2; 
			newPt.z = Z;
			
		//Add to voxelGrid	
		voxelGrid->addPointToCloud(newPt,cloudPtr);
		newPts++;
	}

	std::cout << "Adding " << newPts << " voxels" << std::endl;
}


bool Voxelizer::isVoxelOccupied(pType point){
	return voxelGrid->isVoxelOccupiedAtPoint(point);	
}
//TESTING STUFFS

void Voxelizer::testRayCast(){

	pcl::octree::OctreePointCloudSearch<pType> rayCast(vInfo.voxelSize);
	rayCast.setInputCloud(cloudPtr);
	rayCast.addPointsFromInputCloud();


	Eigen::Vector3f origin( vInfo.minX, vInfo.minY, vInfo.minZ);
	Eigen::Vector3f direction( vInfo.maxX , vInfo.maxY , vInfo.maxZ );
	std::vector<int> voxelInds;
	std::vector<pType, Eigen::aligned_allocator<pType> > voxelList;

	//rayCast.getIntersectedVoxelCenters(origin, direction, voxelList);

	voxelGrid->getApproxIntersectedVoxelCentersBySegment(origin, direction, voxelList, 0.8);

	for(int i = 0 ; i < voxelList.size(); i++){
	 std::cout << voxelList[i].x << " " << voxelList[i].y << " " << voxelList[i].z << std::endl;
	}
	std::cout << voxelInds.size() << std::endl;

}
