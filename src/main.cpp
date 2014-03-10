/*
 * =====================================================================================
 *
 *       Filename:  main.cpp
 *
 *    Description:  Usage is 
*			voxelQuality plyPath transPaths visPaths voxelSize numMaterials depth
*
*		    plyPath - The path to a single building cropped from a ply file
*		    transPaths - The path to a txt file containing the full path to each projection matrix
*		    visPaths - The path to a txt file containing the full path to each image
*		    voxelSize - The desired voxel size in world coordinate units
*		    numMaterials - An estimate of the number of materials on the strucutre
*		    depth - The length along a projected ray that will be used to consider 
*			    if a voxel is occluded or not. Default is 1, 2 is better for taller buildings.
*		    useDiagonalNorms - Flag to turn diangonal norms on, default is 0 (false).
*		    useLightness - use L from HSL color default is 0
 *
 *        Version:  1.0
 *        Created:  01/25/2014 03:59:33 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  David Nilosek (), drn2369@cis.rit.edu
 *        Company:  Rochester Institute of Technology
 *
 * =====================================================================================
 */
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing_rgb.h>

#include <pcl/surface/marching_cubes.h>
#include <pcl/surface/marching_cubes_hoppe.h>
#include <pcl/surface/marching_cubes_rbf.h>
#include <pcl/surface/grid_projection.h>
#include <pcl/io/vtk_lib_io.h>

#include <opencv/highgui.h>

#include "voxelizer.h"
#include "camera.h"
#include "IO.h"
#include "RayCaster.h"


#define VAR_WEIGHT 0.1
#define DIS_WEIGHT 100

#define SQ_DIST 0.09 //should be based on voxel size....
#define CLR_DIST 10
#define CLR_DIST2 5
#define NRM_DIST 0.06

struct voxelInfo{
	std::vector<cv::Vec3f> colors;
	cv::Vec3b AvgColor;
	double colorVariance;
	double nearestDist;
	double energy;
	//bool isOccluder;
};


void writeDualContouring(std::string filePath, double minZ, pcl::PointCloud<pType>::Ptr& cloud){

	filePath += ".xyzn";
	
	//Write out file
	ofstream write;
	write.open(filePath.c_str());

	write << "# ground " << minZ << std::endl;
	
	for(int i = 0; i < cloud->size(); i++){

		pType pt = cloud->points[i];
		write << pt.x << " " << pt.y << " " << pt.z << " " 
		      << pt.normal_x << " " << pt.normal_y << " " << pt.normal_z << std::endl;

	}

	write.close();
}


void getWeightedColor(cv::Mat &image,cv::Point2f imgPt, cv::Vec3f &colorVec, cv::Mat &kernel){

		//Get the color
		for(int i = 0; i < kernel.cols; i++){
			for(int j = 0; j < kernel.rows;j++){

				//Get location
				cv::Point2f tmpPt = imgPt;
				tmpPt.x += i - kernel.cols/2;
				tmpPt.y += j - kernel.rows/2;

				//Get color
				cv::Vec3f tmpClr = image.at<cv::Vec3f>(tmpPt);	
	
				//Add weighted color
				colorVec += tmpClr*kernel.at<float>(j,i);			
			}
		}
	

}

bool customRegionGrowing(const pType& point_a, const pType& point_b, float squared_distance){

	Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.normal, point_b_normal = point_b.normal;

	float dr = point_a.r - point_b.r; 
	float dg = point_a.g - point_b.g;
	float db = point_a.b - point_b.b;

	float clrDst = std::sqrt(dr*dr + dg*dg + db*db);	
	if(squared_distance < SQ_DIST){

		if( clrDst < CLR_DIST){
			return(true);
		}
		if( std::fabs(point_a_normal.dot(point_b_normal)) < NRM_DIST){
			return(true);
		}

	}else{

		if( clrDst < CLR_DIST2){
			return(true);
		}

	}
	return(false);	
}

void calculateEnergy(voxelInfo &vInf){

	//Calculate enegery using color variance and 
	//nearest dist
	if( vInf.colorVariance == -1){
		vInf.energy = -1;
	}else{
		vInf.energy = std::exp( -1*(vInf.colorVariance/VAR_WEIGHT)) * std::exp(-1*(vInf.nearestDist/DIS_WEIGHT));
	}

}

void createMask(Camera cam, Voxelizer::voxelInfo vInfo, cv::Size sz, cv::Mat &mask){

		//Create two masks
		cv::Mat maskB = cv::Mat::zeros(sz,CV_8U); 
		cv::Mat maskT = cv::Mat::zeros(sz,CV_8U); 


		//Mask will be bounding box of voxel cube
		cv::Point2f b1,b2,b3,b4,t1,t2,t3,t4;

		cam.world2image(cv::Point3f(vInfo.minX,vInfo.minY,vInfo.minZ),b1);
		cam.world2image(cv::Point3f(vInfo.minX,vInfo.maxY,vInfo.minZ),b2);
		cam.world2image(cv::Point3f(vInfo.maxX,vInfo.maxY,vInfo.minZ),b3);
		cam.world2image(cv::Point3f(vInfo.maxX,vInfo.minY,vInfo.minZ),b4);
		
	
		cam.world2image(cv::Point3f(vInfo.minX,vInfo.minY,vInfo.maxZ),t1);
		cam.world2image(cv::Point3f(vInfo.minX,vInfo.maxY,vInfo.maxZ),t2);
		cam.world2image(cv::Point3f(vInfo.maxX,vInfo.maxY,vInfo.maxZ),t3);
		cam.world2image(cv::Point3f(vInfo.maxX,vInfo.minY,vInfo.maxZ),t4);

		//Draw lines on mask
		cv::line(maskB,b1,b2,cv::Scalar(255));
		cv::line(maskB,b2,b3,cv::Scalar(255));
		cv::line(maskB,b3,b4,cv::Scalar(255));
		cv::line(maskB,b4,b1,cv::Scalar(255));
		
		cv::line(maskT,t1,t2,cv::Scalar(255));
		cv::line(maskT,t2,t3,cv::Scalar(255));
		cv::line(maskT,t3,t4,cv::Scalar(255));
		cv::line(maskT,t4,t1,cv::Scalar(255));
	
		//Fill masks
		cv::Point bottomMid, topMid;

		bottomMid.x = (b3.x - b1.x)/2 + b1.x;
		bottomMid.y = (b3.y - b1.y)/2 + b1.y;

		topMid.x = (t3.x - t1.x)/2 + t1.x;
		topMid.y = (t3.y - t1.y)/2 + t1.y;
		
		cv::floodFill(maskB, bottomMid, cv::Scalar(255));	
		cv::floodFill(maskT, topMid, cv::Scalar(255));	
		
		//Combine masks
		cv::bitwise_or(maskT,maskB,mask);	

}

void createVoxelCube( Voxelizer estSurface, pcl::PointCloud<pType>::Ptr& voxelCloud){


	cv::Mat zeroLevel;

	estSurface.fillFromLevel(zeroLevel,0);
	
	//Get contours
	std::vector<std::vector<cv::Point> > contours;
	std::vector<std::vector<cv::Point> > contoursPoly;
	cv::findContours(zeroLevel,contours,CV_RETR_LIST,CV_CHAIN_APPROX_NONE);

	std::cout << "Num contours: " << contours.size() << std::endl;
	//Draw contours
	cv::Mat mask = cv::Mat::zeros(zeroLevel.size(),CV_8U);

	for(int i = 0; i < contours.size(); i++){
		
		//Make sure we are large
		if(contours[i].size() < 50) continue;

		//Approx the point	
		std::vector<cv::Point> approx;
	
		cv::approxPolyDP(contours[i],approx,cv::arcLength(cv::Mat(contours[i]),true) * 0.02,true);

		//store
		contoursPoly.push_back(approx);	
	}
	for(int i = 0; i < contoursPoly.size(); i++){
		cv::drawContours(mask,contoursPoly,i,cv::Scalar(255),CV_FILLED);
	}


	//Fill cloud
	Voxelizer::voxelInfo vInfo = estSurface.getVoxelInfo();

	//Find nonzero elements
	cv::Mat locs;
	cv::findNonZero(mask,locs);

	//Iterate over nonzero elements
	std::vector<pType> pointsToAdd(locs.total());

	for(int i = 0; i < locs.total(); i++){

	   cv::Point loc_P = locs.at<cv::Point>(i);
	   
	      //If it exists
	      if(mask.at<uchar>(loc_P)){
			//Calculate point location in Voxel Grid
			pType newPt;

			//Use voxel center
			newPt.x =  vInfo.minX + ((double)loc_P.x)*vInfo.voxelSize - vInfo.voxelSize/2;
			newPt.y =  vInfo.minY + ((double)loc_P.y)*vInfo.voxelSize - vInfo.voxelSize/2; 
			newPt.z = 0;
		
			//Store point
			pointsToAdd[i] = newPt;
		}
	}

	//Calculate number of Z levels
	double zRange = vInfo.maxZ - vInfo.minZ;
	int zSize = (int)(zRange/vInfo.voxelSize + 1);

	//Fill in each level
	for(int i = 0; i < zSize; i++){
		//Calculate Z level
	       	double Z = i*vInfo.voxelSize + vInfo.voxelSize/2 + vInfo.minZ; 
		//Fill in each point at this level
		for(int j = 0; j < pointsToAdd.size(); j++){

			pType ptToAdd = pointsToAdd[j];

			ptToAdd.z = Z;

			voxelCloud->push_back(ptToAdd); 
		} 
	}
}

void createVoxelCube( Voxelizer::voxelInfo vInfo,  pcl::PointCloud<pType>::Ptr& voxelCloud){

	//Calculate sides of cube
	double xRange = vInfo.maxX - vInfo.minX;
	double yRange = vInfo.maxY - vInfo.minY;
	double zRange = vInfo.maxZ - vInfo.minZ;

	double voxelSize = vInfo.voxelSize;
	int xSize = (int)(xRange/voxelSize + 1);
	int ySize = (int)(yRange/voxelSize + 1);
	int zSize = (int)(zRange/voxelSize + 1);
		
	//Fill cloud
	double x,y,z;
	pType pointToAdd;
	for(int i  = 0; i < xSize; i++){
		//Calculate X
        	x = i*voxelSize + voxelSize/2 + vInfo.minX;
	    for(int j = 0; j < ySize; j++){
	        //Calculate Y
	    	 y = j*voxelSize + voxelSize/2 + vInfo.minY;
	       for(int k = 0; k < zSize; k++){
	       	   //Calculate Z 
	       	   z = k*voxelSize + voxelSize/2 + vInfo.minZ; 
		  
		  //Create point
		  pointToAdd.x = x;
		  pointToAdd.y = y;
		  pointToAdd.z = z;

		  //Add point to cloud
		  voxelCloud->push_back(pointToAdd);
	       }
	    }
	}
}

bool getColorStatistics(std::vector<cv::Vec3f> colors, cv::Vec3f &AvgColor, cv::Vec3f &colorVariance, int &numIntersections){


	//Iterate through colors and calculate number of intersections and mean.
	double blue = 0.0;
	double red = 0.0;
	double green = 0.0;

	for(int i = 0; i < colors.size(); i++){
		double tBlue = (double)colors[i].val[0]/360;
		double tGreen = (double)colors[i].val[1];
		double tRed = (double)colors[i].val[2];
		
		//If there is a color (yeah pure black will be ignored..)
		if( tBlue || tRed || tGreen ){
			blue += tBlue;
			red += tRed;
			green += tGreen;

			numIntersections++;
		}
	}
	
	//If there are no intersections, return false
	if(numIntersections == 0){
		return false;
	}

	//If it is greater than 0, calculate avg color
		blue /=  numIntersections;	
		red /=  numIntersections;
		green /= numIntersections;

	//Iterative through colors again and calculate variance
	double bVar = 0.0;
	double rVar = 0.0;
	double gVar = 0.0;

	for(int i = 0; i < colors.size(); i++){
		
		double tBlue = (double)colors[i].val[0]/360;
		double tGreen = (double)colors[i].val[1];
		double tRed = (double)colors[i].val[2];

		if( tBlue || tRed || tGreen ){
			bVar += ( (tBlue - blue) * (tBlue - blue) );
			rVar += ( (tRed - red) * (tRed - red) );
			gVar += ( (tGreen - green) * (tGreen - green) );

		}
	}

	//Calculate variance
	bVar = bVar/numIntersections;
	rVar = rVar/numIntersections;
	gVar = gVar/numIntersections;

	//Fill out input variables
	AvgColor.val[0] = (float)blue;
	AvgColor.val[1] = (float)green;
	AvgColor.val[2] = (float)red;

	colorVariance.val[0] = (float)bVar;
	colorVariance.val[1] = (float)gVar;
	colorVariance.val[2] = (float)rVar;

	return true;
}

double hue2rgb(double p, double q, double t){

	if( t < 0 ) t += 1;
	if( t > 1) t -= 1;

	if(t < 0.166666) return ( p + ( q - p ) * 6 * t);
	if(t < 0.5) return q;
	if(t < 0.666666) return (p + (q - p) * (0.666666 - t)*6);
	return p;
}

void hls2bgr(cv::Vec3f HLS, cv::Vec3b &BGR){

	float H = HLS.val[0];
	float L = HLS.val[1];
	float S = HLS.val[2];

	float r,g,b;

	if( S == 0.0){
	   r=1;g=1;b=1;
	}else{

	 double q = L < 0.5 ? L * (1 + S) : L + S - L * S;
	 double p = 2 * L - q;

	 r = hue2rgb(p,q,H + 0.333333);
	 g = hue2rgb(p,q,H);
	 b = hue2rgb(p,q,H - 0.333333);
	}


	//Fill in BGR
	BGR.val[0] = (int)(b*255);
	BGR.val[1] = (int)(g*255);
	BGR.val[2] = (int)(r*255);


}

void bgr2hls(cv::Vec3b BGR, cv::Vec3f &HLS){

	float b = ((float)BGR.val[0])/255;
	float g = ((float)BGR.val[1])/255;
	float r = ((float)BGR.val[2])/255;


	//Find min and max 
	float min = 1.1;
	float max = -0.1;
	int maxID;

	if( b < min) min = b; 
	if( b > max){ max = b; maxID = 3;}

	if( g < min) min = g;
	if( g > max){ max = g; maxID = 2;}

	if( r < min) min = r;
	if( r > max){ max = r; maxID = 1;}

	float h,s;
	float l = (max + min)/2;

	if(max == min){
	  h = 0;
	  s = 0;
	}else{

	   float d = max - min;
           s = l > 0.5 ? d / (2 - max - min) : d / (max + min);

	   switch(maxID){
		case 1: h = (g - b) / d + ( g < b ? 6 : 0); break; //r
		case 2: h = (b - r) / d + 2; break; //g
		case 3: h = (r - g) / d + 4; break; //b
	   }
	
	   h /= 6;
	}

	HLS.val[0] = h;
        HLS.val[1] = l;
        HLS.val[2] = s;
}

int main ( int argc, char *argv[] )
{
	//Definitions
	pcl::PointCloud<pType>::Ptr cloud(new pcl::PointCloud<pType>);
	pcl::PointCloud<pType>::Ptr cloudFiltered(new pcl::PointCloud<pType>);
	pcl::PointCloud<pType>::Ptr voxelCloudFull(new pcl::PointCloud<pType>);
	pcl::PointCloud<pType>::Ptr voxelCloudCx(new pcl::PointCloud<pType>);
	pcl::PointCloud<pType>::Ptr voxelCloud(new pcl::PointCloud<pType>);
	pcl::PointCloud<pType>::Ptr knownVoxels(new pcl::PointCloud<pType>);
	pcl::PointCloud<pType>::Ptr knownVoxelsE(new pcl::PointCloud<pType>);
	pcl::PointCloud<pType>::Ptr estimatedVoxels(new pcl::PointCloud<pType>);
	pcl::PointCloud<pType>::Ptr estimatedVoxelsE(new pcl::PointCloud<pType>);
	pcl::PointCloud<pType>::Ptr estimatedVoxelsT(new pcl::PointCloud<pType>);
	pcl::PointCloud<pType>::Ptr estimatedVoxelsC(new pcl::PointCloud<pType>);
	pcl::PointCloud<pType>::Ptr knownAndEstVoxels(new pcl::PointCloud<pType>);
	pcl::PointCloud<pType>::Ptr knownAndEstVoxelsE(new pcl::PointCloud<pType>);
	pcl::PointCloud<pType>::Ptr knownAndEstVoxelsC(new pcl::PointCloud<pType>);
	pcl::PointCloud<pType>::Ptr knownAndEstSurf(new pcl::PointCloud<pType>);

	pcl::PLYReader reader;	
	pcl::PLYWriter writer;
	std::stringstream convert;
	std::stringstream convert2;
	std::stringstream convert3;
	std::stringstream convert4;
	std::stringstream convert5;
	float voxelSize;
	int numMat;
	int depth;
	int dNorms;
	int useL;
	std::vector<std::string> projFileList;
	std::vector<std::string> imgFileList;
	std::vector<Camera> cameras;
/* ---------------- READ IN IMAGE INFORMATION  -----------------*/

	//Read in projection matrix location list
	ifstream file(argv[2]);

	//Read in
	std::string tmpP;
	while(file >> tmpP){
		projFileList.push_back(tmpP);
	}
	file.close();
	
	//Read in image location list
	ifstream fileIm(argv[3]);

	//Read in
	std::string tmpI;
	while(fileIm >> tmpI){
		imgFileList.push_back(tmpI);
	}
	fileIm.close();


	//Read each projection matrix file and create camera
	for(int i = 0; i < projFileList.size(); i++){
		//Read in camera
		TXT_Reader txtReader(projFileList[i].c_str());
		
		//Create camera
		Camera cam(txtReader.getCam());
		cameras.push_back(cam);
	}

/* ---------------- CLEAN INPUT POINT CLOUD -----------------*/
	//Read Point cloud in
	reader.read(argv[1],*cloud);
	std::cout << "Input point cloud has: " << cloud->points.size() << " points" << std::endl;

	//Get voxel size
	convert << argv[4];
	convert >> voxelSize;

	//Get numMat
        convert2 << argv[5];
	convert2 >> numMat;

	convert3 << argv[6];
	convert3 >> depth;

	convert4 << argv[7];
	convert4 >> dNorms;
	
	convert5 << argv[8];
	convert5 >> useL;
	//Clean point cloud using radius removal (require at least 4 points within 2x the voxel size);
	pcl::RadiusOutlierRemoval<pType> radiusRemove;
	radiusRemove.setInputCloud(cloud);
	radiusRemove.setRadiusSearch(2*voxelSize);
	radiusRemove.setMinNeighborsInRadius(4);

		//Apply filter
		radiusRemove.filter(*cloudFiltered);

	std::cout << "After filtering cloud has: " << cloudFiltered->size() << " points" << std::endl;
/* ---------------- SET UP VOXEL POINT CLOUDS ---------------- */

	//Voxelize point cloud
	std::vector<int> knownVoxelPts;
	std::vector<int> estVoxelPts;
	Voxelizer voxelize(cloudFiltered, voxelSize);

	//Clean it
	voxelize.depthClean();	

	//voxelize.writeVoxelCenters("depthClean.ply");
	//Fill cloud for future use	
	createVoxelCube(voxelize, voxelCloudFull);
	
	 //Get bounding box
	Voxelizer::voxelInfo vInfo = voxelize.getVoxelInfo();
	std::vector<double> boundingBox;
	boundingBox.push_back(vInfo.minX);
	boundingBox.push_back(vInfo.maxX);
	boundingBox.push_back(vInfo.minY);
	boundingBox.push_back(vInfo.maxY);
	boundingBox.push_back(vInfo.minZ);
	boundingBox.push_back(vInfo.maxZ);

	//Get points
	voxelize.getVoxels(knownAndEstSurf);

	//Make new voxelizer
	Voxelizer knownAndEstSurfV(knownAndEstSurf,voxelSize,boundingBox);

	//Estimate the surface
	knownAndEstSurfV.estimateSurface(dNorms);
	
	knownAndEstSurfV.getVoxels(knownVoxels,knownVoxelPts,estimatedVoxels,estVoxelPts);
	knownAndEstSurfV.getVoxels(knownVoxelsE,knownVoxelPts,estimatedVoxelsE,estVoxelPts);

	 
	double maxNum = (double)*std::max_element(knownVoxelPts.begin(),knownVoxelPts.end());

	//writer.write<pType>("knownVoxels.ply", *knownVoxels);
	//writer.write<pType>("estimatedVoxels.ply", *estimatedVoxels);
	//writer.write<pType>("knownVoxelsE.ply", *knownVoxels);
	//writer.write<pType>("estimatedVoxelsE.ply", *estimatedVoxels);


	//Write centers
	//std::string cenPath = "radiusCleanedCenters.ply";
	//voxelize.writeVoxelCenters(cenPath);

/* ---------------- CALCULATE ENERGY FOR ESTIMATED SURFACE  ---------------- */

	//Create structure to hold voxel information
	std::vector<voxelInfo> estimatedInfo(estimatedVoxels->size());

	//Init the info
	for(int i = 0; i < estimatedInfo.size(); i++){
		voxelInfo newVInfo;

		//Init the vector
		newVInfo.colors.resize(imgFileList.size());

		//Init the values (-1 means undefined)
		newVInfo.AvgColor = cv::Vec3b(0,0,0);
		newVInfo.colorVariance = -1;
		newVInfo.nearestDist = -1;
		newVInfo.energy = -1;

		//Store voxel info
		estimatedInfo[i] = newVInfo;
	}

	//Gather colors from imagery
	
	//Set up ray caster for occlusion model
	RayCaster oclC(estimatedVoxels,voxelSize,boundingBox);

	//Precalculate gaussian weights	
	int kernelSize = 3;	
	cv::Mat gaussKernel = cv::Mat::zeros(kernelSize,kernelSize,CV_32F);	

	float sigma = 0.3*((kernelSize-2)/2 - 1) + 0.8;

	float totalWeight = 0.0;	
	for(int i = 0; i < kernelSize; i++){
		for(int j = 0; j < kernelSize; j++){
				
			float xWeight = std::exp(-1*std::pow((i - (kernelSize-1)/2),2)/std::pow(2*sigma,2));
			float yWeight = std::exp(-1*std::pow((j - (kernelSize-1)/2),2)/std::pow(2*sigma,2));
			float weight = xWeight*yWeight;
			
			totalWeight += weight;
			gaussKernel.at<float>(j,i) = weight;
		}
	}

	//Scale by total weight
	gaussKernel *= 1/totalWeight;

	for(int i = 0; i < imgFileList.size(); i++){
		

		std::cout << "Processing: " << imgFileList[i] << std::endl;
		//Read in image (image list and cameras are in same order)
		cv::Mat image = cv::imread(imgFileList[i]);
		image.convertTo(image,CV_32FC3,1./255);

		//Convert to HLS (bring 0-255 to 0-1)
		cv::cvtColor(image,image,CV_BGR2HLS);
		
		//Get camera
		Camera cam = cameras[i];

		//Project each voxel from estimated surface to get color
			//MAYBE USE WEIGHTED AVG FROM VOXEL AREA
	
		cv::Point2f imgPt;
		for(int j = 0; j < estimatedVoxels->size(); j++){

			pType pt = estimatedVoxels->points[j];

			//Project world point into frame
			std::vector<int> intersections;
			cv::Point3f origin,direction;
			cv::Point3f worldPt(pt.x,pt.y,pt.z);
			cam.world2image(worldPt,imgPt);
			
			//Get ray
			cam.getRay(imgPt, origin, direction);

			//Cast ray into voxels, get only the first voxel
			oclC.castRay(origin,direction,intersections,depth);

			//START OF TEST//
			/*	pcl::PointCloud<pType>::Ptr testPtCloud(new pcl::PointCloud<pType>);
				pcl::PointCloud<pType>::Ptr testPt(new pcl::PointCloud<pType>);
				pt.r =255;
				pt.b = 0;
				pt.g = 0;
				testPt->push_back(pt);
			for(int k = 0; k < intersections.size(); k++){
				pType tmpPt = estimatedVoxels->points[intersections[k]];

				tmpPt.r = (int)(((float)k/(float)intersections.size())*255);
				tmpPt.g = (int)(((float)k/(float)intersections.size())*255);
				tmpPt.b = (int)(((float)k/(float)intersections.size())*255);
				
				testPtCloud->push_back(tmpPt);
			}
				writer.write<pType>("ray.ply",*testPtCloud);
				writer.write<pType>("pt.ply",*testPt);
			*/
			//END OF TEST//

			//If the first intersection is not this point, consider it occluded
			bool occluded = true;

			for(int k = 0; k < intersections.size(); k++){
				if( j == intersections[k] ){
					occluded = false;
					break;
				}
			}

			if( !occluded ){
				//Get color and save it
				cv::Vec3f colorV;
				getWeightedColor(image,imgPt,colorV,gaussKernel);

				//estimatedInfo[j].colors[i] = image.at<cv::Vec3f>(imgPt);
				estimatedInfo[j].colors[i] = colorV;
			}else{
				estimatedInfo[j].colors[i] = cv::Vec3f(0,0,0);
			}
		}
	}

	//Calculate energy for each voxel
	 Voxelizer knownVoxelizer(knownVoxels,voxelSize,boundingBox);
std::ofstream outEnE;
outEnE.open("energies.txt");

	for(int i = 0; i < estimatedInfo.size(); i++){

		//Get point location
		pType pt;

		pt.x = estimatedVoxels->points[i].x;
		pt.y = estimatedVoxels->points[i].y;
		pt.z = estimatedVoxels->points[i].z;

		//Calculate color variance
		cv::Vec3f AvgColor;
		cv::Vec3f colorVariance;
		cv::Vec3b BGRcolor;
		int numIntersections = 0;

		getColorStatistics(estimatedInfo[i].colors, AvgColor, colorVariance, numIntersections);

		if(numIntersections > 0){
			//Calculate total variance only using H and S
			estimatedInfo[i].colorVariance = std::sqrt( colorVariance.val[0]*colorVariance.val[0] +
				    		       	            colorVariance.val[2]*colorVariance.val[2] );
	
			//Find closest known point
			int ind;
			float dist;
			knownVoxelizer.findApproxNearest(pt, ind, dist);

			//Store distance
			estimatedInfo[i].nearestDist = dist;

			//Convert HLS to BRG and store
			hls2bgr(AvgColor, estimatedInfo[i].AvgColor);
			//estimatedInfo[i].AvgColor = AvgColor;
	
			//Calculate the energy
			calculateEnergy(estimatedInfo[i]);

		}else{
			//Otherwise energy is set to 0.5
			estimatedInfo[i].energy = 0.5;
		}
		//Modulate energy by max num of points in voxel

		//estimatedInfo[i].energy *= std::exp(-1.0/1.0);

		outEnE << estimatedInfo[i].energy << std::endl;
	}
outEnE.close();
	   
	//Colorize point clouds
	pcl::copyPointCloud(*estimatedVoxels,*estimatedVoxelsC);
	for(int i = 0; i < estimatedVoxels->size(); i++){
	
		//Convert RGB to HLS
		cv::Vec3b BGR;
		cv::Vec3f HLS;

		BGR = estimatedInfo[i].AvgColor;
		bgr2hls(BGR,HLS);
	  
            if(useL){  
		estimatedVoxels->points[i].b = (int)(HLS.val[1]*255);
	    }else{
		estimatedVoxels->points[i].b = 0;//(int)(HLS.val[1]*255);
            }
		estimatedVoxels->points[i].g = (int)(HLS.val[2]*255);
		estimatedVoxels->points[i].r = (int)(HLS.val[0]*255);
	    
		estimatedVoxelsC->points[i].b = (int)estimatedInfo[i].AvgColor.val[0];
		estimatedVoxelsC->points[i].g = (int)estimatedInfo[i].AvgColor.val[1];
		estimatedVoxelsC->points[i].r = (int)estimatedInfo[i].AvgColor.val[2];
		
		estimatedVoxelsE->points[i].r = (int)(estimatedInfo[i].energy*255);
		estimatedVoxelsE->points[i].g = (int)(estimatedInfo[i].energy*255);
		estimatedVoxelsE->points[i].b = (int)(estimatedInfo[i].energy*255);
	}

	//Only keep energies 0.5 and up
	for(int i = 0; i < estimatedInfo.size(); i++){
		if(estimatedInfo[i].energy >= 0.5){
			estimatedVoxelsT->push_back(estimatedVoxels->points[i]);
		}
	}

	//Save the cloud
	//writer.write<pType>("estimatedVoxelsEnergy.ply",*estimatedVoxelsE);
	//writer.write<pType>("estimatedVoxelsColor.ply",*estimatedVoxels);

/*
for(int t = 1; t < 11; t++){
stringstream ss2;
double thresh = ((double)t)/10.0 - 0.05;

ss2 << "energyCloudsEst/energyEst_" << thresh << ".ply";

pcl::PointCloud<pcl::PointXYZRGBNormal> outCloud3;
for(int i = 0; i < estimatedInfo.size(); i++){

	//if( voxelDesc[i].energy != -1 && voxelDesc[i].energy > 0.95){
	if( estimatedInfo[i].energy != -1 && estimatedInfo[i].energy > thresh){

		pcl::PointXYZRGB pointToAdd2;

		//if( estimatedInfo[i].AvgColor.val[0] || estimatedInfo[i].AvgColor.val[1] || estimatedInfo[i].AvgColor.val[2]){

//			pointToAdd2.x =  estimatedVoxels->points[i].x;
//			pointToAdd2.y =  estimatedVoxels->points[i].y;
//			pointToAdd2.z =  estimatedVoxels->points[i].z;
	
//			pointToAdd2.b = estimatedInfo[i].AvgColor.val[0];
//			pointToAdd2.g = estimatedInfo[i].AvgColor.val[1];
//			pointToAdd2.r = estimatedInfo[i].AvgColor.val[2];

			outCloud3.push_back(estimatedVoxels->points[i]);
		}
	}

}

	writer.write<pType>(ss2.str(),outCloud3);
}
*/

//Write true color voxels out

*knownAndEstVoxelsC = *knownVoxels;
*knownAndEstVoxelsC += *estimatedVoxelsC;

writer.write<pType>("voxelsTrueColor.ply",*knownAndEstVoxelsC);
/* ------------ FILL KNOWN SURFACE ENERGIES ---------------- */

	for(int i = 0; i < knownVoxelsE->size(); i++){

		//Convert RGB to HLS
		cv::Vec3b BGR;
		cv::Vec3f HLS;

		BGR.val[0] = knownVoxels->points[i].b;
		BGR.val[1] = knownVoxels->points[i].g;
		BGR.val[2] = knownVoxels->points[i].r;
		bgr2hls(BGR,HLS);

	     if(useL){
		knownVoxels->points[i].b = (int)(HLS.val[1]*255);
	     }else{
		knownVoxels->points[i].b = 0;//(int)(HLS.val[1]*255);
	     }
		knownVoxels->points[i].g = (int)(HLS.val[2]*255);
		knownVoxels->points[i].r = (int)(HLS.val[0]*255);
		
		knownVoxelsE->points[i].r = 255;
		knownVoxelsE->points[i].g = 255;
		knownVoxelsE->points[i].b = 255;
	}

/* ------------ COMBINE KNOWN AND ESTIMATED SURFACES ---------------- */
	//Concat known and estimated point clouds
	*knownAndEstVoxels = *knownVoxels;
	*knownAndEstVoxels += *estimatedVoxels;

	*knownAndEstVoxelsE = *knownVoxelsE;
	*knownAndEstVoxelsE += *estimatedVoxelsE;


	//Save

	//writer.write<pType>("voxelColored.ply",*knownAndEstVoxels);
	//writer.write<pType>("voxelEnergies.ply",*knownAndEstVoxelsE);

/*	//Set up voxelInfo for known voxels
	std::vector<voxelInfo> knownInfo(knownVoxels->size());

	//Init
	for(int i = 0; i < knownInfo.size(); i++){

		voxelInfo newVInfo;

		//Init the values (-1 means undefined)
		newVInfo.AvgColor = cv::Vec3b(knownVoxels->points[i].r,
					      knownVoxels->points[i].g,
					      knownVoxels->points[i].b);
		newVInfo.colorVariance = 0;
		newVInfo.nearestDist = 0;
		newVInfo.energy = 1;//std::exp(-1*(double)knownVoxelPts[i]/maxNum -1 );

		//Store voxel info
		knownInfo[i] = newVInfo;
	}

	//Concat voxel infos
	knownInfo.insert(knownInfo.end(), estimatedInfo.begin(), estimatedInfo.end());

	//Create new voxelizer object for concat point cloud
	Voxelizer knownAndEstVox(knownAndEstVoxels,voxelSize,boundingBox);
*/
/* ------------ CALCUALTE CONVEX HULL OF ESTIMATED SURFACE ---------------- */
/*
	std::cout << "Calculating Convex Hull of Estimated Surface" << std::endl;

	pcl::ConvexHull<pType> chull;
	pcl::PointCloud<pType>::Ptr hull(new pcl::PointCloud<pType>);
	std::vector<pcl::Vertices> polygons;
	int dimension = 3;

	chull.setInputCloud(knownAndEstVoxels);
	chull.setDimension(dimension);
	chull.reconstruct(*hull,polygons);
*/
/* ------------ CALCUALTE CONCAVE HULL OF ESTIMATED SURFACE ---------------- */
/*
	std::cout << "Calculating Concave Hull of Estimated Surface" << std::endl;

	pcl::ConcaveHull<pType> cchull;
	pcl::PointCloud<pType>::Ptr hull2(new pcl::PointCloud<pType>);
	std::vector<pcl::Vertices> polygons2;

	cchull.setInputCloud(knownAndEstVoxels);
	cchull.setDimension(dimension);
	cchull.setAlpha(voxelSize*2);
	cchull.reconstruct(*hull2,polygons2);
*/
/* ---------------- FURTHER ESTIMATE SURFACE  ---------------- */
	
	//Create a point cloud with centers at every voxel
	//std::cout << "Creating voxel cube" << std::endl;
	//createVoxelCube(vInfo, voxelCloudFull);
	//createVoxelCube(knownAndEstVox, voxelCloudFull);

/*
	//Create Ray Caster 
	std::vector<double> boundingBox;
	boundingBox.push_back(vInfo.minX);
	boundingBox.push_back(vInfo.maxX);
	boundingBox.push_back(vInfo.minY);
	boundingBox.push_back(vInfo.maxY);
	boundingBox.push_back(vInfo.minZ);
	boundingBox.push_back(vInfo.maxZ);
*/
	//Crop to convex hull
/*	std::cout << "Cropping voxel cube to convex hull" << std::endl;

	pcl::CropHull<pType> cropFilter;

	cropFilter.setInputCloud(voxelCloudFull);
	cropFilter.setHullCloud(hull);
	cropFilter.setHullIndices(polygons);
	cropFilter.setDim(dimension);

	//Crop it
	cropFilter.filter(*voxelCloud);

	std::cout << voxelCloud->size() << " points remaining after crop" << std::endl;
*/
/*
	//Crop to Concave hull
	std::cout << "Cropping voxel cube to concanve hull" << std::endl;

	cropFilter.setInputCloud(voxelCloudCx);
	cropFilter.setHullCloud(hull2);
	cropFilter.setHullIndices(polygons2);
	cropFilter.setCropOutside(0); //Remove points on the inside of the concave hull
	//Crop It
	cropFilter.filter(*voxelCloud);

	std::cout << voxelCloud->size() << " points remaining after crop" << std:: endl;
	//Test write
	writer.write<pType>("voxelCloud.ply",*voxelCloud);
*/
/*	RayCaster rayC(voxelCloud,voxelSize,boundingBox);

	std::string testPath = "testRayVoxels.ply";

	//rayC.writeVoxelCenters(testPath);

	//rayC.getVoxelCenters(voxelCloud);
	//Create object to store voxel data
	//std::vector<std::vector<cv::Vec3f> > voxelColors(voxelCloud->size(), std::vector<cv::Vec3f>(imgFileList.size()));
	std::vector<voxelInfo> voxelDesc(voxelCloud->size());
	std::vector<int> toWrite(voxelCloud->size(),0);

	//Initalize the voxelDesc
	for(int i = 0; i < voxelDesc.size(); i++){
		//New structure to fill
		voxelInfo newVInfo;

		//Initalize the vector
		newVInfo.colors.resize(imgFileList.size());

		//Init the values (-1 means undefined)
		newVInfo.AvgColor = cv::Vec3b(0,0,0);
		newVInfo.colorVariance = -1;
		newVInfo.nearestDist = -1;
		newVInfo.energy = -1;
	//	newVInfo.isOccluder = false;
		
		//Store voxel info
		voxelDesc[i] = newVInfo;
	}
*/
/* --------------- RAY CASTING ------------------------ */
/*
	//Read in each image and raycast area of interest
	for(int i = 0; i < imgFileList.size(); i++){

		std::cout << "Processing: " << imgFileList[i] << std::endl;
		//Read in image (image list and cameras are in same order)
		cv::Mat image = cv::imread(imgFileList[i]);
		image.convertTo(image,CV_32FC3,1./255);

		//Convert to HLS (bring 0-255 to 0-1)
		cv::cvtColor(image,image,CV_BGR2HLS);
		
		
		Camera cam = cameras[i];
			
		//Create binary mask for area of interest
		cv::Mat mask;
		createMask(cam,vInfo,image.size(),mask);

		//TEST WRITE OUT MASK
		std::stringstream ss;
		ss << "masks/image_" << i <<".jpg";
		cv::imwrite(ss.str(), mask);
		//Find ever nonzero pixel in the mask
		cv::Mat nonZeroCoords;
		cv::findNonZero(mask,nonZeroCoords);

		//Iterate over each nonzero point
		for(int j = 0; j < nonZeroCoords.total(); j++){

			//Get image point
			cv::Point imgPt = nonZeroCoords.at<cv::Point>(j); 

			//Get image ray
			std::vector<int> intersections;
			cv::Point3f origin, direction;
			cam.getRay(imgPt,origin,direction);

			//Cast ray
			rayC.castRay(origin,direction,intersections);
*/
/*			//Check to see if ray hit a known voxel;
			bool intersectedKnown = false;
			for(int k = 0; k < intersections.size(); k++){
				if(voxelize.isVoxelOccupied(voxelCloud->points[intersections[k]])){
					intersectedKnown = true;
				}
			}
*/			//If we hit a known voxel, then no need to store colors, just change flag
/*			if(intersectedKnown){
				for(int k = 0; k < intersections.size(); k++){
					voxelDesc[intersections[k]].isOccluder = true;
				}			
			//If we did not, then we store colors
			}else{
*/			//Store colors
/*				for(int k = 0; k < intersections.size(); k++){
					//toWrite[intersections[k]] = 1;
					voxelDesc[intersections[k]].colors[i] = image.at<cv::Vec3f>(imgPt);
					//voxelColors[intersections[k]][i] = image.at<cv::Vec3f>(imgPt);	
				}
			
		}
		
	}
*/
/*
pcl::PLYWriter writer2;
pcl::PointCloud<pcl::PointXYZRGB> outCloud;
pcl::PointXYZRGB tmpPt;
for(int i = 0; i < toWrite.size(); i++){

	if(toWrite[i] && voxelDesc[i].isOccluder){
		tmpPt.x =  voxelCloud->points[i].x;
		tmpPt.y =  voxelCloud->points[i].y;
		tmpPt.z =  voxelCloud->points[i].z;

		tmpPt.r = 150;
		tmpPt.g = 150;
		tmpPt.b = 150;
		outCloud.push_back(tmpPt);
	}

}

writer2.write<pcl::PointXYZRGB>("intersectedVoxels.ply",outCloud);
writer2.write<pType>("voxelCube.ply", *voxelCloud);*/
/* --------------- CALCULATE VOXEL STATISTICS ------------------------ */
/*
std::vector<double> variances(voxelDesc.size(),-1.0);
std::vector<cv::Vec3b> AvgColors(voxelDesc.size());

std::cout << "Calculating Color Statisitcs" << std::endl;
for(int i = 0; i < voxelDesc.size(); i++){

pcl::PointXYZRGB tmpPt;
	cv::Vec3f AvgColor;
	cv::Vec3f colorVariance;
	cv::Vec3b BGRcolor;
	int numIntersections = 0;

	//Only check non-occuluder voxels
   // if(!voxelDesc[i].isOccluder){
	if(getColorStatistics(voxelDesc[i].colors, AvgColor, colorVariance, numIntersections)){

		//Make sure it has at least 1 intersections
		if(numIntersections > 1){

			//Calculate total variance only using H and S
			double totalV = std::sqrt( colorVariance.val[0]*colorVariance.val[0] +
						   colorVariance.val[2]*colorVariance.val[2] );

			//Convert to rgb
			hls2bgr(AvgColor,BGRcolor);
			
			//Store values
			voxelDesc[i].colorVariance = totalV;
			voxelDesc[i].AvgColor = BGRcolor;
*/			/*	tmpPt.x =  voxelCloud->points[i].x;
				tmpPt.y =  voxelCloud->points[i].y;
				tmpPt.z =  voxelCloud->points[i].z;

				//Convert to rgb
				hls2bgr(AvgColor,BGRcolor);

				//Fill in
				tmpPt.b = BGRcolor.val[0];
				tmpPt.g = BGRcolor.val[1];
				tmpPt.r = BGRcolor.val[2];
	
				AvgColors[i] = BGRcolor;
				outCloud.push_back(tmpPt);*/
		/*	}
		}
	}*/

     //}

//}


/* --------------- FILL VOXEL DISTANCE AND ENERGY INFO ------------------------ */
/*
std::ofstream outEn;
outEn.open("energies.txt");

std::vector<float> dists(voxelDesc.size());

std::cout << "Calculating Voxel Energy" << std::endl;
for(int i = 0; i < voxelDesc.size(); i++){

 //Only process non-occulders
 //if(!voxelDesc[i].isOccluder){
	//Get point location
	pType pt;

	pt.x = voxelCloud->points[i].x;
	pt.y = voxelCloud->points[i].y;
	pt.z = voxelCloud->points[i].z;

	//Find closest known point
	int ind;
	float dist;
	knownAndEstVox.findApproxNearest(pt, ind, dist);

	//Store distance
	voxelDesc[i].nearestDist = dist;

	dists[i] = dist;

	//Calculate enegery
	calculateEnergy(voxelDesc[i]);

	//Modulate energy by energy of nearest voxel
	voxelDesc[i].energy *= knownInfo[ind].energy;
  //}
	//TESTING write out
	outEn << voxelDesc[i].energy << std::endl;

}
outEn.close();
*/


/* --------------- FILTER BY ENERGY ------------------------ */
/*

for(int t = 1; t < 11; t++){
stringstream ss;
double thresh = ((double)t)/10.0 - 0.05;

ss << "energyClouds/energy_" << thresh << ".ply";

pcl::PointCloud<pcl::PointXYZRGBNormal> outCloud2;
for(int i = 0; i < voxelDesc.size(); i++){

	//if( voxelDesc[i].energy != -1 && voxelDesc[i].energy > 0.95){
	if( voxelDesc[i].energy != -1 && voxelDesc[i].energy > thresh){

		pcl::PointXYZRGBNormal pointToAdd;

		if( voxelDesc[i].AvgColor.val[0] || voxelDesc[i].AvgColor.val[1] || voxelDesc[i].AvgColor.val[2]){

			pointToAdd.x =  voxelCloud->points[i].x;
			pointToAdd.y =  voxelCloud->points[i].y;
			pointToAdd.z =  voxelCloud->points[i].z;
	
			pointToAdd.b = voxelDesc[i].AvgColor.val[0];
			pointToAdd.g = voxelDesc[i].AvgColor.val[1];
			pointToAdd.r = voxelDesc[i].AvgColor.val[2];

			outCloud2.push_back(pointToAdd);
		}
	}



}

writer.write<pType>(ss.str(),outCloud2);

}

*/
/* ------------------ TESTING AREA ------------------------- */
/*
	std::ofstream out;
	out.open("vars.txt");

	for(int i = 0; i < variances.size(); i++){
		out << variances[i] << std::endl;
	}

	out.close();

	out.open("dists.txt");
	for(int i = 0; i < variances.size(); i++){
		out << dists[i] << std::endl;
	}
	out.close();

	writer.write<pType>("radiusCleaned.ply", *cloudFiltered);
	//writer.write<pType>("voxelCube.ply", *voxelCloud);
	//writer.write<pcl::PointXYZRGB>("varianceFiltered.ply",outCloud);
	//writer.write<pcl::PointXYZRGBA>("energyFiltered.ply",outCloud2);
*/

/* ------------------ TRIANGULATE ------------------------- */
/*std::cout << "Triangulating Point Cloud" << std::endl;
	//Create search tree
	pcl::search::KdTree<pType>::Ptr tree2(new pcl::search::KdTree<pType>);
	tree2->setInputCloud(knownAndEstVoxels);


	//Init objects
	//pcl::GreedyProjectionTriangulation<pType> gp3;
	pcl::MarchingCubesRBF<pType> mc;
	pcl::PolygonMesh triangles;


	//Set parameters
	mc.setGridResolution(2*voxelSize,2*voxelSize,2*voxelSize);
	mc.setIsoLevel(0.5);
	mc.setSearchMethod(tree2);
	mc.setInputCloud(knownAndEstVoxels);

	//Reconstruct
	mc.reconstruct(triangles);
*//*
	//Set the max distance between connected points
	gp3.setSearchRadius(5*voxelSize);

	//Set other parameters
	gp3.setMu(2.5);
	gp3.setMaximumNearestNeighbors(26);
	gp3.setMaximumSurfaceAngle(M_PI/4);
	gp3.setMinimumAngle(0);
	gp3.setMaximumAngle(2*M_PI/3);
	gp3.setNormalConsistency(false);

	//Get results
	gp3.setInputCloud(knownAndEstVoxels);
	gp3.setSearchMethod(tree2);
	gp3.reconstruct(triangles);
*/
	
//	std::cout << triangles.polygons.size() << std::endl;
	//Save results
//	pcl::io::savePolygonFilePLY("mesh.ply",triangles);	

/* ------------------ SEGMENT ------------------------- */

	std::cout << "Segmenting" << std::endl;

//	pcl::IndicesClustersPtr clusters(new pcl::IndicesClusters), small_clusters(new pcl::IndicesClusters), large_clusters(new pcl::IndicesClusters);
/*	pcl::ConditionalEuclideanClustering<pType> cec(true);

	cec.setInputCloud(knownAndEstVoxels);
	cec.setConditionFunction(&customRegionGrowing);
	cec.setClusterTolerance(voxelSize);
	cec.setMinClusterSize( knownAndEstVoxels->points.size() / 3000);
	cec.setMaxClusterSize( knownAndEstVoxels->points.size() / 2);
	cec.segment(*clusters);
	cec.getRemovedClusters(small_clusters,large_clusters);

	//Change colors for visualzation
	
	//Color too small RED
	for(int i = 0; i < small_clusters->size(); ++i){
		for(int j = 0; j < (*small_clusters)[i].indices.size(); ++j){
			knownAndEstVoxels->points[(*small_clusters)[i].indices[j]].r = 255;
			knownAndEstVoxels->points[(*small_clusters)[i].indices[j]].g = 0;
			knownAndEstVoxels->points[(*small_clusters)[i].indices[j]].b = 0;
		}
	}

	//Color too big BLUE
	for(int i = 0; i < large_clusters->size(); ++i){
		for(int j = 0; j < (*large_clusters)[i].indices.size(); ++j){
			knownAndEstVoxels->points[(*large_clusters)[i].indices[j]].r = 0;
			knownAndEstVoxels->points[(*large_clusters)[i].indices[j]].g = 0;
			knownAndEstVoxels->points[(*large_clusters)[i].indices[j]].b = 255;
		}
	}

	//Random color for eveything else
	for(int i = 0; i < clusters->size();i++){
		int red = std::rand() % 255;
		int green = std::rand() % 255;
		int blue = std::rand() % 255;

		for(int j = 0; j < (*clusters)[i].indices.size(); ++j){
			knownAndEstVoxels->points[(*clusters)[i].indices[j]].r = red;
			knownAndEstVoxels->points[(*clusters)[i].indices[j]].r = green;
			knownAndEstVoxels->points[(*clusters)[i].indices[j]].r = blue;
		}
	}

*/

	pcl::search::Search<pType>::Ptr tree = boost::shared_ptr<pcl::search::Search<pType> > (new pcl::search::KdTree<pType>);

	//Copy normals
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::copyPointCloud(*knownAndEstVoxels,*normals);

	pcl::IndicesPtr indices(new std::vector<int>);
	pcl::PassThrough<pType> pass;
	pass.setInputCloud(knownAndEstVoxels);
	pass.setFilterFieldName("z");
	pass.setFilterLimits(vInfo.minZ,vInfo.maxZ);
	pass.filter(*indices);

	//Grow region
	std::vector<pcl::PointIndices> clusters;
	pcl::RegionGrowingRGB<pType> reg;

	reg.setInputCloud(knownAndEstVoxels);
	reg.setInputNormals(normals);
	reg.setIndices(indices);
	reg.setSearchMethod(tree);
	reg.setNormalTestFlag(true);
	reg.setCurvatureTestFlag(true);
	reg.setSmoothModeFlag(true);
	
	reg.setDistanceThreshold(2*voxelSize);
	reg.setPointColorThreshold(15);
	reg.setRegionColorThreshold(5);
	reg.setMinClusterSize(knownAndEstVoxels->points.size()*0.001);
	reg.setMaxClusterSize(knownAndEstVoxels->points.size()*0.5);
	
	reg.setSmoothnessThreshold(12.0 / 180 * M_PI);
	reg.setCurvatureThreshold(1.0);
	

	reg.extract(clusters);

	//Save inital clusters
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloudInit = reg.getColoredCloud() ;
	writer.write<pcl::PointXYZRGB>("segmentedVoxelsInit.ply",*colored_cloudInit);

	std::vector<cv::Vec2f> hueAndSatMeans(clusters.size());
	//Get Mean of clusters
	for(int i = 0; i < clusters.size(); i++){
		//Definitions
		std::vector<cv::Vec3f> colors(clusters[i].indices.size());
		for(int j = 0; j < clusters[i].indices.size(); j++){
			//Convert color to HSL
			cv::Vec3b BGR;
			cv::Vec3f HLS;

			BGR.val[0] = knownAndEstVoxels->points[clusters[i].indices[j]].b;
			BGR.val[1] = knownAndEstVoxels->points[clusters[i].indices[j]].g;
			BGR.val[2] = knownAndEstVoxels->points[clusters[i].indices[j]].r;
		
			bgr2hls(BGR,HLS);
			
			HLS.val[0] *= 360;
			colors[j] = HLS;
		}

		//Get color stats
		cv::Vec3f AvgColor, variance;
		int numI = 0;
		getColorStatistics(colors, AvgColor, variance, numI);

		//Store hue and saturmation average
		hueAndSatMeans[i] = cv::Vec2f(AvgColor.val[0],AvgColor.val[2]);
	}

	//Cluster based on num of materials
	cv::Mat colorMeans(clusters.size(),3,CV_32F), labels, centers;

	//Fill in means
	for(int i = 0; i < hueAndSatMeans.size(); i++){
		colorMeans.at<float>(i,1) = hueAndSatMeans[i].val[0];
		colorMeans.at<float>(i,2) = hueAndSatMeans[i].val[1];
		colorMeans.at<float>(i,3) = 0.0;
	}	

	//Cluster
	double compact;
	std::vector<double> compacts;
/*
	ofstream writeC;
	writeC.open("objectiveFunction.txt");
	
      for(int i = 2; i < 80; i++){
	cv::Mat cMeans,lbls,cnts;
	colorMeans.copyTo(cMeans);
	compact = cv::kmeans(cMeans, i, lbls, cv::TermCriteria(cv::TermCriteria::COUNT,100,1), 1, cv::KMEANS_RANDOM_CENTERS, cnts);

	writeC << i << " " << compact << std::endl;
      }
	writeC.close();
*/
	compact = cv::kmeans(colorMeans, numMat, labels, cv::TermCriteria(cv::TermCriteria::COUNT,100,1), 1, cv::KMEANS_RANDOM_CENTERS, centers);
	//Create new point clusters
	std::vector<pcl::PointIndices> clustersK(numMat);

	for(int i = 0; i < labels.total(); i++){
		int label = labels.at<int>(i,0);

		//Merge labels	
		clustersK[label].indices.insert( clustersK[label].indices.end(),
						 clusters[i].indices.begin(),
						 clusters[i].indices.end() );

	}

	//Create colored cloud

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>) ;
		//fill cloud
	for(int i = 0; i < knownAndEstVoxels->points.size(); i++){
		pcl::PointXYZRGB point;
		point.x = knownAndEstVoxels->points[i].x;
		point.y = knownAndEstVoxels->points[i].y;
		point.z = knownAndEstVoxels->points[i].z;

		//Red is default no cluster
		point.r = 255;	
		point.g = 0;	
		point.b = 0;	

		colored_cloud->push_back(point);
	}

	//Make colormap
	std::vector<unsigned char> colors;
	for(int i = 0; i < clustersK.size(); i++){
		colors.push_back(static_cast<unsigned char> (rand() % 255));
		colors.push_back(static_cast<unsigned char> (rand() % 255));
		colors.push_back(static_cast<unsigned char> (rand() % 255));
	}

	//Fill colors
	int next_color = 0;
	int numK = 0;
	for(int i = 0; i < clustersK.size(); i++){
		std::cout<< clustersK[i].indices.size() << std::endl;
		for(int j = 0; j < clustersK[i].indices.size(); j++){
			colored_cloud->points[clustersK[i].indices[j]].r = colors[3 * next_color]; 
			colored_cloud->points[clustersK[i].indices[j]].g = colors[3 * next_color + 1]; 
			colored_cloud->points[clustersK[i].indices[j]].b = colors[3 * next_color + 2]; 
		}
		next_color++;
	}

	std::cout<< compact << std::endl;

	writer.write<pcl::PointXYZRGB>("segmentedVoxels.ply",*colored_cloud);


/* ------------------ TRIANGULATE ------------------------- */
/*
std::cout << "Triangulating Point Cloud" << std::endl;
	//Create search tree
	pcl::search::KdTree<pType>::Ptr tree2(new pcl::search::KdTree<pType>);
	tree2->setInputCloud(knownAndEstVoxels);


	//Init objects
	pcl::PolygonMesh triangles;
	//pcl::GreedyProjectionTriangulation<pType> gp3;
	pcl::MarchingCubesRBF<pType> mc;


	//Set parameters
	//mc.setGridResolution(2*voxelSize,2*voxelSize,2*voxelSize);
	mc.setLeafSize(voxelSize);
	mc.setIsoLevel(0.5);
//	mc.setPaddingSize(3);
//	mc.setNearestNeighborNum(100);
//	mc.setMaxBinarySearchLevel(10);

	mc.setSearchMethod(tree2);
	mc.setInputCloud(knownAndEstVoxels);

	//Reconstruct
	mc.reconstruct(triangles);

 //Set the max distance between connected points
	gp3.setSearchRadius(5*voxelSize);

	//Set other parameters
	gp3.setMu(2.5);
	gp3.setMaximumNearestNeighbors(26);
	gp3.setMaximumSurfaceAngle(M_PI/4);
	gp3.setMinimumAngle(0);
	gp3.setMaximumAngle(2*M_PI/3);
	gp3.setNormalConsistency(false);

	//Get results
	gp3.setInputCloud(knownAndEstVoxels);
	gp3.setSearchMethod(tree2);
	gp3.reconstruct(triangles);


//	std::cout << triangles.polygons.size() << std::endl;
	//Save results
	pcl::io::savePolygonFilePLY("mesh.ply",triangles);	


	return EXIT_SUCCESS;
*/
	//std::string dcPath = "dualContouringInput";
	//writeDualContouring(dcPath, vInfo.minZ, knownAndEstVoxels);
}				/* ----------  end of function main  ---------- */
