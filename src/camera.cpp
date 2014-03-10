/*
 * =====================================================================================
 *
 *       Filename:  camera.cpp
 *
 *    Description:  Class for camera
 *
 *        Version:  1.0
 *        Created:  01/25/2014 05:00:58 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  David Nilosek (), drn2369@cis.rit.edu
 *        Company:  Rochester Institute of Technology
 *
 * =====================================================================================
 */

#include "camera.h"

Camera::Camera(cv::Mat proj){

	//Set projection matrix
	setProjectionMatrix(proj);

	//Calculate and store origin
	m_origin = this->calcOrigin();
}

cv::Mat Camera::calcOrigin(){

	//Define
	double x,y,z;

	//Get origin of camera from projection matrix's right null space

	cv::SVD svd(m_proj, cv::SVD::FULL_UV);

	//Right null vector is the last col of V
	//Convert to inhomo
	
	x = svd.vt.at<double>(3,0)/svd.vt.at<double>(3,3);
	y = svd.vt.at<double>(3,1)/svd.vt.at<double>(3,3);
	z = svd.vt.at<double>(3,2)/svd.vt.at<double>(3,3);

	cv::Mat pos = (cv::Mat_<double>(3,1) << x,y,z);

	return pos;
}

void Camera::world2image(cv::Point3f worldPt, cv::Point2f &imgPt){

	//Make world point homogenious
	cv::Mat worldPtH = (cv::Mat_<double>(4,1) << worldPt.x, worldPt.y, worldPt.z, 1.0);

	//Project
	cv::Mat imgPtH = m_proj*worldPtH;

	//Calculate inhomogenious image point
	imgPt.x = (float)(imgPtH.at<double>(0,0)/imgPtH.at<double>(2,0));
	imgPt.y = (float)(imgPtH.at<double>(1,0)/imgPtH.at<double>(2,0));
}

//From page 162 in hartley
void Camera::getRay(cv::Point imgPt, cv::Point3f &origin, cv::Point3f &direction){

	cv::Mat M = m_proj(cv::Rect(0,0,3,3));
	cv::Mat p4 = m_proj.col(3);

	cv::Mat imPtM = (cv::Mat_<double>(3,1) << imgPt.x , imgPt.y, 1);

	cv::Mat oriM = M.inv()*p4;

	origin.x = -1*oriM.at<double>(0,0);
	origin.y = -1*oriM.at<double>(1,0);
	origin.z = -1*oriM.at<double>(2,0);

	cv::Mat dirM = M.inv()*imPtM;

	double normDir = cv::norm(dirM);

	direction.x = dirM.at<double>(0,0) / normDir;
	direction.y = dirM.at<double>(1,0) / normDir;
	direction.z = dirM.at<double>(2,0) / normDir;
}

//Assumes pointing down Z and focal length is in pixels
double Camera::calculateGSD(){

	//Decompose projection matrix
	cv::Mat cameraMat, rotMat, transV;
	cv::decomposeProjectionMatrix(m_proj,cameraMat,rotMat,transV);

 	//get focal length
	double f = cameraMat.at<double>(0,0);

	//get Z 
	transV = this->calcOrigin();
	double z = transV.at<double>(2,0);

	//calculate gsd (assuming focal length is in pixels)
	// gsd = z/f; 
	double gsd = std::fabs(z/f); //fabs incase Z is negative 

	return gsd;
}
