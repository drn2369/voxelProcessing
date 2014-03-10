/*
 * =====================================================================================
 *
 *       Filename:  camera.h
 *
 *    Description:  Interface for camera class
 *
 *        Version:  1.0
 *        Created:  01/25/2014 04:57:49 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  David Nilosek (), drn2369@cis.rit.edu
 *        Company:  Rochester Institute of Technology
 *
 * =====================================================================================
 */
#ifndef CAMERA_H
#define CAMERA_H


#include <cv.h>

/*
 * =====================================================================================
 *        Class:  Camera
 *  Description:  Class for camera
 * =====================================================================================
 */
class Camera
{
	public:
		/* ====================  LIFECYCLE     ======================================= */
		Camera ();                             /* constructor */
		Camera (cv::Mat projMat);
		/* ====================  ACCESSORS     ======================================= */
		cv::Mat getProjectionMatrix(){return m_proj;}
		cv::Mat getOrigin(){return m_origin;}
		/* ====================  MUTATORS      ======================================= */
		void setProjectionMatrix(cv::Mat p){ m_proj = p;}
		/* ====================  OPERATORS     ======================================= */
		void world2image(cv::Point3f worldPt, cv::Point2f &imgPt);
		void getRay(cv::Point imgPt, cv::Point3f &origin, cv::Point3f &direction);
		double calculateGSD();
		/* ====================  DATA MEMBERS  ======================================= */
	protected:
		cv::Mat calcOrigin();
	private:
		cv::Mat m_proj;
		cv::Mat m_origin;
}; /* -----  end of class Camera  ----- */

#endif
