#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#define pi 3.1415926
using namespace cv;
double distance(Point pt1, Point pt2, Point center);
/** @function main */
int main(int argc, char** argv)
{
	Mat src, src_gray;
	Mat src_ROI;


	/// Read the image
	src = imread("F:\\12356.jpg", 1);

	if (!src.data)
	{
		return -1;
	}
	
	
//-----图像预处理----------------------------------------------------------------------
	/// Convert it to gray
	cvtColor(src, src_gray, CV_BGR2GRAY);
	cv::equalizeHist(src_gray, src_gray);
	/// Reduce the noise so we avoid false circle detection
	GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);


//------检测圆（正圆）-------------------------------------------------------------------
	std::vector<Vec3f> circles;

	/// Apply the Hough Transform to find the circles
	HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows / 8, 200, 100, 0, 0);

	/// Draw the circles detected
	/*for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		
		int radius = cvRound(circles[i][2]);
		// circle center
		circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// circle outline
		circle(src, center, radius, Scalar(0, 0, 255), 3, 8, 0);

	}*/
	
	
//-------筛选圆（半径大且整个圆在图内）-------------------------------------------------------------------------

	int pos = 0;
	int max = -1;
	for (size_t i = 0; i < circles.size(); i++)
		{
			Vec3f f = circles[i];
		    if (f[2]>max && f[0] + f[2]<src.rows && f[0] - f[2] >= 0 && f[1] + f[2]<src.cols && f[1] - f[2]>0)
			   {
			     max = f[2];
			     pos = i;
			    }
		 }
	 Point center(circles[pos][0], circles[pos][1]);//找到的圆心
	 int   radius = circles[pos][2];//找到的半径
	 
	//circle(src, center, radius, Scalar(255), 2);

	 ///----在图中标出表盘-----------------------------------------------------


	// circle center
	circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);
	// circle outline
	circle(src, center, radius, Scalar(0, 0, 255), 3, 8, 0);


//-------设置圆形ROI(设置完保存在src_ROI里)-----------------------------------------------------------------------------

	Mat mask = Mat::zeros(src.size(), CV_8UC1);
	std::vector<std::vector<Point>> contour;
	std::vector<Point> pts;
	for (int i = 0; i < 360; i++)
	{
		pts.push_back(Point(center.x+ radius*cos((i*pi/180.0)), center.y+ radius*sin(i*pi/180.0)));
	}
	contour.push_back(pts);
	drawContours(mask, contour, 0, Scalar::all(255), -1);
	src.copyTo(src_ROI, mask);


//-------检测指针--------------------------------------------------------------------------------------------------------------

	Mat src_ROI_Gray;
	
	// 将原图转换成灰度图
	cvtColor(src_ROI, src_ROI_Gray, CV_BGR2GRAY);

	// 使用3*3内核来进行降噪处理
	blur(src_ROI_Gray, src_ROI_Gray, Size(3, 3));
	//GaussianBlur(src_ROI_Gray, src_ROI_Gray, Size(9, 9), 2, 2);
	// Canny算子
	Canny(src_ROI_Gray, src_ROI_Gray, 3, 9, 3);
	
	std::vector<Vec4i> lines;

	HoughLinesP(src_ROI_Gray, lines, 1, CV_PI / 180, 50, 0.75*radius, 10);

	

	for (int i = 0; i < lines.size(); i++)
	{
		if(distance(Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]),center)<=0.05*radius)
		{
			line(src, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 1, 8, 0);
		}
	}



//-------显示最终图像----------------------------------------------------------------------
	/// Show your results

	namedWindow("Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE);
	imshow("Hough Circle Transform Demo", src);
	imshow("ROI", src_ROI);
	imshow("ROI_Gray", src_ROI_Gray);
	waitKey(0);
	return 0;

}

double distance(Point pt1,Point pt2,Point center) {
	double A, B, C, dis;
	// 化简两点式为一般式
	// 两点式公式为(y - y1)/(x - x1) = (y2 - y1)/ (x2 - x1)
	// 化简为一般式为(y2 - y1)x + (x1 - x2)y + (x2y1 - x1y2) = 0
	 // A = y2 - y1
	// B = x1 - x2
	// C = x2y1 - x1y2
	 A = pt2.y - pt1.y;
	B = pt1.x - pt2.x;
	C = pt2.x * pt1.y - pt1.x * pt2.y;
	//中心点坐标(coreX,coreY)
	double coreX, coreY;
	coreX = center.x;
	coreY = center.y;
	// 距离公式为d = |A*x0 + B*y0 + C|/√(A^2 + B^2)
	 dis = abs(A * coreX + B * coreY + C) / sqrt(A * A + B * B);
	 return dis;
}



















/*#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>

#include<iostream>


using namespace std;
using namespace cv;

int main()
{
	
	Mat image_original= imread("F:\\12356.jpg", 1);
	Mat image_gray;
	Mat image_binary;
	Mat image_equalize;
	Mat image_blur;
	Mat image_open;
	Mat image_canny;


	///----------------------------------------------------------------------------


	cvtColor(image_original, image_gray, CV_BGR2GRAY);//三通道的图转化为1通道的灰度图  
	 
	
	equalizeHist(image_gray, image_equalize);

	//medianBlur(image_equalize, image_blur, 5);


	//threshold(image_blur, image_binary, 145, 255, THRESH_BINARY);//通过阈值操作把灰度图变成二值图 
	
    ///GaussianBlur(image_equalize, image_blur,)

	threshold(image_equalize, image_binary, 145, 255, THRESH_BINARY);//通过阈值操作把灰度图变成二值图 
	
	//morphologyEx(image_binary, image_open, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(15, 15)));

	//const CvArr* _image_binary=(CvArr*)&image_binary;
	//CvArr* _image_canny = (CvArr*)&image_canny;
	cv::Canny(image_binary, image_canny, 50, 150, 3);
	cvtColor(image_canny, image_canny, CV_GRAY2BGR);
	///--------------------------------------------------------------------------------

	 
		std::vector<Vec3f> circles;//储存检测圆的容器 
	
	//参数为：待检测图像，检测结果，检测方法（这个参数唯一）,累加器的分辨率，两个圆间的距离，canny门限的上限（下限自动设为上限的一半），圆心所需要的最小的投票数，最大和最小半径  
		 
		HoughCircles(image_canny, circles, CV_HOUGH_GRADIENT, 1, 50, 100, 100, 100, 300);
	
		 //找出圆盘（因为最大的不一定是的，所以加了几个限制条件）
	 int pos = 0;
	int max = -1;
	 for (size_t i = 0; i < circles.size(); i++)
		 {
		     Vec3f f = circles[i];
		     if (f[2]>max && f[0] + f[2]<image_canny.rows && f[0] - f[2] >= 0 && f[1] + f[2]<image_canny.cols && f[1] - f[2]>0)
			     {
			         max = f[2];
			        pos = i;
			     }
		 }
	 Point center(circles[pos][0], circles[pos][1]);//找到的圆心
	 int   radius = circles[pos][2];//找到的半径
	 circle(image_canny, center, radius, Scalar(255), 2);


	///--------------------------------------------------------------------------------
	imshow("原始图像", image_original);
	waitKey(0);
	imshow("灰度图", image_gray);
	waitKey(0);
	imshow("直方图均衡化图像", image_equalize);
	waitKey(0);
	imshow("中值滤波图像", image_blur);
	waitKey(0);
	imshow("二值化图像", image_binary);
	waitKey(0);
	imshow("边缘检测", image_canny);
	waitKey(0);
	
	
	//imwrite("Gray.jpg", image);
	
	//waitKey(0);
	
	return 0;
}*/