#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>
using namespace std;
using namespace cv;
int findCircles(Mat src, Mat midImage,Mat &src11, Mat src1,Mat &src1m, Mat &imageroi, double dThreshold1, double dThreshold2, int iSize, int iMinArea)
{
	int count = 0;
	Mat imageROI, src0;
	Canny(src, src0, dThreshold1, dThreshold2, iSize);
	//找轮廓  
	vector<vector<Point>> q_vPointContours;
	findContours(src0, q_vPointContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
	//drawContours(src0, q_vPointContours, -1, Scalar(0,255,0));//显示轮廓  
	//筛选目标轮廓点  
	vector<Point> vfindContours(0);
	size_t q_iAmountContours = q_vPointContours.size();
	size_t iIndex = 0;
	for (iIndex = 0; iIndex < q_iAmountContours; iIndex++)
	{
		//根据圆的面积判断是否为目标圆  
		double ddarea = contourArea(q_vPointContours[iIndex]);
		if (iMinArea >ddarea)
			continue;
		else
			count++;
		//cout << ddarea << endl;
		//存储目标圆的轮廓点  
		size_t findCount = q_vPointContours[iIndex].size();
		for (int i = 0; i < findCount; i++)
			vfindContours.push_back(q_vPointContours[iIndex][i]);

		//采用椭圆拟合来得到圆  
		RotatedRect rectElli = fitEllipse(vfindContours);
		float fR = MIN(rectElli.size.width, rectElli.size.height);// 是否为圆，可以比较这两个值，若十分接近或相等，就是一个正圆  
		//cout << "fitEllipse 中心: " << rectElli.center.x << ", " << rectElli.center.y << "  半径:" << fR / 2 << endl;
		circle(midImage, Point(rectElli.center), fR / 2, Scalar(0, 0, 0), 2);//圆周  
		circle(midImage, Point(rectElli.center), 5, Scalar(0, 0, 0), 3);//圆心  
		vector<Rect>boundRect(q_vPointContours.size());
		int x0 = 0, y0 = 0, w0 = 0, h0 = 0;
		boundRect[iIndex] = boundingRect((Mat)q_vPointContours[iIndex]);
		x0 = boundRect[iIndex].x;  //获得第i个外接矩形的左上角的x坐标
		y0 = boundRect[iIndex].y; //获得第i个外接矩形的左上角的y坐标
		w0 = boundRect[iIndex].width; //获得第i个外接矩形的宽度
		h0 = boundRect[iIndex].height; //获得第i个外接矩形的高度
		rectangle(midImage, Point(x0, y0), Point(x0 + w0, y0 + h0), Scalar(0, 255, 0), 2, 8); //绘制第i个外接矩形
		rectangle(src1, Point(x0, y0), Point(x0 + w0, y0 + h0), Scalar(0, 255, 0), 2, 8); //绘制第i个外接矩形
		rectangle(src11, Point(x0, y0), Point(x0 + w0, y0 + h0), Scalar(0, 255, 0), 2, 8); //绘制第i个外接矩形
		//cout << w0 << endl;
		//cout << w0 << endl;
		//cout << h0 << endl;
		imageroi = src(Rect(x0, y0, w0, h0));
		//imageroi = imageROI;
	//	imshow("ROI", imageroi);
		src1m= src1(Rect(x0, y0, w0, h0));
	}
	return count;
}