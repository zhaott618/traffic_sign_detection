#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>
using namespace std;
using namespace cv;
int findCircles1(Mat src, Mat midImage,Mat &src1, Mat &imageroi, double dThreshold1, double dThreshold2, int iSize, int iMinArea)
{
	int count = 0;
	Mat imageROI, src0;
	Canny(src, src0, dThreshold1, dThreshold2, iSize);
	//找轮廓  
	vector<vector<Point>> q_vPointContours1;
	findContours(src0, q_vPointContours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
	//drawContours(src0, q_vPointContours, -1, Scalar(0,255,0));//显示轮廓  
	//筛选目标轮廓点  
	vector<Point> vfindContours1;
	size_t q_iAmountContours1 = q_vPointContours1.size();
	size_t iIndex = 0;
	for (iIndex = 0; iIndex < q_iAmountContours1; iIndex++)
	{
		//根据圆的面积判断是否为目标圆  
		double ddarea1 = contourArea(q_vPointContours1[iIndex]);
		if (iMinArea >ddarea1)
			continue;
		else
			count++;
		//cout << ddarea1 << endl;
		//存储目标圆的轮廓点  
		size_t findCount1 = q_vPointContours1[iIndex].size();
		for (int i = 0; i < findCount1; i++)
			vfindContours1.push_back(q_vPointContours1[iIndex][i]);

		//采用椭圆拟合来得到圆  
		RotatedRect rectElli1 = fitEllipse(vfindContours1);
		float fR = MIN(rectElli1.size.width, rectElli1.size.height);// 是否为圆，可以比较这两个值，若十分接近或相等，就是一个正圆  
		//cout << "fitEllipse 中心: " << rectElli1.center.x << ", " << rectElli1.center.y << "  半径:" << fR / 2 << endl;
		circle(midImage, Point(rectElli1.center), fR / 2, Scalar(0, 0, 0), 2);//圆周  
		circle(midImage, Point(rectElli1.center), 5, Scalar(0, 0, 0), 3);//圆心  
		vector<Rect>boundRect1(q_vPointContours1.size());
		int x0 = 0, y0 = 0, w0 = 0, h0 = 0;
		boundRect1[iIndex] = boundingRect((Mat)q_vPointContours1[iIndex]);
		x0 = boundRect1[iIndex].x;  //获得第i个外接矩形的左上角的x坐标
		y0 = boundRect1[iIndex].y; //获得第i个外接矩形的左上角的y坐标
		w0 = boundRect1[iIndex].width; //获得第i个外接矩形的宽度
		h0 = boundRect1[iIndex].height; //获得第i个外接矩形的高度
		rectangle(midImage, Point(x0, y0), Point(x0 + w0, y0 + h0), Scalar(0, 255, 0), 2, 8); //绘制第i个外接矩形
		rectangle(src1, Point(x0, y0), Point(x0 + w0, y0 + h0), Scalar(0, 255, 0), 2, 8); //绘制第i个外接矩形

		//cout << w0 << endl;
		//cout << h0 << endl;
		imageroi= src(Rect(x0, y0, w0, h0));
		
	//	imshow("ROI", imageROI);
	}
	return count;
}