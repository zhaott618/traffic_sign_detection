#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <math.h>
#include<iostream>
#include <string.h>
using namespace std;
using namespace cv;
#define SHAPE 3 //要检测的多边形边数shape 检测形状 3为三角形，4矩形，5为五边形……
CvSeq* contours;//边缘
CvRect rect;
int c1;
//////////////////////////////////////////////////////////////////
//函数功能：用向量来做COSα=两向量之积/两向量模的乘积求两条线段夹角
//输入：   线段3个点坐标pt1,pt2,pt0,最后一个参数为公共点
//输出：   线段夹角，单位为角度
//////////////////////////////////////////////////////////////////
double angle(CvPoint* pt1, CvPoint* pt2, CvPoint* pt0)
{
	double dx1 = pt1->x - pt0->x;
	double dy1 = pt1->y - pt0->y;
	double dx2 = pt2->x - pt0->x;
	double dy2 = pt2->y - pt0->y;
	double angle_line = (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);//余弦值
	return acos(angle_line) * 180 / 3.141592653;
}
//////////////////////////////////////////////////////////////////
//函数功能：采用多边形逼近检测，通过约束条件寻找多边形
//输入：   img 原图像
//          storage 存储
//          minarea，maxarea 检测多边形的最小/最大面积
//          minangle,maxangle 检测多边形边夹角范围，单位为角度  
//输出：   多边形序列
//////////////////////////////////////////////////////////////////
CvSeq* findSquares4(IplImage* img, IplImage* img0, CvMemStorage* storage, int minarea, int minangle, int maxangle)
{
	int c=0;
	CvSize sz = cvSize(img->width & -2, img->height & -2);
	IplImage* timg = cvCloneImage(img);//拷贝一次img
	CvSeq* result;
	double s, t;
	CvSeq* squares = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvPoint), storage);
	//cvCanny(timg,timg,50,200,3);
	//cvShowImage("ss", timg);
	//cvSetImageROI(timg, cvRect(0, 0, sz.width, sz.height));
	cvFindContours(timg, storage, &contours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
	//cout << contours << endl;
	while (contours)
	{ //多边形逼近   
		result = cvApproxPoly(contours, sizeof(CvContour), storage, CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0);
		//如果是凸多边形并且面积在范围内
		if (result->total == SHAPE && fabs(cvContourArea(result, CV_WHOLE_SEQ)) > minarea  &&  cvCheckContourConvexity(result))
		{
			c++;
			s = 0;
			//判断每一条边
			for (int i = 0; i < SHAPE + 1; i++)
			{
				if (i >= 2)
				{   //角度            
					t = fabs(angle((CvPoint*)cvGetSeqElem(result, i), (CvPoint*)cvGetSeqElem(result, i - 2), (CvPoint*)cvGetSeqElem(result, i - 1)));
					s = s > t ? s : t;
				}
			}
			//这里的S为直角判定条件 单位为角度
			if (s > minangle && s < maxangle)
			for (int i = 0; i < SHAPE; i++)
				cvSeqPush(squares, (CvPoint*)cvGetSeqElem(result, i));


			vector<Point>points;
			for (int i = 0; i < contours->total; i++)
			{
				CvPoint p;
				CvPoint*p1 = (CvPoint*)cvGetSeqElem(contours, i);
				p.x = p1->x;
				p.y = p1->y;
				points.push_back(p);
			}
			rect = boundingRect(points);
			CvPoint pt1, pt2;
			pt1.x = rect.x;
			pt1.y = rect.y;
			pt2.x = rect.x + rect.width;
			pt2.y = rect.y + rect.height;
			cvRectangle(img0, pt1, pt2, Scalar(0, 0, 0), 3);
		}
		contours = contours->h_next;
	}
	c1 = c;
	cvReleaseImage(&timg);
	return squares;
}


//////////////////////////////////////////////////////////////////
//函数功能：画出所有矩形
//输入：   img 原图像
//          squares 多边形序列
//          wndname 窗口名称
//输出：   图像中标记多边形
//////////////////////////////////////////////////////////////////
void drawSquares(IplImage* img, IplImage* img0,IplImage*img1, CvSeq* squares, const char* wndname)
{
	IplImage*imageROI = cvCloneImage(img0);
	CvSeqReader reader;
	IplImage* cpy = cvCloneImage(img);
	CvPoint pt[SHAPE];
	int i;
	cvStartReadSeq(squares, &reader, 0);
	for (i = 0; i < squares->total; i += SHAPE)
	{
		CvPoint* rect = pt;
		int count = SHAPE;
		for (int j = 0; j < count; j++)
		{
			memcpy(pt + j, reader.ptr, squares->elem_size);
			CV_NEXT_SEQ_ELEM(squares->elem_size, reader);
		}
		cvPolyLine(cpy, &rect, &count, 1, 1, Scalar(0, 0, 0), 3, CV_AA, 0);//彩色绘制
		cvPolyLine(img1, &rect, &count, 1, 1, Scalar(0, 0, 0), 3, CV_AA, 0);//彩色绘制
	}
	if (c1)
	{

	//	cvShowImage(wndname, cpy);
		cvSetImageROI(imageROI, cvRect(rect.x, rect.y, rect.width, rect.height));
	//	cvShowImage("ROI", imageROI);
		cvSaveImage("C://Users//yuanze//Desktop//test1.jpg", imageROI);
		cout << "前方障碍物" << endl;
	}
	else
		cout << "未找到交通标志" << endl;
	cvReleaseImage(&imageROI);
	cvReleaseImage(&cpy);
}