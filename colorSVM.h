#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <fstream>
#include "stdlib.h"
#include"colormoment.h"
//声明命名空间
using namespace std;
using namespace cv;
//using namespace cv::ml;
#define s 10000
//!训练数据参数
const int sample_num_perclass = 20;     //训练每类图片数量
const int class_num = 3;                //训练类数
//!所有图片尺寸归一化
const int image_cols = 1000;              //定义图片尺寸
const int image_rows = 1000;              //定义图片尺寸
//!生成的训练文件保存位置
char SVMName[40] = "C://judgecolor//SVM.xml";              //分类器的训练生成的名字,读取时也按照这个名字来
#define RW 0//0为读取现有的分类器,1表示重新训练一个分类器
Size size = Size(image_cols, image_rows);
int colorsvm(Mat src)
{
#if RW  
	//!读取训练数据
	Mat trainingData = Mat::zeros(sample_num_perclass*class_num, 9, CV_32FC1);          //填入图像的7个Hu矩

	Mat trainingLabel = Mat::zeros(sample_num_perclass*class_num, 1, CV_32SC1);
	vector<float>descriptors;//结果数组 
	char buf[50];                       //字符缓冲区
	for (int i = 0; i<class_num; i++)        //不同了类的循环
	{
		for (int j = 0; j<sample_num_perclass; j++)      //一个类中的图片数量
		{
			//!生成图片的路径(不同类的图片被放在了不同的文件夹下)
			sprintf(buf, "C://judgecolor//%d//%d.jpg", i, j + 1);
			//!读取
			Mat src = imread(buf, 1);
			//!重设尺寸（归一化）
			Mat reImg;
			resize(src, reImg, size, CV_INTER_CUBIC);

			//!求Hu矩
			float *descriptors;
			float *dstPoi = trainingData.ptr<float>(i*sample_num_perclass + j);  //指向源的指针
			descriptors = colorMom(reImg);
			for (int r = 0; r < 9; r++)
				dstPoi[r] = descriptors[r];
			//!添加对该数据的分类标签
			int *labPoi = trainingLabel.ptr<int>(i*sample_num_perclass + j);
			labPoi[0] = i;
		}
	}
	imwrite("res.png", trainingData);
	//cout << descriptors.size() << endl;
	//!创建SVM支持向量机并训练数据
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setC(0.01);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	svm->train(trainingData, ROW_SAMPLE, trainingLabel);
	svm->save(SVMName);
#else
	//读取xml文件
	//Ptr<SVM> svm = SVM::load<SVM>(SVMName);
	CvSVM svm;
	svm.load(SVMName);
#endif
	//!读取一副图片进行测试
	Mat temp;
	temp = src;
	Mat dst;
	int Class;
	resize(temp, dst, size, CV_INTER_CUBIC);

	Mat pre(1, 9, CV_32FC1);
	float *descriptor;
	float *p = pre.ptr<float>(0);
	descriptor = colorMom(temp);
	for (int i = 0; i < 9; i++)
		p[i] = descriptor[i];
	int res = svm->predict(pre);
	//cout << res << endl;
	switch (res)
	{
	case 0:Class = 1; break;
	case 1:Class = 2; break;
	case 2:Class = 3; break;
	}
	return Class;
}