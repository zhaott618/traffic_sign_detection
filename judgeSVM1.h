//#-*-coding=utf-8
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <fstream>
#include "stdlib.h"
//声明命名空间
using namespace std;
using namespace cv;
using namespace cv::ml;
#define s 10000
//!训练数据参数
const int sample_num_perclass1 = 33;     //训练每类图片数量
const int class_num1 = 3;                //训练类数
//!所有图片尺寸归一化
const int image_cols1 = 800;              //定义图片尺寸
const int image_rows1 = 800;              //定义图片尺寸
//!生成的训练文件保存位置
char SVMName1[40] = "C://judge1//SVM1.xml";              //分类器的训练生成的名字,读取时也按照这个名字来
#define RW1 0//0为读取现有的分类器,1表示重新训练一个分类器
//!程序入口
Size size1 = Size(image_cols1, image_rows1);
void judgesvm1(Mat &src, int classify)
{
if RW1   
	//!读取训练数据
	Mat trainingData = Mat::zeros(sample_num_perclass1*class_num1, s, CV_32FC1);          //填入图像的7个Hu矩

	Mat trainingLabel = Mat::zeros(sample_num_perclass1*class_num1, 1, CV_32SC1);
	vector<float>descriptors;//结果数组 
	char buf[50];                       //字符缓冲区
	for (int i = 0; i<class_num1; i++)        //不同了类的循环
	{
		for (int j = 0; j<sample_num_perclass1; j++)      //一个类中的图片数量
		{
			//!生成图片的路径(不同类的图片被放在了不同的文件夹下)
			sprintf(buf, "C://judge%d//%d//%d.jpg", classify, i, j + 1);
			//!读取
			Mat src = imread(buf, -1);
			//!重设尺寸（归一化）
			Mat reImg;
			resize(src, reImg, size1, CV_INTER_CUBIC);
			Mat canny;
			Canny(reImg, canny, 50, 200, 3);
			//!求Hu矩
			HOGDescriptor *hog = new HOGDescriptor(Size(image_cols1, image_rows1), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);  //具体意思见参考文章1,2             
			hog->compute(reImg, descriptors, Size(1, 1), Size(0, 0));
			//!将Hu矩填入训练数据集里
			float *dstPoi = trainingData.ptr<float>(i*sample_num_perclass1 + j);  //指向源的指针
			for (int r = 0; r<s; r++)
				dstPoi[r] = (float)descriptors[r];
			//!添加对该数据的分类标签
			int *labPoi = trainingLabel.ptr<int>(i*sample_num_perclass1 + j);
			labPoi[0] = i;
		}
	}
	imwrite("res.png", trainingData);
	//cout << descriptors.size() << endl;
	//!创建SVM支持向量机并训练数据
	Ptr<SVM> svm1 = SVM::create();
	svm1->setType(SVM::C_SVC);
	svm1->setC(0.01);
	svm1->setKernel(SVM::LINEAR);
	svm1->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	svm1->train(trainingData, ROW_SAMPLE, trainingLabel);
	svm1->save(SVMName1);
#else
	//读取xml文件
	Ptr<SVM> svm1 = StatModel::load<SVM>(SVMName1);
#endif
	//!读取一副图片进行测试
	Mat temp1 = imread("C://Users//yuanze//Desktop//40test3.jpg");
	//cvtColor(temp1, temp1, CV_RGB2GRAY);
	//cout << temp1 << endl;
	//cout << temp.channels() << endl;
	/*Mat temp;
	temp = src;*/
	//cout << temp << endl;

	Mat dst;
	resize(temp1, dst, size1, CV_INTER_CUBIC);
	HOGDescriptor *hog = new HOGDescriptor(Size(image_cols1, image_rows1), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);  //具体意思见参考文章1,2       
	vector<float>descriptor;//结果数组       
	hog->compute(dst, descriptor, Size(1, 1), Size(0, 0));
	Mat pre(1, s, CV_32FC1);
	float *p = pre.ptr<float>(0);
	for (int i = 0; i<s; i++)
		p[i] = descriptor[i];

	int res = svm1->predict(pre);
	//cout << res << endl;
	switch (res)
	{
	case 0:cout << "右转" << endl; break;
	case 1:cout << "左转" << endl; break;
	case 2:cout << "直行" << endl; break;
	}
}