#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <fstream>
#include "stdlib.h"
//ÉùÃ÷ÃüÃû¿Õ¼ä
using namespace std;
using namespace cv;
using namespace cv::ml;
#define s 10000
//!ÑµÁ·Êý¾Ý²ÎÊý
const int sample_num_perclass2 = 33;     //ÑµÁ·Ã¿ÀàÍ¼Æ¬ÊýÁ¿
const int class_num2 = 5;                //ÑµÁ·ÀàÊý
//!ËùÓÐÍ¼Æ¬³ß´ç¹éÒ»»¯
const int image_cols2 = 800;              //¶¨ÒåÍ¼Æ¬³ß´ç
const int image_rows2 = 800;              //¶¨ÒåÍ¼Æ¬³ß´ç
//!Éú³ÉµÄÑµÁ·ÎÄ¼þ±£´æÎ»ÖÃ
char SVMName2[40] = "C://judge2//SVM2.xml";              //·ÖÀàÆ÷µÄÑµÁ·Éú³ÉµÄÃû×Ö,¶ÁÈ¡Ê±Ò²°´ÕÕÕâ¸öÃû×ÖÀ´
#define RW2 0//0Îª¶ÁÈ¡ÏÖÓÐµÄ·ÖÀàÆ÷,1±íÊ¾ÖØÐÂÑµÁ·Ò»¸ö·ÖÀàÆ÷
//!³ÌÐòÈë¿Ú
Size size2 = Size(image_cols2, image_rows2);
void judgesvm2(Mat &src, int classify)
{
#if RW2  
	//!¶ÁÈ¡ÑµÁ·Êý¾Ý
	Mat trainingData = Mat::zeros(sample_num_perclass2*class_num2, s, CV_32FC1);          //ÌîÈëÍ¼ÏñµÄ7¸öHu¾Ø

	Mat trainingLabel = Mat::zeros(sample_num_perclass2*class_num2, 1, CV_32SC1);
	vector<float>descriptors;//½á¹ûÊý×é 
	char buf[50];                       //×Ö·û»º³åÇø
	for (int i = 0; i<class_num2; i++)        //²»Í¬ÁËÀàµÄÑ­»·
	{
		for (int j = 0; j<sample_num_perclass2; j++)      //Ò»¸öÀàÖÐµÄÍ¼Æ¬ÊýÁ¿
		{
			//!Éú³ÉÍ¼Æ¬µÄÂ·¾¶(²»Í¬ÀàµÄÍ¼Æ¬±»·ÅÔÚÁË²»Í¬µÄÎÄ¼þ¼ÐÏÂ)
			sprintf(buf, "C://judge%d//%d//%d.jpg", classify, i, j + 1);
			//!¶ÁÈ¡
			Mat src = imread(buf, -1);
			//!ÖØÉè³ß´ç£¨¹éÒ»»¯£©
			Mat reImg;
			resize(src, reImg, size2, CV_INTER_CUBIC);
			Mat canny;
			Canny(reImg, canny, 50, 200, 3);
			//!ÇóHu¾Ø
			HOGDescriptor *hog = new HOGDescriptor(Size(image_cols2, image_rows2), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);  //¾ßÌåÒâË¼¼û²Î¿¼ÎÄÕÂ1,2             
			hog->compute(reImg, descriptors, Size(1, 1), Size(0, 0));
			//!½«Hu¾ØÌîÈëÑµÁ·Êý¾Ý¼¯Àï
			float *dstPoi = trainingData.ptr<float>(i*sample_num_perclass2 + j);  //Ö¸ÏòÔ´µÄÖ¸Õë
			for (int r = 0; r<s; r++)
				dstPoi[r] = (float)descriptors[r];
			//!Ìí¼Ó¶Ô¸ÃÊý¾ÝµÄ·ÖÀà±êÇ©
			int *labPoi = trainingLabel.ptr<int>(i*sample_num_perclass2 + j);
			labPoi[0] = i;
		}
	}
	imwrite("res.png", trainingData);
	//cout << descriptors.size() << endl;
	//!´´½¨SVMÖ§³ÖÏòÁ¿»ú²¢ÑµÁ·Êý¾Ý
	Ptr<SVM> svm2 = SVM::create();
	svm2->setType(SVM::C_SVC);
	svm2->setC(0.01);
	svm2->setKernel(SVM::LINEAR);
	svm2->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	svm2->train(trainingData, ROW_SAMPLE, trainingLabel);
	svm2->save(SVMName2);
#else
	//¶ÁÈ¡xmlÎÄ¼þ
	Ptr<SVM> svm2 = SVM::load<SVM>(SVMName2);
#endif
	//!¶ÁÈ¡Ò»¸±Í¼Æ¬½øÐÐ²âÊÔ
	Mat temp2 = imread("C://Users//yuanze//Desktop//40test4.jpg");
	//cout << temp1<< endl;
	//Mat temp;
	//temp = src.clone();

	//cout << temp << endl;
	/*if (temp.channels() == 3)
	cvtColor(temp, temp, CV_BGR2GRAY);*/

	//imshow("44", temp);
	Mat dst;
	resize(temp2, dst, size2, CV_INTER_CUBIC);
	HOGDescriptor *hog = new HOGDescriptor(Size(image_cols2, image_rows2), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);  //¾ßÌåÒâË¼¼û²Î¿¼ÎÄÕÂ1,2       
	vector<float>descriptor2;//½á¹ûÊý×é       
	hog->compute(dst, descriptor2, Size(1, 1), Size(0, 0));
	Mat pre(1, s, CV_32FC1);
	float *p = pre.ptr<float>(0);
	for (int i = 0; i<s; i++)
		p[i] = descriptor2[i];
	int res = svm2->predict(pre);
	//cout << res << endl;
	switch (res)
	{
	case 0:cout << "½ûÖ¹" << endl; break;
	case 1:cout << "ÏÞËÙ60" << endl; break;
	case 2:cout << "ÏÞËÙ40" << endl; break;
	case 3:cout << "ÏÞËÙ30" << endl; break;
	case 4:cout << "ÏÞËÙ20" << endl; break;
	}
}


/*hog特征维数计算
size_t HOGDescriptor::getDescriptorSize() const
{
CV_Assert(blockSize.width % cellSize.width == 0 &&
blockSize.height % cellSize.height == 0);
CV_Assert((winSize.width - blockSize.width) % blockStride.width == 0 &&
(winSize.height - blockSize.height) % blockStride.height == 0 );
return (size_t)nbins*
(blockSize.width/cellSize.width)*
(blockSize.height/cellSize.height)*
((winSize.width - blockSize.width)/blockStride.width + 1)*
((winSize.height - blockSize.height)/blockStride.height + 1);
}
*/