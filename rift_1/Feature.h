#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class PhaseCongruency; 
using namespace cv;
using namespace std;



void FeatureDetection(const cv::Mat& m, int npt, std::vector<cv::Point2f>& kpts);//fast������⣬mΪ���ͼ��nptΪ��ȡ��ǰnpt�������������������������kpts��

cv::Mat CreateMIM(const PhaseCongruency& obj);//����������ͼMIM

cv::Mat convertDescriptors(const std::vector<std::vector<float>>& des);// ��������ת��Ϊ cv::Mat ����

// �� cv::Point2f ת��Ϊ cv::KeyPoint
std::vector<cv::KeyPoint> convertToKeyPoints(const std::vector<cv::Point2f>& points);

cv::Mat extractPatch(const cv::Mat& img, const cv::Point2f& center, int r);

// ���㲢��������������
std::vector<float> computeDescriptor(const cv::Mat& patch, int no=6, int nbin=6);

void FeatureDescribe(std::vector<cv::Point2f>& kpts, const cv::Mat& MIM, std::vector<vector<float>>& des,int no=6, int nbin=6, int patch_size=96);  //����������ͼ���ֳ�no x no��С���򣬶�ÿ��С�������ֱ��ͼ