#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class PhaseCongruency; 
using namespace cv;
using namespace std;



void FeatureDetection(const cv::Mat& m, int npt, std::vector<cv::Point2f>& kpts);//fast特征检测，m为检测图像，npt为提取的前npt个特征，将特征点的坐标存放至kpts中

cv::Mat CreateMIM(const PhaseCongruency& obj);//获得最大索引图MIM

cv::Mat convertDescriptors(const std::vector<std::vector<float>>& des);// 将描述符转换为 cv::Mat 类型

// 将 cv::Point2f 转换为 cv::KeyPoint
std::vector<cv::KeyPoint> convertToKeyPoints(const std::vector<cv::Point2f>& points);

cv::Mat extractPatch(const cv::Mat& img, const cv::Point2f& center, int r);

// 计算并返回特征描述符
std::vector<float> computeDescriptor(const cv::Mat& patch, int no=6, int nbin=6);

void FeatureDescribe(std::vector<cv::Point2f>& kpts, const cv::Mat& MIM, std::vector<vector<float>>& des,int no=6, int nbin=6, int patch_size=96);  //特征描述将图像块分成no x no的小区域，对每个小区域计算直方图