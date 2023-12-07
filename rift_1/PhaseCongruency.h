#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include"Utils.h"
using namespace cv;
using namespace std;

class PhaseCongruency {
public:
    PhaseCongruency(Mat& img, int noiseMethod=-1, int nscale=4, double minWaveLength=3, double mult=2.1, double sigmaOnf=0.55, double k=2.0, int norient=6, float cutOff=0.45, int g=10);//构造函数
    void applySinCos(const cv::Mat& src, cv::Mat& sinMat, cv::Mat& cosMat);//计算矩阵中的每个值的sin,cos
    void prepareImageForFFT();//计算图像的最佳傅里叶变换尺寸，加速计算。将数据存放至paddedImage
    void FourierTransform();//傅里叶变换
    void calculatePolarMatrices();//初始化radius，theta，sintheta，costheta
    void phasecong3();
    

    cv::Mat image;  // 输入图像
    cv::Mat paddedImage;//图像的最佳傅里叶变换尺寸
    cv::Mat imageFFT; // 图像的FFT结果
    cv::Mat frequencySpectrum;  //频谱
    cv::Mat phaseSpectrum;      //相位谱
    cv::Mat radius;     //极坐标半径
    cv::Mat theta;      //极坐标角度
    cv::Mat sintheta;   //极坐标角度sin值
    cv::Mat costheta;   //极坐标角度cos值
    cv::Mat M;  //最大矩
    cv::Mat m;  //最小矩
    cv::Mat OR; //每个像素点的特征方向
    cv::Mat featType;   //相位特征
    cv::Mat pcSum;  //相位和
    vector<vector<cv::Mat>> EO; //滤波器响应，包含了图像在不同尺度和方向上经过滤波处理后的响应的复数数组
    vector<cv::Mat> PC; //相位一致性模型
    float T;    //噪声阈值


    int noiseMethod;
    int nscale;
    double minWaveLength;
    double mult;
    double sigmaOnf;
    double k;
    int norient;
    float cutOff;
    int g ;
    const double epsilon = 1e-6; // 一个小常数以避免除以零

};



