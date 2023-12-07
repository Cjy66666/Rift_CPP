#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include"Utils.h"
using namespace cv;
using namespace std;

class PhaseCongruency {
public:
    PhaseCongruency(Mat& img, int noiseMethod=-1, int nscale=4, double minWaveLength=3, double mult=2.1, double sigmaOnf=0.55, double k=2.0, int norient=6, float cutOff=0.45, int g=10);//���캯��
    void applySinCos(const cv::Mat& src, cv::Mat& sinMat, cv::Mat& cosMat);//��������е�ÿ��ֵ��sin,cos
    void prepareImageForFFT();//����ͼ�����Ѹ���Ҷ�任�ߴ磬���ټ��㡣�����ݴ����paddedImage
    void FourierTransform();//����Ҷ�任
    void calculatePolarMatrices();//��ʼ��radius��theta��sintheta��costheta
    void phasecong3();
    

    cv::Mat image;  // ����ͼ��
    cv::Mat paddedImage;//ͼ�����Ѹ���Ҷ�任�ߴ�
    cv::Mat imageFFT; // ͼ���FFT���
    cv::Mat frequencySpectrum;  //Ƶ��
    cv::Mat phaseSpectrum;      //��λ��
    cv::Mat radius;     //������뾶
    cv::Mat theta;      //������Ƕ�
    cv::Mat sintheta;   //������Ƕ�sinֵ
    cv::Mat costheta;   //������Ƕ�cosֵ
    cv::Mat M;  //����
    cv::Mat m;  //��С��
    cv::Mat OR; //ÿ�����ص����������
    cv::Mat featType;   //��λ����
    cv::Mat pcSum;  //��λ��
    vector<vector<cv::Mat>> EO; //�˲�����Ӧ��������ͼ���ڲ�ͬ�߶Ⱥͷ����Ͼ����˲���������Ӧ�ĸ�������
    vector<cv::Mat> PC; //��λһ����ģ��
    float T;    //������ֵ


    int noiseMethod;
    int nscale;
    double minWaveLength;
    double mult;
    double sigmaOnf;
    double k;
    int norient;
    float cutOff;
    int g ;
    const double epsilon = 1e-6; // һ��С�����Ա��������

};



