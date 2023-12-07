#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class PhaseCongruency;
using namespace cv;
void PrintMatType(const cv::Mat& mat);//�ж�����ͼ������ͣ������������̨
cv::Mat ifftshift(const cv::Mat& mat);  //����ת�ƣ���0Ƶ�ʷ���ת�������Ͻǣ����ϸ���Ҷ�任��ķֲ�
cv::Mat ConvertFourierImageBack(cv::Mat& complexImage);//�渵��Ҷ�任������һ������ͼ��
cv::Mat combineMagnitudeAndPhase(const cv::Mat& magnitude, const cv::Mat& phase);   //���Ƶ�����Լ���λ�ף����й�һ��
cv::Mat ensureEvenDimensions(cv::Mat inputImage);   //�������ͼ��֤����������Ϊż������������ת��
cv::Mat lowpassfilter(const cv::Mat& sze, double cutoff, int n);
/*����һ����ͨ�˲�szeΪ�����ͼ��
cutoff���˲����Ľ�ֹƵ�ʣ���Χ��0��0.5��
n���˲����Ľ�����nԽ�ߣ�����Խ���͡���n�����Ǵ��ڵ���1����������
ע�⣬n���ӱ���ȷ��������һ��ż����
*/
void createLogGaborFilters(const cv::Mat& radius, const cv::Mat& lp,std::vector<cv::Mat>& logGabor,int nscale=4, double minWaveLength=3, double mult=2.1, double sigmaOnf=0.55);
/*����LogGabor�����˲��������ͨ�˲���ϡ�
radiusΪ������ͼ��lpΪ��ͨ�˲���
logGaborΪ��ž����˲������顣
nscale 4 - С���߶ȵ�����������ʹ�� 3-6 ��ֵ��
minWaveLength 3 - ��С�߶��˲����Ĳ�����
mult 2.1 - �����˲���֮����������ӡ�
sigmaOnf 0.55 - ��˹�������� Gabor �˲�����Ƶ��Ĵ��ݺ����ı�׼�����˲�������Ƶ�ʵı��ʡ�
*/

cv::Mat Atan2(const cv::Mat& y, const cv::Mat& x, bool isabs);//�������ÿ��λ���϶�Ӧ��arctan(y/x)��isabs��ʾ�Ƿ�Խ��ȡ����ֵΪ1��|arctan(y/x)|

void constructAngularFilterSpreadFunction(const cv::Mat& sintheta, const cv::Mat& costheta, cv::Mat& spread,int norient=4, int o=1);
//LogGabor�Ƕ��˲���sintheta,costhetaΪ������ϵ�½Ƕ�sinֵ��cosֵ,norientΪ���м�������oΪ��ǰ����,spreadΪ��Ž���ĽǶ��˲���

float rayleighmode(const cv::Mat& data, int nbins = 50);//��������ģʽ���

float median(const cv::Mat& sumAn_ThisOrient);//������λ��
void convertAngles(cv::Mat& orMat);// ���Ƕ� -pi..0 ת��Ϊ 0..pi,������ 0 �� 180 �ȱ�ʾ
