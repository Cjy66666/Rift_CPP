#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class PhaseCongruency;
using namespace cv;
void PrintMatType(const cv::Mat& mat);//判断输入图像的类型，并输出至控制台
cv::Mat ifftshift(const cv::Mat& mat);  //坐标转移，将0频率分量转移至左上角，符合傅里叶变换后的分布
cv::Mat ConvertFourierImageBack(cv::Mat& complexImage);//逆傅里叶变换，并归一化返回图像
cv::Mat combineMagnitudeAndPhase(const cv::Mat& magnitude, const cv::Mat& phase);   //结合频率谱以及相位谱，进行归一化
cv::Mat ensureEvenDimensions(cv::Mat inputImage);   //将输入的图像保证其行列数都为偶数，方便坐标转移
cv::Mat lowpassfilter(const cv::Mat& sze, double cutoff, int n);
/*创建一个低通滤波sze为输入的图像，
cutoff是滤波器的截止频率，范围是0到0.5。
n是滤波器的阶数，n越高，过渡越陡峭。（n必须是大于等于1的整数）。
注意，n被加倍以确保它总是一个偶数。
*/
void createLogGaborFilters(const cv::Mat& radius, const cv::Mat& lp,std::vector<cv::Mat>& logGabor,int nscale=4, double minWaveLength=3, double mult=2.1, double sigmaOnf=0.55);
/*创建LogGabor径向滤波，并与低通滤波结合。
radius为极坐标图，lp为低通滤波。
logGabor为存放径向滤波的数组。
nscale 4 - 小波尺度的数量，尝试使用 3-6 的值。
minWaveLength 3 - 最小尺度滤波器的波长。
mult 2.1 - 连续滤波器之间的缩放因子。
sigmaOnf 0.55 - 高斯描述对数 Gabor 滤波器在频域的传递函数的标准差与滤波器中心频率的比率。
*/

cv::Mat Atan2(const cv::Mat& y, const cv::Mat& x, bool isabs);//计算矩阵每个位置上对应的arctan(y/x)，isabs表示是否对结果取绝对值为1则|arctan(y/x)|

void constructAngularFilterSpreadFunction(const cv::Mat& sintheta, const cv::Mat& costheta, cv::Mat& spread,int norient=4, int o=1);
//LogGabor角度滤波，sintheta,costheta为极坐标系下角度sin值与cos值,norient为共有几个方向，o为当前方向,spread为存放结果的角度滤波。

float rayleighmode(const cv::Mat& data, int nbins = 50);//计算瑞丽模式误差

float median(const cv::Mat& sumAn_ThisOrient);//计算中位数
void convertAngles(cv::Mat& orMat);// 将角度 -pi..0 转换为 0..pi,方向以 0 到 180 度表示
