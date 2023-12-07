#include"Utils.h"
#include"PhaseCongruency.h"


void PrintMatType(const cv::Mat& mat) {
    int type = mat.type();
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');

    std::cout << "Mat Type: " << r << std::endl;
}

cv::Mat ifftshift(const cv::Mat& mat) {


    cv::Mat cpy;
    mat.copyTo(cpy);


    cv::Mat ret = cpy(cv::Rect(0, 0, cpy.cols & -2, cpy.rows & -2));


    int cx = ret.cols / 2;
    int cy = ret.rows / 2;
    cv::Mat q0(ret, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(ret, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(ret, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(ret, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    cv::Mat tmp; // swap quadrants (Bottom-Right with Top-Left)
    q3.copyTo(tmp);
    q0.copyTo(q3);
    tmp.copyTo(q0);
    q2.copyTo(tmp); // swap quadrant (Bottom-Left with Top-Right)
    q1.copyTo(q2);
    tmp.copyTo(q1);

    return ret;
}

Mat ConvertFourierImageBack(cv::Mat& complexImage) {
    // 逆傅里叶变换
    cv::Mat inverseTransform;
    //cv::dft(complexImage, inverseTransform, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    cv::dft(complexImage, inverseTransform, cv::DFT_INVERSE | cv::DFT_COMPLEX_OUTPUT | cv::DFT_SCALE);
    // 归一化结果
    //cv::normalize(inverseTransform, inverseTransform, 0, 1, cv::NORM_MINMAX);

    // 输出图像
    cv::Mat outputImage;
    outputImage = inverseTransform;
    return outputImage;
}

cv::Mat combineMagnitudeAndPhase(const cv::Mat& magnitude, const cv::Mat& phase) {
    cv::Mat complexImage(magnitude.size(), CV_32FC2);
    for (int y = 0; y < magnitude.rows; ++y) {
        for (int x = 0; x < magnitude.cols; ++x) {
            float mag = magnitude.at<float>(y, x);
            float pha = phase.at<float>(y, x);
            //std::cout << mag << "+" << pha << std::endl;
            complexImage.at<cv::Vec2f>(y, x) = cv::Vec2f(mag * cos(pha), mag * sin(pha));
        }
    }
   // std::cout << "com" << complexImage << std::endl;

    cv::Mat reconstructed;
    reconstructed = ConvertFourierImageBack(complexImage);
    return reconstructed;
}

cv::Mat ensureEvenDimensions(cv::Mat inputImage) {
    int rows = inputImage.rows;
    int cols = inputImage.cols;

    // Check if the number of rows is odd
    if (rows % 2 != 0) {
        inputImage = inputImage(cv::Rect(0, 0, cols, rows - 1));
    }

    // Check if the number of columns is odd
    if (cols % 2 != 0) {
        inputImage = inputImage(cv::Rect(0, 0, cols - 1, inputImage.rows));
    }

    return inputImage;
}

cv::Mat lowpassfilter(const cv::Mat& sze, double cutoff, int n) {
    // 检查 cutoff 的有效性
    if (cutoff < 0 || cutoff > 0.5) {
        throw std::runtime_error("cutoff frequency must be between 0 and 0.5");
    }

    // 检查 n 的有效性
    if (n < 1) {
        throw std::runtime_error("n must be an integer >= 1");
    }

    int rows = sze.rows;
    int cols = sze.cols;

    // 创建范围向量
    cv::Mat xRange = cv::Mat::zeros(1, cols, CV_32F);
    cv::Mat yRange = cv::Mat::zeros(rows, 1, CV_32F);

    // 初始化 xRange 和 yRange
    for (int i = 0; i < cols; ++i) {
        xRange.at<float>(0, i) = (i - cols / 2) / float(cols);
    }
    for (int i = 0; i < rows; ++i) {
        yRange.at<float>(i, 0) = (i - rows / 2) / float(rows);
    }

    // 创建坐标网格
    cv::Mat x, y;
    cv::repeat(xRange, rows, 1, x);
    cv::repeat(yRange, 1, cols, y);
    //cv::transpose(y, y);
    //std::cout << x << std::endl;
    /*std::cout << y << std::endl;*/
    // 计算半径矩阵
    cv::Mat radius;
    cv::sqrt(x.mul(x) + y.mul(y), radius);
    cv::Mat result;
    // 计算滤波器
    cv::pow(radius / cutoff, 2 * n, result);
    cv::Mat f = 1.0 / (1.0 + result);
    f = ifftshift(f);
    
    return f;
}
void createLogGaborFilters(const cv::Mat& radius, const cv::Mat& lp, std::vector<cv::Mat>& logGabor, int nscale, double minWaveLength , double mult, double sigmaOnf) {
    // 初始化 logGabor 滤波器数组
    logGabor.resize(nscale);

    for (int s = 0; s < nscale; ++s) {
        // 计算波长和中心频率
        float wavelength =(float) minWaveLength * std::pow(mult, s);
        float fo = 1.0 /(float) wavelength;  // 滤波器的中心频率

        // 创建对数高斯滤波器
        logGabor[s] = cv::Mat(radius.size(), CV_32F);
        for (int i = 0; i < radius.rows; ++i) {
            for (int j = 0; j < radius.cols; ++j) {
                
                float r = radius.at<float>(i, j) / fo;
                logGabor[s].at<float>(i, j) = std::exp(-std::pow(std::log(r), 2) / (2 * std::pow(std::log(sigmaOnf), 2)));
            }
        }

        // 应用低通滤波器

        logGabor[s] = logGabor[s].mul(lp);

        // 设置滤波器中 0 频率点的值为 0
        logGabor[s].at<float>(0, 0) = 0;
    }
}

cv::Mat Atan2(const cv::Mat& y, const cv::Mat& x, bool isabs) {
    cv::Mat result = cv::Mat(y.size(), CV_32F);
    for (int i = 0; i < y.rows; ++i) {
        for (int j = 0; j < y.cols; ++j) {
            if (isabs)
                result.at<float>(i, j) = abs(std::atan2(y.at<float>(i, j), x.at<float>(i, j)));
            else
                result.at<float>(i, j) = std::atan2(y.at<float>(i, j), x.at<float>(i, j));
        }
    }
    return result;
}


void constructAngularFilterSpreadFunction(const cv::Mat& sintheta, const cv::Mat& costheta, cv::Mat& spread,int norient, int o) {
    // 滤波器角度
    //double angl = (o - 1) * CV_PI / norient;
    double angl = o * CV_PI / norient;

    // 计算角度差异
    cv::Mat ds = sintheta * std::cos(angl) - costheta * std::sin(angl);  // 正弦差
    cv::Mat dc = costheta * std::cos(angl) + sintheta * std::sin(angl);  // 余弦差

    // 初始化 dtheta 和 magnitude 矩阵
    cv::Mat dtheta = cv::Mat(ds.size(), CV_32F);
    cv::Mat magnitude = cv::Mat(ds.size(), CV_32F);  // 用于极径的临时变量
    dtheta = Atan2(ds, dc, 1);
    // 缩放 theta 使余弦扩散函数具有正确的波长并限制在 pi 内
    dtheta = cv::min(dtheta * norient / 2, CV_PI);
    // 计算 cos(dtheta) 并使值范围在 0-1 之间
    spread = cv::Mat(dtheta.size(), dtheta.type());
    dtheta.forEach<float>([&spread](float& val, const int* position) -> void {
        spread.at<float>(position[0], position[1]) = (std::cos(val) + 1) / 2;
        });
}
float rayleighmode(const cv::Mat& data, int nbins) {
    CV_Assert(!data.empty()); // 确保 data 不为空
    CV_Assert(data.channels() == 1); // 确保 data 是单通道的

    double minVal, maxVal;
    cv::minMaxLoc(data, &minVal, &maxVal);

    // 确保 minVal 和 maxVal 不相等
    CV_Assert(minVal <= maxVal);

    // 如果 minVal 等于 maxVal，进行微调以创建有效的直方图边界
    if (minVal == maxVal) {
        minVal -= 0.00001;  // 减少 minVal
        maxVal += 0.00001;  // 增加 maxVal
    }

    // 设置直方图边界和尺寸
    float range[] = { static_cast<float>(minVal), static_cast<float>(maxVal) };
    const float* histRange = { range };
    cv::Mat hist;
    cv::calcHist(&data, 1, 0, cv::Mat(), hist, 1, &nbins, &histRange, true, false);

    // 找到直方图中的最大值和位置
    double maxFreq;
    cv::Point maxInd;
    cv::minMaxLoc(hist, nullptr, &maxFreq, nullptr, &maxInd);

    // 计算瑞利模式
    float binWidth = (maxVal - minVal) / nbins;
    float rmode = minVal + (maxInd.x + 0.5f) * binWidth;
    return rmode;
}

float median(const cv::Mat& sumAn_ThisOrient) {
    // 将 sumAn_ThisOrient 转换为一维向量
    std::vector<float> array;
    if (sumAn_ThisOrient.isContinuous()) {
        array.assign((float*)sumAn_ThisOrient.datastart, (float*)sumAn_ThisOrient.dataend);
    }
    else {
        for (int i = 0; i < sumAn_ThisOrient.rows; ++i) {
            array.insert(array.end(), sumAn_ThisOrient.ptr<float>(i), sumAn_ThisOrient.ptr<float>(i) + sumAn_ThisOrient.cols);
        }
    }

    // 计算中位数
    std::nth_element(array.begin(), array.begin() + array.size() / 2, array.end());
    float median = array[array.size() / 2];
    return median;
}
void convertAngles(cv::Mat& orMat) {
    for (int i = 0; i < orMat.rows; ++i) {
        for (int j = 0; j < orMat.cols; ++j) {
            // 检查并调整角度值
            float& angle = orMat.at<float>(i, j);
            if (angle < 0) {
                angle += CV_PI;
            }
            // 将角度从弧度转换为度
            angle = angle * 180.0 / CV_PI;
        }
    }
}

