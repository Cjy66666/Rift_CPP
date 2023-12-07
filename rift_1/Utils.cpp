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
    // �渵��Ҷ�任
    cv::Mat inverseTransform;
    //cv::dft(complexImage, inverseTransform, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    cv::dft(complexImage, inverseTransform, cv::DFT_INVERSE | cv::DFT_COMPLEX_OUTPUT | cv::DFT_SCALE);
    // ��һ�����
    //cv::normalize(inverseTransform, inverseTransform, 0, 1, cv::NORM_MINMAX);

    // ���ͼ��
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
    // ��� cutoff ����Ч��
    if (cutoff < 0 || cutoff > 0.5) {
        throw std::runtime_error("cutoff frequency must be between 0 and 0.5");
    }

    // ��� n ����Ч��
    if (n < 1) {
        throw std::runtime_error("n must be an integer >= 1");
    }

    int rows = sze.rows;
    int cols = sze.cols;

    // ������Χ����
    cv::Mat xRange = cv::Mat::zeros(1, cols, CV_32F);
    cv::Mat yRange = cv::Mat::zeros(rows, 1, CV_32F);

    // ��ʼ�� xRange �� yRange
    for (int i = 0; i < cols; ++i) {
        xRange.at<float>(0, i) = (i - cols / 2) / float(cols);
    }
    for (int i = 0; i < rows; ++i) {
        yRange.at<float>(i, 0) = (i - rows / 2) / float(rows);
    }

    // ������������
    cv::Mat x, y;
    cv::repeat(xRange, rows, 1, x);
    cv::repeat(yRange, 1, cols, y);
    //cv::transpose(y, y);
    //std::cout << x << std::endl;
    /*std::cout << y << std::endl;*/
    // ����뾶����
    cv::Mat radius;
    cv::sqrt(x.mul(x) + y.mul(y), radius);
    cv::Mat result;
    // �����˲���
    cv::pow(radius / cutoff, 2 * n, result);
    cv::Mat f = 1.0 / (1.0 + result);
    f = ifftshift(f);
    
    return f;
}
void createLogGaborFilters(const cv::Mat& radius, const cv::Mat& lp, std::vector<cv::Mat>& logGabor, int nscale, double minWaveLength , double mult, double sigmaOnf) {
    // ��ʼ�� logGabor �˲�������
    logGabor.resize(nscale);

    for (int s = 0; s < nscale; ++s) {
        // ���㲨��������Ƶ��
        float wavelength =(float) minWaveLength * std::pow(mult, s);
        float fo = 1.0 /(float) wavelength;  // �˲���������Ƶ��

        // ����������˹�˲���
        logGabor[s] = cv::Mat(radius.size(), CV_32F);
        for (int i = 0; i < radius.rows; ++i) {
            for (int j = 0; j < radius.cols; ++j) {
                
                float r = radius.at<float>(i, j) / fo;
                logGabor[s].at<float>(i, j) = std::exp(-std::pow(std::log(r), 2) / (2 * std::pow(std::log(sigmaOnf), 2)));
            }
        }

        // Ӧ�õ�ͨ�˲���

        logGabor[s] = logGabor[s].mul(lp);

        // �����˲����� 0 Ƶ�ʵ��ֵΪ 0
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
    // �˲����Ƕ�
    //double angl = (o - 1) * CV_PI / norient;
    double angl = o * CV_PI / norient;

    // ����ǶȲ���
    cv::Mat ds = sintheta * std::cos(angl) - costheta * std::sin(angl);  // ���Ҳ�
    cv::Mat dc = costheta * std::cos(angl) + sintheta * std::sin(angl);  // ���Ҳ�

    // ��ʼ�� dtheta �� magnitude ����
    cv::Mat dtheta = cv::Mat(ds.size(), CV_32F);
    cv::Mat magnitude = cv::Mat(ds.size(), CV_32F);  // ���ڼ�������ʱ����
    dtheta = Atan2(ds, dc, 1);
    // ���� theta ʹ������ɢ����������ȷ�Ĳ����������� pi ��
    dtheta = cv::min(dtheta * norient / 2, CV_PI);
    // ���� cos(dtheta) ��ʹֵ��Χ�� 0-1 ֮��
    spread = cv::Mat(dtheta.size(), dtheta.type());
    dtheta.forEach<float>([&spread](float& val, const int* position) -> void {
        spread.at<float>(position[0], position[1]) = (std::cos(val) + 1) / 2;
        });
}
float rayleighmode(const cv::Mat& data, int nbins) {
    CV_Assert(!data.empty()); // ȷ�� data ��Ϊ��
    CV_Assert(data.channels() == 1); // ȷ�� data �ǵ�ͨ����

    double minVal, maxVal;
    cv::minMaxLoc(data, &minVal, &maxVal);

    // ȷ�� minVal �� maxVal �����
    CV_Assert(minVal <= maxVal);

    // ��� minVal ���� maxVal������΢���Դ�����Ч��ֱ��ͼ�߽�
    if (minVal == maxVal) {
        minVal -= 0.00001;  // ���� minVal
        maxVal += 0.00001;  // ���� maxVal
    }

    // ����ֱ��ͼ�߽�ͳߴ�
    float range[] = { static_cast<float>(minVal), static_cast<float>(maxVal) };
    const float* histRange = { range };
    cv::Mat hist;
    cv::calcHist(&data, 1, 0, cv::Mat(), hist, 1, &nbins, &histRange, true, false);

    // �ҵ�ֱ��ͼ�е����ֵ��λ��
    double maxFreq;
    cv::Point maxInd;
    cv::minMaxLoc(hist, nullptr, &maxFreq, nullptr, &maxInd);

    // ��������ģʽ
    float binWidth = (maxVal - minVal) / nbins;
    float rmode = minVal + (maxInd.x + 0.5f) * binWidth;
    return rmode;
}

float median(const cv::Mat& sumAn_ThisOrient) {
    // �� sumAn_ThisOrient ת��Ϊһά����
    std::vector<float> array;
    if (sumAn_ThisOrient.isContinuous()) {
        array.assign((float*)sumAn_ThisOrient.datastart, (float*)sumAn_ThisOrient.dataend);
    }
    else {
        for (int i = 0; i < sumAn_ThisOrient.rows; ++i) {
            array.insert(array.end(), sumAn_ThisOrient.ptr<float>(i), sumAn_ThisOrient.ptr<float>(i) + sumAn_ThisOrient.cols);
        }
    }

    // ������λ��
    std::nth_element(array.begin(), array.begin() + array.size() / 2, array.end());
    float median = array[array.size() / 2];
    return median;
}
void convertAngles(cv::Mat& orMat) {
    for (int i = 0; i < orMat.rows; ++i) {
        for (int j = 0; j < orMat.cols; ++j) {
            // ��鲢�����Ƕ�ֵ
            float& angle = orMat.at<float>(i, j);
            if (angle < 0) {
                angle += CV_PI;
            }
            // ���Ƕȴӻ���ת��Ϊ��
            angle = angle * 180.0 / CV_PI;
        }
    }
}

