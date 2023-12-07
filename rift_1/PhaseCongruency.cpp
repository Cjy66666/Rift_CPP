#include "PhaseCongruency.h"
#include <cmath>

PhaseCongruency::PhaseCongruency(Mat& img, int noiseMethod, int nscale, double minWaveLength, double mult, double sigmaOnf, double k, int norient, float cutOff, int g)
{
    if (img.channels() > 1) {
        cv::cvtColor(img, this->image, cv::COLOR_BGR2GRAY);
    }
    else
    {
        this->image = img.clone();
    }
    this->noiseMethod = noiseMethod;
    this->nscale = nscale;
    this->minWaveLength = minWaveLength;
    this->mult = mult;
    this->sigmaOnf = sigmaOnf;
    this->k = k;
    this->norient = norient;
    this->cutOff = cutOff;
    this->g = g;
    this->T = 0;
    vector<vector<cv::Mat>> EO_temp(nscale, vector<cv::Mat>(norient));
    this->EO = EO_temp;
    vector<cv::Mat> PC_temp(norient);
    this->PC = PC_temp;
    FourierTransform();
    this->pcSum = cv::Mat::zeros(image.size(), frequencySpectrum.type());
}

void PhaseCongruency::applySinCos(const cv::Mat& src, cv::Mat& sinMat, cv::Mat& cosMat)
{
    // 创建一个 1x2 的变换矩阵
    cv::Matx12d transformMat(1, 0);

    // 使用cv::transform应用正弦函数
    cv::transform(src, sinMat, transformMat);
    sinMat.forEach<float>([](float& val, const int* position) -> void {
        val = std::sin(val);
        });

    // 使用cv::transform应用余弦函数
    cv::transform(src, cosMat, transformMat);
    cosMat.forEach<float>([](float& val, const int* position) -> void {
        val = std::cos(val);
        });
}


void PhaseCongruency::prepareImageForFFT()     //提高效率可以不用
{
    int m = cv::getOptimalDFTSize(image.rows);
    int n = cv::getOptimalDFTSize(image.cols);
    cv::copyMakeBorder(image, paddedImage, 0, m - image.rows, 0, n - image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
}

void PhaseCongruency::FourierTransform()
{
    // 为傅里叶变换准备实部和虚部
    Mat planes[] = { Mat_<float>(image), Mat::zeros(image.size(), CV_32F) };
    Mat complexImage;
    merge(planes, 2, complexImage);

    // 进行傅里叶变换
    dft(complexImage, complexImage);
    imageFFT = complexImage;
    // 分离实部和虚部
    split(complexImage, planes);

    // 计算幅度
    Mat magnitudeImage;
    magnitude(planes[0], planes[1], magnitudeImage);

    // 计算相位谱
    Mat phaseImage;
    phase(planes[0], planes[1], phaseImage);

    // 直接保存原始幅度和相位信息，不进行对数尺度转换和象限交换
    frequencySpectrum = magnitudeImage;
    phaseSpectrum = phaseImage;
}






void PhaseCongruency::calculatePolarMatrices()
{
    int cols = image.cols;
    int rows = image.rows;
    // 创建xrange和yrange
    cv::Mat xRange = cv::Mat::zeros(1, cols, CV_32F);
    cv::Mat yRange = cv::Mat::zeros(rows, 1, CV_32F);

    if (cols % 2 == 1) {
        for (int i = 0; i < cols; ++i) {
            xRange.at<float>(0, i) = (i - (cols - 1) / 2.0) / (cols - 1);
        }
    }
    else {
        for (int i = 0; i < cols; ++i) {
            xRange.at<float>(0, i) = (i - cols / 2.0) / cols;
        }
    }

    if (rows % 2 == 1) {
        for (int i = 0; i < rows; ++i) {
            yRange.at<float>(i, 0) = (i - (rows - 1) / 2.0) / (rows - 1);
        }
    }
    else {
        for (int i = 0; i < rows; ++i) {
            yRange.at<float>(i, 0) = (i - rows / 2.0) / rows;
        }
    }

    // 创建坐标网格
    cv::Mat x, y;
    cv::repeat(xRange, rows, 1, x);
    cv::repeat(yRange, 1, cols, y);

    // 计算 radius 和 theta
    cv::sqrt(x.mul(x) + y.mul(y), radius);
    theta = cv::Mat(rows, cols, CV_32F);
    for (int i = 0; i < theta.rows; ++i) {
        for (int j = 0; j < theta.cols; ++j) {
            theta.at<float>(i, j) = std::atan2(-y.at<float>(i, j), x.at<float>(i, j));
        }
    }

    // 应用 ifftshift
    radius = ifftshift(radius);
    theta = ifftshift(theta);

    // 调整 radius 的 (1,1) 值
    radius.at<float>(0, 0) = 1;

    // 计算 sintheta 和 costheta
    applySinCos(theta, sintheta, costheta);
}

void PhaseCongruency::phasecong3()
{

    
    cv::Mat covx2 = cv::Mat::zeros(image.size(), frequencySpectrum.type());
    cv::Mat covy2 = cv::Mat::zeros(image.size(), frequencySpectrum.type());
    cv::Mat covxy = cv::Mat::zeros(image.size(), frequencySpectrum.type());
    vector<cv::Mat>EnergyV(3);    //累积总能量向量的矩阵，用于特征定位和类型计算
    EnergyV[0] = cv::Mat::zeros(image.size(), frequencySpectrum.type());
    EnergyV[1] = cv::Mat::zeros(image.size(), frequencySpectrum.type());
    EnergyV[2] = cv::Mat::zeros(image.size(), frequencySpectrum.type());
    std::vector<cv::Mat> logGabor;
    Mat lowPassFilter = lowpassfilter(imageFFT, cutOff, 15);

    createLogGaborFilters(radius, lowPassFilter, logGabor, nscale, 3, 2.1, 0.55);
    
    //主循环
    for (int o = 0; o < norient; o++)
    {
        cv::Mat spread;
        constructAngularFilterSpreadFunction(sintheta, costheta, spread, norient, o);
        cv::Mat sumE_ThisOrient = cv::Mat::zeros(image.size(), frequencySpectrum.type());

        cv::Mat sumO_ThisOrient = cv::Mat::zeros(image.size(), frequencySpectrum.type());
        cv::Mat sumAn_ThisOrient = cv::Mat::zeros(image.size(), frequencySpectrum.type());

        cv::Mat Energy = cv::Mat::zeros(image.size(), frequencySpectrum.type());
        float tau = 0.0;
        cv::Mat maxAn = cv::Mat::zeros(image.size(), frequencySpectrum.type());
        for (int s = 0; s < nscale; s++)
        {
            cv::Mat filter;
            multiply(logGabor[s], spread, filter);
            cv::Mat result;
            filter.convertTo(filter, CV_32F);
            multiply(frequencySpectrum, filter, result);
            EO[s][o] = combineMagnitudeAndPhase(result, phaseSpectrum);  //逆傅里叶变换后的图像即matlab代码中的EO{s,o}
            // 分离实部和虚部
            cv::Mat channels[2];
            cv::split(EO[s][o], channels);// channels[0] 是实部，channels[1] 是虚部
            Mat EO_real = channels[0];
            Mat EO_imag = channels[1];

            // 计算振幅图像
            cv::Mat An;//EO[s][o]的振幅
            cv::magnitude(EO_real, EO_imag, An); // EO_real 是实部，EO_imag 是虚部
            //imshow("an", An);
            //cout << "An" << An.at<float>(0, 100);
            //waitKey(0);
            sumAn_ThisOrient = sumAn_ThisOrient + An;
            sumE_ThisOrient = sumE_ThisOrient + EO_real;
            sumO_ThisOrient = sumO_ThisOrient + EO_imag;

            if (s == 0)
            {
                if (noiseMethod == -1)
                    tau = median(sumAn_ThisOrient) / std::sqrt(std::log(4));
                else if (noiseMethod == -2)
                    tau = rayleighmode(sumAn_ThisOrient);
                maxAn = An;
            }
            else
                maxAn = cv::max(maxAn, An);
        }
        double angl = o * CV_PI / norient;

        // 累积总能量向量
        EnergyV[0] += sumE_ThisOrient;
        EnergyV[1] += cos(angl) * sumO_ThisOrient;
        EnergyV[2] += sin(angl) * sumO_ThisOrient;
        cv::Mat XEnergy = cv::Mat::zeros(sumE_ThisOrient.size(), sumE_ThisOrient.type());
        cv::sqrt(sumE_ThisOrient.mul(sumE_ThisOrient) + sumO_ThisOrient.mul(sumO_ThisOrient), XEnergy);
        XEnergy += epsilon;
        cv::Mat MeanE = sumE_ThisOrient / XEnergy;
        cv::Mat MeanO = sumO_ThisOrient / XEnergy;
        for (int s = 0; s < nscale; s++)
        {
            cv::Mat channels[2];
            cv::split(EO[s][o], channels);// channels[0] 是实部，channels[1] 是虚部
            Mat E = channels[0];
            Mat O = channels[1];
            cv::Mat E_MeanE;
            multiply(E, MeanE, E_MeanE);
            cv::Mat O_MeanO;
            multiply(O, MeanO, O_MeanO);
            cv::Mat E_MeanO;
            multiply(E, MeanO, E_MeanO);
            cv::Mat O_MeanE;
            multiply(O, MeanE, O_MeanE);
            Energy += E_MeanE + O_MeanO - cv::abs(E_MeanO - O_MeanE);
        }

        //自动确定噪声阈值
        if (noiseMethod >= 0)
        {
            this->T = noiseMethod;
        }
        else
        {
            float totalTau = tau * (1.0 - (pow((1.0 / mult), nscale))) / (1.0 - (1.0 / mult));
            float EstNoiseEnergyMean = totalTau * sqrt(CV_PI / 2);
            float EstNoiseEnergySigma = totalTau * sqrt((4 - CV_PI) / 2);
            this->T = EstNoiseEnergyMean + k * EstNoiseEnergySigma;
        }
        Energy = cv::max(Energy - this->T, 0);
        cv::Mat width = (sumAn_ThisOrient / (maxAn + epsilon) - 1) / (nscale - 1);
        cv::Mat result = cutOff - width;
        result = result * g;
        // 计算指数
        cv::exp(result, result);
        cv::Mat weight = 1.0 / (1 + result);
        cv::Mat weight_Energy;
        multiply(weight, Energy, weight_Energy);
        this->PC[o] = weight_Energy / sumAn_ThisOrient;   //在该方向上的相位一致性图像

        this->pcSum = this->pcSum + this->PC[o];
        cv::Mat covx = this->PC[o] * cos(angl);
        cv::Mat covy = this->PC[o] * sin(angl);
        cv::Mat covx_2;
        multiply(covx, covx, covx_2);
        cv::Mat covy_2;
        multiply(covy, covy, covy_2);
        cv::Mat covx_y;
        multiply(covx, covy, covx_y);
        covx2 = covx2 + covx_2;// a
        covy2 = covy2 + covy_2;// c
        covxy = covxy + covx_y;// b
    }
    covx2 = covx2 / (norient / 2);
    covy2 = covy2 / (norient / 2);
    covxy = 4 * covxy / norient;
    cv::Mat covxy_2;
    multiply(covxy, covxy, covxy_2);
    cv::Mat covx2_minus_y2_2;
    multiply(covx2, covy2, covx2_minus_y2_2);
    cv::Mat sqrt_covxy_2_add_covx2_minus_y2_2;
    cv::sqrt(covxy_2 + covx2_minus_y2_2, sqrt_covxy_2_add_covx2_minus_y2_2);
    cv::Mat denom = sqrt_covxy_2_add_covx2_minus_y2_2 + epsilon;
    this->M = (covy2 + covx2 + denom) / 2; // Maximum moment
    this->m = (covy2 + covx2 - denom) / 2; // minimum moment
    this->OR = Atan2(EnergyV[2], EnergyV[1], 0); //hopc中的arctan(b,a)公式8
    convertAngles(this->OR);// 将角度 -pi..0 转换为 0..pi,方向以 0 到 180 度表示
    cv::Mat EnergyV1_2;
    multiply(EnergyV[1], EnergyV[1], EnergyV1_2);
    cv::Mat EnergyV2_2;
    multiply(EnergyV[2], EnergyV[2], EnergyV2_2);
    cv::Mat OddV;
    cv::sqrt(EnergyV1_2 + EnergyV2_2, OddV);
    this->featType = Atan2(EnergyV[0], OddV, 0);  //相位 φno

}