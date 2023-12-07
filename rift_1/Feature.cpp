#include"Feature.h"
#include"PhaseCongruency.h"

void FeatureDetection(const cv::Mat& m, int npt, std::vector<cv::Point2f>& kpts)
{
    double a, b;
    cv::minMaxLoc(m, &a, &b);
    Mat im = (m - a) / (b - a);  //��һ��ͼ��
    Mat Feature;
    im.convertTo(Feature, CV_8U, 255.0); //����ֵ�������Ŵ�
    // ʹ�� FAST �㷨���������
    std::vector<cv::KeyPoint> keypoints;
    cv::FAST(Feature, keypoints, 0.0001, 1);
    //std::cout << "Number of keypoints detected: " << keypoints.size() << std::endl;
    // ѡ����ǿ�� npt ��������
    cv::KeyPointsFilter::retainBest(keypoints, npt);

    //// ����������
    //cv::Mat outputImage = img.clone();
    //cv::drawKeypoints(img, keypoints, outputImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    //// ��ʾͼ��
    //cv::imshow("Feature Points", outputImage);
    //cv::waitKey(0);

    // ��ȡ�������λ��
    kpts.clear();
    for (auto& kp : keypoints) {
        kpts.push_back(kp.pt);
    }

}

cv::Mat CreateMIM(const PhaseCongruency& obj)
{
    int o = obj.norient;
    int s = obj.nscale;
    int yim = obj.EO[0][0].rows;
    int xim = obj.EO[0][0].cols;

    // ��ʼ��������о���
    //std::vector<cv::Mat> CS(no, cv::Mat::zeros(yim, xim, CV_32F));
    std::vector<cv::Mat> CS(o);
    for (int j = 0; j < o; ++j) {
        CS[j] = cv::Mat::zeros(yim, xim, CV_32F);
    }
    for (int j = 0; j < o; ++j) {
        for (int i = 0; i < s; ++i) {
            cv::Mat channels[2];
            cv::split(obj.EO[i][j], channels);
            // �������ͼ��
            cv::Mat An;//EO[s][o]�����
            cv::magnitude(channels[0], channels[1], An);
            // �ۼ�ģֵ
            CS[j] += An;
        }
    }

    // ��ÿ������λ���ҵ���Ӧ���ķ���
    cv::Mat MIM = cv::Mat::zeros(yim, xim, CV_32F);
    for (int y = 0; y < yim; ++y) {
        for (int x = 0; x < xim; ++x) {
            float maxVal = -1;
            int maxIdx = -1;
            for (int j = 0; j < o; ++j) {
                float val = CS[j].at<float>(y, x);
                //cout << val;
                //cout << j;
                if (val >= maxVal) {
                    maxVal = val;
                    maxIdx = j;
                }
            }
            MIM.at<float>(y, x) = static_cast<float>(maxIdx);
        }
    }
    return MIM;
}

cv::Mat convertDescriptors(const std::vector<std::vector<float>>& des) {
    cv::Mat desMat(des.size(), des[0].size(), CV_32F);
    for (size_t i = 0; i < des.size(); ++i) {
        memcpy(desMat.ptr<float>(static_cast<int>(i)), des[i].data(), des[i].size() * sizeof(float));
    }
    return desMat;
}


// �� cv::Point2f ת��Ϊ cv::KeyPoint
std::vector<cv::KeyPoint> convertToKeyPoints(const std::vector<cv::Point2f>& points) {
    std::vector<cv::KeyPoint> keypoints;
    for (const auto& pt : points) {
        keypoints.push_back(cv::KeyPoint(pt, 1.0f));
    }
    return keypoints;
}

cv::Mat extractPatch(const cv::Mat& img, const cv::Point2f& center, int r) {
    int x = static_cast<int>(center.x);
    int y = static_cast<int>(center.y);
    int r2 = r / 2;
    cv::Rect roi(std::max(x - r2, 0), std::max(y - r2, 0), std::min(r, img.cols - x + r2), std::min(r, img.rows - y + r2));

    return img(roi).clone();
}

// ���㲢��������������
std::vector<float> computeDescriptor(const cv::Mat& patch, int no, int nbin) {
    // ����ֱ��ͼ���ҵ����ֵ����
    int histSize[] = { nbin };
    float range[] = { 0, static_cast<float>(nbin) };
    const float* ranges[] = { range };
    cv::Mat hist;
    cv::calcHist(&patch, 1, 0, cv::Mat(), hist, 1, histSize, ranges, true, false);
    double maxVal;
    cv::Point maxLoc;
    cv::minMaxLoc(hist, nullptr, &maxVal, nullptr, &maxLoc);

    // ����ͼ����ֵ   ��ת������
    cv::Mat patch_rot = patch - maxLoc.y + 1;
    patch_rot.forEach<float>([no](float& p, const int* position) -> void {
        if (p < 0) p += no;
        });
    //cv::Mat patch_rot = patch;
    // ��������������

    int ys = patch_rot.rows;
    int xs = patch_rot.cols;
    std::vector<float> descriptor(no * no * nbin, 0);
    for (int j = 0; j < no; ++j) {
        for (int i = 0; i < no; ++i) {
            cv::Rect roi(round((i)*xs / no), round((j)*ys / no), xs / no, ys / no);
            cv::Mat clip = patch_rot(roi);
            cv::calcHist(&clip, 1, 0, cv::Mat(), hist, 1, histSize, ranges, true, false);
            for (int k = 0; k < nbin; ++k) {
                descriptor[j * no * nbin + i * nbin + k] = hist.at<float>(k);
            }
        }
    }
    // �������Ĺ�һ��
    float normVal = cv::norm(descriptor, cv::NORM_L2);
    if (normVal != 0) {
        for (auto& val : descriptor) {
            val /= normVal;
        }
    }

    return descriptor;
}

void FeatureDescribe(std::vector<cv::Point2f>& kpts, const cv::Mat& MIM, std::vector<vector<float>>& des,int no, int nbin,int patch_size)
{
    for (const auto& pt : kpts) {
        cv::Mat patch = extractPatch(MIM, pt, patch_size);
        // ����ȡ�� patch ���д���
        cv::Mat resizedPatch;
        cv::resize(patch, resizedPatch, cv::Size(patch_size + 1, patch_size + 1));
        des.push_back(computeDescriptor(patch, no, nbin));
    }
}
