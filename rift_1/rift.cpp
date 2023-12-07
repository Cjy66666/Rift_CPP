// rift_1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <algorithm>
#include <chrono>
#include "PhaseCongruency.h"
#include"Utils.h"
#include"Feature.h"
using namespace std;
using namespace cv;




// 示例使用
int main() {

    Mat img1 = imread("your image path");
    Mat img2 = imread("your image path 2");
    auto start = std::chrono::high_resolution_clock::now();
    img1 = ensureEvenDimensions(img1);
    img2 = ensureEvenDimensions(img2);
    Mat img1_color = img1;
    Mat img2_color = img2;
    cvtColor(img1, img1, CV_BGR2GRAY);
    cvtColor(img2, img2, CV_BGR2GRAY);
    PhaseCongruency test1(img1);
    PhaseCongruency test2(img2);
    test1.calculatePolarMatrices();  //计算极坐标
    test1.phasecong3();
    test2.calculatePolarMatrices();
    test2.phasecong3();

    cv::Mat MIM1 =CreateMIM(test1); //获取最大索引图
    cv::Mat MIM2 = CreateMIM(test2);

    std::vector<cv::Point2f> kpts1; //特征点坐标
    std::vector<cv::Point2f> kpts2;
    FeatureDetection(test1.m, 5000,kpts1);//求特征点
    FeatureDetection(test2.m, 5000, kpts2);
    FeatureDetection(test1.M, 1000, kpts1);
    FeatureDetection(test2.M, 1000, kpts2);
    std::vector<vector<float>> des1;//特征描述符
    std::vector<vector<float>> des2;
    FeatureDescribe(kpts1,MIM1,des1);
    FeatureDescribe(kpts2, MIM2, des2);

    cv::Mat desMat1 = convertDescriptors(des1);
    cv::Mat desMat2 = convertDescriptors(des2);
    auto end = std::chrono::high_resolution_clock::now();

    // 计算运行时间
    std::chrono::duration<double, std::milli> duration = end - start;

    // 打印运行时间
    std::cout << "运行时间: " << duration.count()/1000 << " 秒" << std::endl;

    cv::BFMatcher matcher(cv::NORM_L2);

    // 使用 knnMatch 找到每个关键点的两个最近邻
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher.knnMatch(desMat1, desMat2, knnMatches, 2);

    // 应用 Lowe's Ratio Test
    std::vector<cv::DMatch> goodMatches;
    for (const auto& matchPair : knnMatches) {
        if (matchPair[0].distance < 1 * matchPair[1].distance) {
            goodMatches.push_back(matchPair[0]);
        }
    }

    // 提取 Lowe's Ratio Test 过滤后的匹配的关键点坐标
    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : goodMatches) {
        points1.push_back(kpts1[match.queryIdx]);
        points2.push_back(kpts2[match.trainIdx]);
    }

    // 使用 RANSAC 过滤错误的匹配
    cv::Mat inlierMask;
    cv::findHomography(points1, points2, cv::RANSAC, 1, inlierMask);

    // 从匹配中提取 RANSAC 过滤后的点
    std::vector<cv::DMatch> ransacMatches;
    for (size_t i = 0; i < inlierMask.rows; ++i) {
        if (inlierMask.at<uchar>(i)) {
            ransacMatches.push_back(goodMatches[i]);
        }
    }
    std::vector<cv::KeyPoint> keyPoints1 = convertToKeyPoints(kpts1);
    std::vector<cv::KeyPoint> keyPoints2 = convertToKeyPoints(kpts2);

    // 绘制 RANSAC 过滤后的匹配结果
    cv::Mat imgMatches;
    cv::drawMatches(img1_color, keyPoints1, img2_color, keyPoints2, ransacMatches, imgMatches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //cv::drawMatches(img1_color, keyPoints1, img2_color, keyPoints2, knnMatches, imgMatches);
    // 显示匹配结果
    cv::namedWindow("Matches", cv::WINDOW_NORMAL);
    cv::imshow("Matches", imgMatches);
    cv::waitKey(0);
    return 0;
}










