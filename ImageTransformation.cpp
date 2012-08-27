#include "ImageTransformation.hpp"

#pragma mark - ImageTransformation default implementation

bool ImageTransformation::canTransformKeypoints() const
{
    return false;
}

void ImageTransformation::transform(float t, const Keypoints& source, Keypoints& result) const
{
}

cv::Mat ImageTransformation::getHomography(float t, const cv::Mat& source) const
{
    return cv::Mat::eye(3, 3, CV_64FC1);
}


ImageTransformation::~ImageTransformation()
{
}

bool ImageTransformation::findHomography( const Keypoints& source, const Keypoints& result, const Matches& input, Matches& inliers, cv::Mat& homography)
{
    if (input.size() < 8)
        return false;
    
    std::vector<cv::Point2f> srcPoints, dstPoints;
    const int pointsCount = input.size();
    
    for (int i=0; i<pointsCount; i++)
    {
        srcPoints.push_back(source[input[i].trainIdx].pt);
        dstPoints.push_back(result[input[i].queryIdx].pt);
    }
    
    std::vector<unsigned char> status;
    homography = cv::findHomography(srcPoints, dstPoints, CV_FM_RANSAC, 3, status);
    
    inliers.clear();
    for (int i=0; i<pointsCount; i++)
    {
        if (status[i])
        {
            inliers.push_back(input[i]);
        }
    }
    
    return true;
}

#pragma mark - ImageScalingTransformation implementation

ImageRotationTransformation::ImageRotationTransformation(float startAngleInDeg, float endAngleInDeg, float step, cv::Point2f rotationCenterInUnitSpace)
: ImageTransformation("Rotation")
, m_startAngleInDeg(startAngleInDeg)
, m_endAngleInDeg(endAngleInDeg)
, m_step(step)
, m_rotationCenterInUnitSpace(rotationCenterInUnitSpace)
{
    // Fill the arguments
    for (float arg = startAngleInDeg; arg <= endAngleInDeg; arg += step)
        m_args.push_back(arg);
}

std::vector<float> ImageRotationTransformation::getX() const
{
    return m_args;
}

void ImageRotationTransformation::transform(float t, const cv::Mat& source, cv::Mat& result) const
{
    cv::Point2f center(source.cols * m_rotationCenterInUnitSpace.x, source.cols * m_rotationCenterInUnitSpace.y);
    cv::Mat rotationMat = cv::getRotationMatrix2D(center, t, 1);
    cv::warpAffine(source, result, rotationMat, source.size());
}

cv::Mat ImageRotationTransformation::getHomography(float t, const cv::Mat& source) const
{
    cv::Point2f center(source.cols * m_rotationCenterInUnitSpace.x, source.cols * m_rotationCenterInUnitSpace.y);
    cv::Mat rotationMat = cv::getRotationMatrix2D(center, t, 1);
    
    cv::Mat h = cv::Mat::zeros(3,3, CV_64FC1);
    h(cv::Range(0,2), cv::Range(0,3)) = rotationMat;
    return h;
}


#pragma mark - ImageScalingTransformation implementation

ImageScalingTransformation::ImageScalingTransformation(float minScale, float maxScale, float step)
: ImageTransformation("Scaling")
, m_minScale(minScale)
, m_maxScale(maxScale)
, m_step(step)
{
    // Fill the arguments
    for (float arg = minScale; arg <= maxScale; arg += step)
        m_args.push_back(arg);
}

std::vector<float> ImageScalingTransformation::getX() const
{
    return m_args;
}

void ImageScalingTransformation::transform(float t, const cv::Mat& source, cv::Mat& result)const
{
    cv::Size dstSize(static_cast<int>(source.cols * t + 0.5f), static_cast<int>(source.rows * t + 0.5f));
    cv::resize(source, result, dstSize, CV_INTER_AREA);
}

#pragma mark - GaussianBlurTransform implementation

GaussianBlurTransform::GaussianBlurTransform(int maxKernelSize)
: ImageTransformation("Gaussian blur")
, m_maxKernelSize(maxKernelSize)
{
    for (int arg = 1; arg <= maxKernelSize; arg++)
        m_args.push_back(static_cast<float>(arg));
}

std::vector<float> GaussianBlurTransform::getX() const
{
    return m_args;
}

void GaussianBlurTransform::transform(float t, const cv::Mat& source, cv::Mat& result)const
{
    int kernelSize = static_cast<int>(t) * 2 + 1;
    cv::GaussianBlur(source, result, cv::Size(kernelSize,kernelSize), 0);
}

#pragma mark - BrightnessImageTransform implementation

BrightnessImageTransform::BrightnessImageTransform(int min, int max, int step)
: ImageTransformation("Brightness change")
, m_min(min)
, m_max(max)
, m_step(step)
{
    for (int arg = min; arg <= max; arg += step)
        m_args.push_back(static_cast<float>(arg));
}

std::vector<float> BrightnessImageTransform::getX() const
{
    return m_args;
}

void BrightnessImageTransform::transform(float t, const cv::Mat& source, cv::Mat& result)const
{
    result = source + cv::Scalar(t,t,t,t);
}