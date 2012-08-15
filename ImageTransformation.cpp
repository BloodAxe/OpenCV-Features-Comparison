#include "ImageTransformation.hpp"

#pragma mark - ImageTransformation default implementation

bool ImageTransformation::canTransformKeypoints() const
{
    return false;
}

void ImageTransformation::transform(float t, const Keypoints& source, Keypoints& result) const
{
    
}

ImageTransformation::~ImageTransformation()
{
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
    cv::Size dstSize(source.cols * t, source.rows * t);
    cv::resize(source, result, dstSize, CV_INTER_AREA);
}

#pragma mark - GaussianBlurTransform implementation

GaussianBlurTransform::GaussianBlurTransform(int maxKernelSize)
: ImageTransformation("Gaussian blur")
, m_maxKernelSize(maxKernelSize)
{
    for (float arg = 1; arg <= maxKernelSize; arg++)
        m_args.push_back(arg);
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
    for (int arg = min; arg <= max; arg++)
        m_args.push_back(arg);
}

std::vector<float> BrightnessImageTransform::getX() const
{
    return m_args;
}

void BrightnessImageTransform::transform(float t, const cv::Mat& source, cv::Mat& result)const
{
    result = source + cv::Scalar(t,t,t,t);
}