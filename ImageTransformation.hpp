#ifndef ImageTransformation_hpp
#define ImageTransformation_hpp

#include <opencv2/opencv.hpp>

typedef std::vector<cv::KeyPoint> Keypoints;
typedef cv::Mat                   Descriptors;
typedef std::vector<cv::DMatch>   Matches;

class ImageTransformation
{
public:
    std::string name;
    
	virtual std::vector<float> getX() const = 0;
    
	virtual void transform(float t, const cv::Mat& source, cv::Mat& result) const = 0;

    virtual bool canTransformKeypoints() const;
    virtual void transform(float t, const Keypoints& source, Keypoints& result) const;

    virtual cv::Mat getHomography(float t, const cv::Mat& source) const;

    virtual ~ImageTransformation();

    static bool findHomography( const Keypoints& source, const Keypoints& result, const Matches& input, Matches& inliers, cv::Mat& homography);

    
protected:

    ImageTransformation(const std::string& transformationName)
    : name(transformationName)
    {
        
    }
};

class ImageRotationTransformation : public ImageTransformation
{
public:
    ImageRotationTransformation(float startAngleInDeg, float endAngleInDeg, float step, cv::Point2f rotationCenterInUnitSpace);
    
	virtual std::vector<float> getX() const;
    
	virtual void transform(float t, const cv::Mat& source, cv::Mat& result)const ;
    
    virtual cv::Mat getHomography(float t, const cv::Mat& source) const;

private:
    float m_startAngleInDeg;
    float m_endAngleInDeg;
    float m_step;
    
    cv::Point2f m_rotationCenterInUnitSpace;
    
    std::vector<float> m_args;
};

class ImageScalingTransformation : public ImageTransformation
{
public:
    ImageScalingTransformation(float minScale, float maxScale, float step);
    
	virtual std::vector<float> getX() const;
    
	virtual void transform(float t, const cv::Mat& source, cv::Mat& result)const ;

    virtual cv::Mat getHomography(float t, const cv::Mat& source) const;

private:
    float m_minScale;
    float m_maxScale;
    float m_step;
    
    std::vector<float> m_args;
};

class GaussianBlurTransform : public ImageTransformation
{
public:
    GaussianBlurTransform(int maxKernelSize);
    
	virtual std::vector<float> getX() const;
    
	virtual void transform(float t, const cv::Mat& source, cv::Mat& result)const ;
private:
    int m_maxKernelSize;
    std::vector<float> m_args;
};

class BrightnessImageTransform : public ImageTransformation
{
public:
    BrightnessImageTransform(int min, int max, int step);
        
	virtual std::vector<float> getX() const;
    
	virtual void transform(float t, const cv::Mat& source, cv::Mat& result)const ;
    
private:
    int m_min;
    int m_max;
    int m_step;
    std::vector<float> m_args;
};

class CombinedTransform : public ImageTransformation
{
public:
    typedef enum
    {
        // Generate resulting X vector as list of all possible combinations of first and second args
        Full,
        
        // Largest argument vector used as is, the values for other vector is copied
        Extrapolate,
        
        // Smallest argument vector used as is, the values for other vector is interpolated from other
        Interpolate
    } ParamCombinationType;
    
    CombinedTransform(cv::Ptr<ImageTransformation> first, cv::Ptr<ImageTransformation> second, ParamCombinationType type = Extrapolate);
        
	virtual std::vector<float> getX() const ;
    
	virtual void transform(float t, const cv::Mat& source, cv::Mat& result) const ;
    
    virtual bool canTransformKeypoints() const;
    virtual void transform(float t, const Keypoints& source, Keypoints& result) const;
    
    virtual cv::Mat getHomography(float t, const cv::Mat& source) const;
    
private:
    std::vector< float >                   m_x;
    std::vector< std::pair<float, float> > m_params;
    
    cv::Ptr<ImageTransformation> m_first;
    cv::Ptr<ImageTransformation> m_second;
};

class PerspectiveTransform : public ImageTransformation
{
public:
    PerspectiveTransform(int count);
    
	virtual std::vector<float> getX() const;
    
	virtual void transform(float t, const cv::Mat& source, cv::Mat& result) const;
    
    virtual cv::Mat getHomography(float t, const cv::Mat& source) const;
    
private:
    static cv::Mat warpPerspectiveRand( cv::RNG& rng );
    
    float m_min;
    float m_max;
    float m_step;
    
    std::vector<float>   m_args;
    std::vector<cv::Mat> m_homographies;
};

#endif