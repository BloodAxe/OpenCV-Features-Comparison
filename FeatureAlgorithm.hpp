#ifndef FeatureAlgorithm_hpp
#define FeatureAlgorithm_hpp

#include <opencv2/opencv.hpp>

typedef std::vector<cv::KeyPoint> Keypoints;
typedef cv::Mat                   Descriptors;
typedef std::vector<cv::DMatch>   Matches;

class FeatureAlgorithm
{
public:
    FeatureAlgorithm(std::string name, cv::Ptr<cv::FeatureDetector> d, cv::Ptr<cv::DescriptorExtractor> e, cv::Ptr<cv::DescriptorMatcher> m);
    
	std::string name;
		

    bool knMatchSupported;
    
	bool extractFeatures(const cv::Mat& image, Keypoints& kp, Descriptors& desc) const;
    
    void matchFeatures(const Descriptors& train, const Descriptors& query, Matches& matches) const;
    void matchFeatures(const Descriptors& train, const Descriptors& query, int k, std::vector<Matches>& matches) const;
    
    
private:
    cv::Ptr<cv::FeatureDetector>     detector;
	cv::Ptr<cv::DescriptorExtractor> extractor;
	cv::Ptr<cv::DescriptorMatcher>   matcher;
};

#endif