#ifndef FeatureAlgorithm_hpp
#define FeatureAlgorithm_hpp

#include <opencv2/opencv.hpp>

typedef std::vector<cv::KeyPoint> Keypoints;
typedef cv::Mat                   Descriptors;
typedef std::vector<cv::DMatch>   Matches;

//! Represents combination of feature detector, descripotr extractor and matcher algorithms for test
class FeatureAlgorithm
{
public:
    FeatureAlgorithm(std::string name, cv::Ptr<cv::FeatureDetector> d, cv::Ptr<cv::DescriptorExtractor> e, cv::Ptr<cv::DescriptorMatcher> m);

    FeatureAlgorithm(std::string name, cv::Ptr<cv::Feature2D> featureEngine, cv::Ptr<cv::DescriptorMatcher> m);

    //! Human-friendly name of detection/extraction/matcher combination.
	std::string name;

    //! If true, a KNN-matching and ratio test will be enabled for matching descriptors.
    bool knMatchSupported;
    
    //! Extracts feature points and compute descriptors from given image.
	bool extractFeatures(const cv::Mat& image, Keypoints& kp, Descriptors& desc) const;
    
    //! Finds correspondences using regular match.
    void matchFeatures(const Descriptors& train, const Descriptors& query, Matches& matches) const;
    
    //! KNN match features.
    void matchFeatures(const Descriptors& train, const Descriptors& query, int k, std::vector<Matches>& matches) const;
    
    
private:
    cv::Ptr<cv::Feature2D>           featureEngine;

    cv::Ptr<cv::FeatureDetector>     detector;
	cv::Ptr<cv::DescriptorExtractor> extractor;
	cv::Ptr<cv::DescriptorMatcher>   matcher;
};

#endif