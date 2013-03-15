#include "FeatureAlgorithm.hpp"
#include <cassert>

FeatureAlgorithm::FeatureAlgorithm(std::string n, cv::Ptr<cv::FeatureDetector> d, cv::Ptr<cv::DescriptorExtractor> e, cv::Ptr<cv::DescriptorMatcher> m)
: name(n)
, knMatchSupported(false)
, detector(d)
, extractor(e)
, matcher(m)
{
    assert(d);
    assert(e);
    assert(m);
}

FeatureAlgorithm::FeatureAlgorithm(std::string n, cv::Ptr<cv::Feature2D> fe, cv::Ptr<cv::DescriptorMatcher> m)
: name(n)
, knMatchSupported(false)
, featureEngine(fe)
, matcher(m)
{
}


bool FeatureAlgorithm::extractFeatures(const cv::Mat& image, Keypoints& kp, Descriptors& desc) const
{
    assert(!image.empty());

    if (featureEngine)
    {
        (*featureEngine)(image, cv::noArray(), kp, desc);
    }
    else
    {
        detector->detect(image, kp);
    
        if (kp.empty())
            return false;
    
        extractor->compute(image, kp, desc);
    }
    
    
    return kp.size() > 0;
}

void FeatureAlgorithm::matchFeatures(const Descriptors& train, const Descriptors& query, Matches& matches) const
{
    matcher->match(query, train, matches);
}

void FeatureAlgorithm::matchFeatures(const Descriptors& train, const Descriptors& query, int k, std::vector<Matches>& matches) const
{
    assert(knMatchSupported);
    matcher->knnMatch(query, train, matches, k);
}

