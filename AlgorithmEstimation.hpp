#ifndef AlgorithmEstimation_hpp
#define AlgorithmEstimation_hpp

#include "FeatureAlgorithm.hpp"
#include "ImageTransformation.hpp"

struct FrameMatchingStatistics
{
    FrameMatchingStatistics();
    
    int totalKeypoints;
    
	float argumentValue;
	float percentOfMatches;
	float ratioTestFalseLevel;
	float meanDistance;
	float stdDevDistance;
    
    double consumedTimeMs;
    
    static std::ostream& header(std::ostream& str);
};

std::ostream& operator<<(std::ostream& str, const FrameMatchingStatistics& stat);
std::ostream& operator<<(std::ostream& str, const std::vector<FrameMatchingStatistics>& stat);

bool computeMatchesDistanceStatistics(const Matches& matches, float& meanDistance, float& stdDev);

bool performEstimation(const FeatureAlgorithm& alg,
                       const ImageTransformation& transformation,
                       const cv::Mat& sourceImage,
                       std::vector<FrameMatchingStatistics>& stat);


#endif