#ifndef AlgorithmEstimation_hpp
#define AlgorithmEstimation_hpp

#include "CollectedStatistics.hpp"
#include "FeatureAlgorithm.hpp"
#include "ImageTransformation.hpp"

bool computeMatchesDistanceStatistics(const Matches& matches, float& meanDistance, float& stdDev);


bool performEstimation(const FeatureAlgorithm& alg,
  const ImageTransformation& transformation,
  const cv::Mat& sourceImage,
  SingleRunStatistics& stat);


#endif