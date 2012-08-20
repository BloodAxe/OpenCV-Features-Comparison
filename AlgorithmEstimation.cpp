#include "AlgorithmEstimation.hpp"

bool computeMatchesDistanceStatistics(const Matches& matches, float& meanDistance, float& stdDev)
{
  if (matches.empty())
    return false;

  std::vector<float> distances(matches.size());
  for (size_t i=0; i<matches.size(); i++)
    distances[i] = matches[i].distance;

  cv::Scalar mean, dev;
  cv::meanStdDev(distances, mean, dev);

  meanDistance = static_cast<float>(mean.val[0]);
  stdDev       = static_cast<float>(dev.val[0]);

  return false;
}

void ratioTest(const std::vector<Matches>& knMatches, float maxRatio, Matches& goodMatches)
{
  goodMatches.clear();

  for (size_t i=0; i< knMatches.size(); i++)
  {
    const cv::DMatch& best = knMatches[i][0];
    const cv::DMatch& good = knMatches[i][1];

    assert(best.distance <= good.distance);
    float ratio = (best.distance / good.distance);

    if (ratio <= maxRatio)
    {
      goodMatches.push_back(best);
    }
  }
}

bool performEstimation
(
    const FeatureAlgorithm& alg,
    const ImageTransformation& transformation,
    const cv::Mat& sourceImage,
    std::vector<FrameMatchingStatistics>& stat
)
{
  Keypoints   sourceKp;
  Descriptors sourceDesc;

  cv::Mat gray;
  if (sourceImage.channels() == 3)
      cv::cvtColor(sourceImage, gray, CV_BGR2GRAY);
  else if (sourceImage.channels() == 4)
      cv::cvtColor(sourceImage, gray, CV_BGRA2GRAY);
  else if(sourceImage.channels() == 1)
      gray = sourceImage;

  if (!alg.extractFeatures(gray, sourceKp, sourceDesc))
    return false;

  std::vector<float> x = transformation.getX();
  stat.resize(x.size());

  const int count = x.size();

  cv::Mat     transformedImage;
  Keypoints   resKpReal;
  Descriptors resDesc;
  Matches     matches;

  // To convert ticks to milliseconds
  const double toMsMul = 1000. / cv::getTickFrequency();

#pragma omp parallel for private(transformedImage, resKpReal, resDesc, matches)
  for (int i = 0; i < count; i++)
  {
    float       arg = x[i];
    FrameMatchingStatistics& s = stat[i];

    transformation.transform(arg, gray, transformedImage);

    int64 start = cv::getTickCount();

    alg.extractFeatures(transformedImage, resKpReal, resDesc);

    // Initialize required fields
    s.isValid        = resKpReal.size() > 0;
    s.argumentValue  = arg;

    if (!s.isValid)
        continue;

    if (alg.knMatchSupported)
    {
      std::vector<Matches> knMatches;
      alg.matchFeatures(sourceDesc, resDesc, 2, knMatches);
      ratioTest(knMatches, 0.75, matches);

      // Compute percent of false matches that were rejected by ratio test
      s.ratioTestFalseLevel = (float)(knMatches.size() - matches.size()) / (float) knMatches.size();
    }
    else
    {
      alg.matchFeatures(sourceDesc, resDesc, matches);
    }

    int64 end = cv::getTickCount();

    Matches correctMatches;
    cv::Mat homography;
    bool homographyFound = ImageTransformation::findHomography(sourceKp, resKpReal, matches, correctMatches, homography);

    // Some simple stat:
    s.isValid        = homographyFound;
    s.totalKeypoints = resKpReal.size();
    s.consumedTimeMs = (end - start) * toMsMul;

    // Compute overall percent of matched keypoints
    s.percentOfMatches      = (float) matches.size() / (float)(std::min(sourceKp.size(), resKpReal.size()));
    s.correctMatchesPercent = (float) correctMatches.size() / (float)matches.size();

    // Compute matching statistics
    computeMatchesDistanceStatistics(correctMatches, s.meanDistance, s.stdDevDistance);
  }

  return true;
}

