#include "AlgorithmEstimation.hpp"

FrameMatchingStatistics::FrameMatchingStatistics()
{
    totalKeypoints = 0;
    argumentValue = 0;
	percentOfMatches = 0;
	ratioTestFalseLevel = 0;
	meanDistance = 0;
	stdDevDistance = 0;
}


bool computeMatchesDistanceStatistics(const Matches& matches, float& meanDistance, float& stdDev)
{
    if (matches.empty())
        return false;
    
    std::vector<float> distances(matches.size());
    for (int i=0; i<matches.size(); i++)
        distances[i] = matches[i].distance;
    
    cv::Scalar mean, dev;
    cv::meanStdDev(distances, mean, dev);
    
    meanDistance = mean.val[0];
    stdDev        = dev.val[0];
    
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

bool performEstimation(const FeatureAlgorithm& alg,
                       const ImageTransformation& transformation,
                       const cv::Mat& sourceImage,
                       std::vector<FrameMatchingStatistics>& stat)
{
	Keypoints   sourceKp;
	Descriptors sourceDesc;
    
	if (!alg.extractFeatures(sourceImage, sourceKp, sourceDesc))
		return false;
    
	std::vector<float> x = transformation.getX();
	stat.resize(x.size());
    
	//const bool useTransformedKp = transformation.canTransformKeypoints();
    
#pragma omp parallel for
	for (int i=0; i<x.size(); i++)
	{
		float       arg = x[i];
		cv::Mat     res;
		Keypoints   resKpEstimate;
		Keypoints   resKpReal;
		Descriptors resDesc;
        Matches     matches;
        FrameMatchingStatistics& s = stat[i];

		transformation.transform(arg, sourceImage, res);
        alg.extractFeatures(res, resKpReal, resDesc);
        
        cv::imshow("Transformed image", res);
        cv::waitKey(5);
        
        if (alg.knMatchSupported)
        {
            std::vector<Matches> knMatches;
            alg.matchFeatures(sourceDesc, resDesc, 2, knMatches);
            ratioTest(knMatches, 0.75, matches);
            
            // Compute percent of false matches that were rejected by ratio test
            s.ratioTestFalseLevel =(float)(knMatches.size() - matches.size()) / (float) knMatches.size();
        }
        else
        {
            alg.matchFeatures(sourceDesc, resDesc, matches);
        }
        
        // Some simple stat:
        s.argumentValue  = arg;
        s.totalKeypoints = resKpReal.size();
        
		// Compute overall percent of matched keypoints
        s.percentOfMatches         = (float) matches.size() / (float)(std::max(sourceKp.size(), resKpReal.size()));
        
        // Compute matching statistics
        computeMatchesDistanceStatistics(matches, s.meanDistance, s.stdDevDistance);
	}
    
    return true;
}

#pragma mark - Printing statistics

std::ostream& header(std::ostream& str)
{
    return str
    << "\"Value\""               << "\t"
    << "\"Total keypoints\""               << "\t"
    << "\"Percent of matches\""  << "\t"
    << "\"Mean distance\""       << "\t"
    << "\"StdDev\""      << "\t"
    << "\"Ratio test false level\"";
}


std::ostream& operator<<(std::ostream& str, const FrameMatchingStatistics& stat)
{
    return str << stat.argumentValue << "\t"
               << stat.totalKeypoints << "\t"
               << stat.percentOfMatches << "\t"
               << stat.meanDistance << "\t"
               << stat.stdDevDistance << "\t"
               << stat.ratioTestFalseLevel;
}

std::ostream& operator<<(std::ostream& str, const std::vector<FrameMatchingStatistics>& stat)
{
    header(str) << std::endl;
    std::copy(stat.begin(), stat.end(), std::ostream_iterator<FrameMatchingStatistics>(str, "\n"));
    return str;
}

