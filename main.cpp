
#include "ImageTransformation.hpp"
#include "AlgorithmEstimation.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

struct Line
{
    float                                 argument;
    std::vector<const FrameMatchingStatistics*> stats;
};

struct GroupedByArgument
{
    std::vector<std::string> algorithms;
    std::vector<Line>        lines;
};

class CollectedStatistics
{
public:
    SingleRunStatistics& getStatistics(std::string algorithmName, std::string transformationName)
    {
        return m_allStats[std::make_pair(algorithmName, transformationName)];
    }
    
    typedef std::map<std::string, const SingleRunStatistics*> InnerGroup;
    typedef std::map<std::string, InnerGroup>                 OuterGroup;

    typedef std::map<std::string, GroupedByArgument>          OuterGroupLine;

    OuterGroup groupByAlgorithmThenByTransformation() const
    {
        OuterGroup result;
        
        for (std::map<Key, SingleRunStatistics>::const_iterator i = m_allStats.begin(); i != m_allStats.end(); ++i)
        {
            result[i->first.first][i->first.second] = &(i->second);
        }
        
        return result;
    }
    
    OuterGroupLine groupByTransformationThenByAlgorithm() const
    {
        OuterGroup result;
        
        for (std::map<Key, SingleRunStatistics>::const_iterator i = m_allStats.begin(); i != m_allStats.end(); ++i)
        {
            result[i->first.second][i->first.first] = &(i->second);
        }
        
        OuterGroupLine line;
        
        for (OuterGroup::const_iterator tIter = result.begin(); tIter != result.end(); ++tIter)
        {
            std::string transformationName               = tIter->first;
            const CollectedStatistics::InnerGroup& inner = tIter->second;
            
            GroupedByArgument& lineStat = line[transformationName];
            
            std::vector<const SingleRunStatistics*> statitics;
            
            for (CollectedStatistics::InnerGroup::const_iterator algIter = inner.begin(); algIter != inner.end(); ++algIter)
            {
                std::string algName = algIter->first;
                
                lineStat.algorithms.push_back(algName);
                statitics.push_back(algIter->second);
            }
            
            const SingleRunStatistics& firstStat = *statitics.front();
            int argumentsCount = firstStat.size();
            
            for (int i=0; i<argumentsCount; i++)
            {
                Line l;
                l.argument = firstStat[i].argumentValue;
                
                for (int algIndex = 0; algIndex < statitics.size(); algIndex++)
                {
                    const SingleRunStatistics& s = *statitics[algIndex];
                    
                    l.stats.push_back(&s[i]);
                }

                lineStat.lines.push_back(l);
            }
        }
        
                
        return line;
    }
    
private:
    typedef std::pair<std::string, std::string> Key;
    
    std::map<Key, SingleRunStatistics> m_allStats;
};

std::ostream& printPerformanceStatistics(std::ostream& str, const CollectedStatistics& stat);
std::ostream& printMatchingRatioStatistics(std::ostream& str, const CollectedStatistics& stat);

int main(int argc, const char* argv[])
{
    // Print OpenCV build info:
    // std::cout << cv::getBuildInformation() << std::endl;
    
    std::vector<FeatureAlgorithm>              algorithms;
    std::vector<cv::Ptr<ImageTransformation> > transformations;
    
    // Initialize list of algorithm tuples:
    algorithms.push_back(FeatureAlgorithm("ORB - 2",
                                          new cv::ORB(),
                                          new cv::ORB(),
                                          new cv::BFMatcher(cv::NORM_HAMMING, false)));
    
    algorithms.push_back(FeatureAlgorithm("ORB - 3",
                                          new cv::ORB(500,1.2, 8,31,0, 3),
                                          new cv::ORB(500,1.2, 8,31,0, 3),
                                          new cv::BFMatcher(cv::NORM_HAMMING2, false)));

    algorithms.push_back(FeatureAlgorithm("ORB - 4",
                                          new cv::ORB(500,1.2, 8,31,0, 4),
                                          new cv::ORB(500,1.2, 8,31,0, 4),
                                          new cv::BFMatcher(cv::NORM_HAMMING2, false)));

    
    algorithms.push_back(FeatureAlgorithm("FAST+BRIEF",
                                          new cv::FastFeatureDetector(50),
                                          new cv::BriefDescriptorExtractor(),
                                          new cv::BFMatcher(cv::NORM_HAMMING, false)));

    
    algorithms.push_back(FeatureAlgorithm("SURF-BruteForce",
                                          new cv::SurfFeatureDetector(),
                                          new cv::SurfDescriptorExtractor(),
                                          new cv::BFMatcher(cv::NORM_L2, false)));

    algorithms.push_back(FeatureAlgorithm("SURF-Flann",
                                          new cv::SurfFeatureDetector(),
                                          new cv::SurfDescriptorExtractor(),
                                          new cv::FlannBasedMatcher()));

    /*
    algorithms.push_back(FeatureAlgorithm("SURF-FREAK",
                                          new cv::SurfFeatureDetector(),
                                          new cv::FREAK(),
                                          new cv::FlannBasedMatcher()));

    algorithms.push_back(FeatureAlgorithm("ORB-FREAK",
                                          new cv::OrbFeatureDetector(),
                                          new cv::FREAK(),
                                          new cv::FlannBasedMatcher()));
     */

    // Initialize list of used transformations:
    transformations.push_back(new GaussianBlurTransform(9));
    transformations.push_back(new BrightnessImageTransform(-127, +127,1));
    transformations.push_back(new ImageRotationTransformation(0, 360, 1, cv::Point2f(0.5f,0.5f)));
    transformations.push_back(new ImageScalingTransformation(0.25, 2, 0.01));
    
    if (argc < 2)
    {
        std::cout << "At least one input image should be passed" << std::endl;
    }
    
    for (int imageIndex = 1; imageIndex < argc; imageIndex++)
    {
        std::string testImagePath(argv[imageIndex]);
        cv::Mat testImage = cv::imread(testImagePath);
        
        CollectedStatistics fullStat;
        
        if (testImage.empty())
        {
            std::cout << "Cannot read image from " << testImagePath << std::endl;
        }
        
        std::cout << "[" << testImagePath << "]" << std::endl;
        
        for (size_t algIndex = 0; algIndex < algorithms.size(); algIndex++)
        {
            const FeatureAlgorithm& alg   = algorithms[algIndex];

            for (size_t transformIndex = 0; transformIndex < transformations.size(); transformIndex++)
            {
                const ImageTransformation& trans = *transformations[transformIndex].obj;

                performEstimation(alg, trans, testImage.clone(), fullStat.getStatistics(alg.name, trans.name));
            }
        }
        
        printPerformanceStatistics(std::cout, fullStat);
        printMatchingRatioStatistics(std::cout, fullStat);
    }
    
    return 0;
}

std::ostream& printMatchingRatioStatistics(std::ostream& str, const CollectedStatistics& stat)
{
    CollectedStatistics::OuterGroupLine report = stat.groupByTransformationThenByAlgorithm();
    
    for (CollectedStatistics::OuterGroupLine::const_iterator tIter = report.begin(); tIter != report.end(); ++tIter)
    {
        std::string transformationName = tIter->first;
        str << "[" << transformationName << "]" << std::endl;

        const GroupedByArgument& inner = tIter->second;

        str << "Argument\t";
        for (size_t i=0; i<inner.algorithms.size(); i++)
        {
            str << "\"" << inner.algorithms[i] << "\"\t";
        }
        str << std::endl;
        
        for (size_t i=0; i<inner.lines.size();i++)
        {
            const Line& l = inner.lines[i];
            str << l.argument << "\t";
            
            for (size_t j=0; j< l.stats.size(); j++)
            {
                str << l.stats[j]->percentOfMatches << "\t";
            }
            
            str << std::endl;
        }
    }

    return str;
}

std::ostream& printPerformanceStatistics(std::ostream& str, const CollectedStatistics& stat)
{
    str << "\"Algorithm\"\t"
        << "\"Average time per Frame\"\t"
        << "\"Average time per KeyPoint\"" << std::endl;

    CollectedStatistics::OuterGroup report = stat.groupByAlgorithmThenByTransformation();

    for (CollectedStatistics::OuterGroup::const_iterator alg = report.begin(); alg != report.end(); ++alg)
    {
        std::vector<double> timePerFrames;
        std::vector<double> timePerKeyPoint;
        
        for (CollectedStatistics::InnerGroup::const_iterator tIter = alg->second.begin(); tIter != alg->second.end(); ++tIter)
        {
            const SingleRunStatistics& runStatistics = *tIter->second;
            for (size_t i=0; i<runStatistics.size(); i++)
            {
                timePerFrames.push_back(runStatistics[i].consumedTimeMs);
                timePerKeyPoint.push_back(runStatistics[i].totalKeypoints > 0 ? (runStatistics[i].consumedTimeMs / runStatistics[i].totalKeypoints) : 0);
            }
        }
        
        double avgPerFrame    = std::accumulate(timePerFrames.begin(),   timePerFrames.end(), 0.0)     / timePerFrames.size();
        double avgPerKeyPoint = std::accumulate(timePerKeyPoint.begin(), timePerKeyPoint.end(), 0.0) / timePerKeyPoint.size();
        
        str << "\"" << alg->first << "\"" << "\t"
            << avgPerFrame << "\t"
            << avgPerKeyPoint << std::endl;
    }
    
    return str;
}
