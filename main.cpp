
#include "ImageTransformation.hpp"
#include "AlgorithmEstimation.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <algorithm>
#include <numeric>


int main(int argc, const char* argv[])
{
    // Print OpenCV build info:
    // std::cout << cv::getBuildInformation() << std::endl;
    
    std::vector<FeatureAlgorithm>              algorithms;
    std::vector<cv::Ptr<ImageTransformation> > transformations;
    
    // Initialize list of algorithm tuples:

    algorithms.push_back(FeatureAlgorithm("SURF-FREAK",
                                          new cv::SurfFeatureDetector(),
                                          new cv::FREAK(),
                                          new cv::BFMatcher(cv::NORM_HAMMING)));

    algorithms.push_back(FeatureAlgorithm("ORB-FREAK",
                                          new cv::OrbFeatureDetector(),
                                          new cv::FREAK(),
                                          new cv::BFMatcher(cv::NORM_HAMMING)));
    
    algorithms.push_back(FeatureAlgorithm("ORB - 2",
                                          new cv::ORB(),
                                          new cv::ORB(),
                                          new cv::BFMatcher(cv::NORM_HAMMING, false)));
    
    algorithms.push_back(FeatureAlgorithm("ORB - 3",
                                          new cv::ORB(500, 1.2f, 8,31, 0, 3),
                                          new cv::ORB(500, 1.2f, 8,31, 0, 3),
                                          new cv::BFMatcher(cv::NORM_HAMMING2, false)));

    algorithms.push_back(FeatureAlgorithm("ORB - 4",
                                          new cv::ORB(500, 1.2f, 8, 31, 0, 4),
                                          new cv::ORB(500, 1.2f, 8, 31, 0, 4),
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
        
        //std::cout << "[" << testImagePath << "]" << std::endl;
        
        for (size_t algIndex = 0; algIndex < algorithms.size(); algIndex++)
        {
            const FeatureAlgorithm& alg   = algorithms[algIndex];

            for (size_t transformIndex = 0; transformIndex < transformations.size(); transformIndex++)
            {
                const ImageTransformation& trans = *transformations[transformIndex].obj;

                performEstimation(alg, trans, testImage.clone(), fullStat.getStatistics(alg.name, trans.name));
            }
        }
        
        fullStat.printPerformanceStatistics(std::cout);
        fullStat.printStatistics(std::cout, StatisticsElementPercentOfCorrectMatches);
        fullStat.printStatistics(std::cout, StatisticsElementMatchingRatio);
        fullStat.printStatistics(std::cout, StatisticsElementMeanDistance);
    }
    
    return 0;
}

