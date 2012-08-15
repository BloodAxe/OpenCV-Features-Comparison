
#include "ImageTransformation.hpp"
#include "AlgorithmEstimation.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

int main(int argc, const char* argv[])
{
    std::vector<FeatureAlgorithm>              algorithms;
    std::vector<cv::Ptr<ImageTransformation> > transformations;
    
    algorithms.push_back(FeatureAlgorithm("ORB - 2",
                                          new cv::ORB(),
                                          new cv::ORB(),
                                          new cv::BFMatcher(cv::NORM_HAMMING, false)));
    
    algorithms.push_back(FeatureAlgorithm("ORB - 3",
                                          new cv::ORB(500,1.2, 8,31,0, 3),
                                          new cv::ORB(500,1.2, 8,31,0, 3),
                                          new cv::BFMatcher(cv::NORM_HAMMING2, false)));

    algorithms.push_back(FeatureAlgorithm("ORB - 4",
                                          new cv::ORB(500,1.2, 8,31,0, 3),
                                          new cv::ORB(500,1.2, 8,31,0, 3),
                                          new cv::BFMatcher(cv::NORM_HAMMING2, false)));
    
    /*
    algorithms.push_back(FeatureAlgorithm("ORB",
                                          new cv::SurfFeatureDetector(2000,4),
                                          new cv::FREAK(),
                                          new cv::BFMatcher(cv::NORM_HAMMING, false)));
    */
    
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
        
        if (testImage.empty())
        {
            std::cout << "Cannot read image from " << testImagePath << std::endl;
        }
        
        std::cout << "[" << testImagePath << "]" << std::endl;
        
        for (int transformIndex = 0; transformIndex < transformations.size(); transformIndex++)
        {
            const ImageTransformation& trans = *transformations[transformIndex].obj;
            std::cout << "Transformation:" << trans.name << std::endl;

            for (size_t algIndex = 0; algIndex < algorithms.size(); algIndex++)
            {
                const FeatureAlgorithm& alg      = algorithms[algIndex];
                std::cout << "Algorithm:" << alg.name << std::endl;
                
                std::vector<FrameMatchingStatistics> stat;
                
                performEstimation(alg, trans, testImage.clone(), stat);
                std::cout << stat << std::endl;
            }
        }
    }
    return 0;
}