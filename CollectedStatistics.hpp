#ifndef CollectedStatistics_hpp
#define CollectedStatistics_hpp

#include <iostream>
#include <vector>
#include <map>
#include <string>

typedef enum 
{
    StatisticsElementPointsCount,
    StatisticsElementPercentOfCorrectMatches,
    StatisticsElementPercentOfMatches,
    StatisticsElementMeanDistance,
    StatisticsElementMatchingRatio,
    StatisticsElementHomographyError,
    StatisticsElementPatternLocalization,

} StatisticElement;

struct FrameMatchingStatistics
{
    FrameMatchingStatistics();

    int totalKeypoints;

    float argumentValue;
    float percentOfMatches;
    float ratioTestFalseLevel;
    float meanDistance;
    float stdDevDistance;
    float correctMatchesPercent;
    float homographyError;

    double consumedTimeMs;
    bool   isValid;

    std::ostream& writeElement(std::ostream& str, StatisticElement elem) const;
};

typedef std::vector<FrameMatchingStatistics> SingleRunStatistics;

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
    typedef std::map<std::string, const SingleRunStatistics*> InnerGroup;
    typedef std::map<std::string, InnerGroup>                 OuterGroup;
    typedef std::map<std::string, GroupedByArgument>          OuterGroupLine;


    SingleRunStatistics& getStatistics(std::string algorithmName, std::string transformationName);

    OuterGroup groupByAlgorithmThenByTransformation() const;
    OuterGroupLine groupByTransformationThenByAlgorithm() const;

    std::ostream& printPerformanceStatistics(std::ostream& str) const;
    std::ostream& printStatistics(std::ostream& str, StatisticElement elem) const;

private:
    typedef std::pair<std::string, std::string> Key;

    std::map<Key, SingleRunStatistics> m_allStats;
};

#endif
