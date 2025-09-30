#ifndef NEURAL_NETWORK_TRAININGDATA_H
#define NEURAL_NETWORK_TRAININGDATA_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "Eigen/Dense"

class TrainingData {
private:
    static int reverseInt(int i);
public:
    Eigen::VectorXd input;
    Eigen::VectorXd target;

    TrainingData(Eigen::VectorXd input, Eigen::VectorXd target);
    TrainingData(const unsigned char* start, size_t size, unsigned char target);

    static std::vector<unsigned char> readMNISTimages(std::string filePath, int& imageNumber, int& nRows, int &nCols);
    static std::vector<unsigned char> readMNISTlabels(std::string labes);
    static std::vector<TrainingData> getMNISTdata(std::string imagesPath, std::string labelsPath);
    static std::vector<TrainingData> getTextData(std::string dataPath, std::string labelsPath);
    static std::vector<TrainingData> generateXorData(int dataSize);
    static std::vector<TrainingData> generateSpiralData(int dataSize);
    static std::vector<TrainingData> generateLinearData(int dataSize);
    static std::vector<TrainingData> generateCircleData(int dataSize);
    static std::vector<TrainingData> generateMoonsData(int dataSize);
    static std::vector<TrainingData> generateCubeData(int dataSize);
    static std::vector<TrainingData> generatePyramidData(int dataSize);
};

#endif //NEURAL_NETWORK_TRAININGDATA_H
