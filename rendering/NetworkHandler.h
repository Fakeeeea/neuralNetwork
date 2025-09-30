//
// Created by aaa on 14/09/2025.
//

#ifndef NEURAL_NETWORK_NETWORKHANDLER_H
#define NEURAL_NETWORK_NETWORKHANDLER_H

#include "../NeuralNetwork.h"

class NetworkHandler {
private:

    struct TrainingSession {
        bool imagesLoaded = false;

        std::vector<TrainingData> trainData;
        std::vector<TrainingData> testData;
        std::vector<TrainingData> validationData;
        std::vector<std::vector<TrainingData>> miniBatches;

        int epoch = 0;
        int minibatch = 0;
    };

    struct NetworkStatus {
        bool _networkCreated = false;
        bool _training = false;
    };

    int _epochsNum = 0;
    int _miniBatchesSize = 0;
    int _trainingStepsPerFrame = 1;

public:

    enum NetworkType {
        //MNIST,
        XOR,
        SPIRAL,
        LINEAR,
        CIRCLE,
        MOONS,
        CUBE,
        PYRAMID
    };

    enum DataType {
        TRAIN,
        TEST,
        VAL
    };

    TrainingSession session;
    NetworkStatus status;
    NeuralNetwork nn;

    Eigen::MatrixXd testOutputs;
    Eigen::MatrixXd trainOutputs;
    Eigen::MatrixXd validationOutputs;

    NetworkHandler();
    NetworkHandler(std::vector<int> startingSizes);

    void update(bool& updateGraphs);
    bool trainNetwork();
    void loadData(std::string trainImagesPath, std::string trainLabelsPath, std::string testImagesPath, std::string testLabelsPath, double validationPercentage, bool MNIST);
    void loadData(int type);
    void createNetwork(std::vector<int> layerSizes, int costType, double regularizationFactor, double learningRate, bool regularize);
    void setLayerSizes(std::vector<int> newLayerSizes);
    void updateSettings(int costType, double regularizationFactor, double learningRate, bool regularize);
    void updateTrainingSessionSettings(int miniBatchesS, int epochN);
    void toggleTraining();
    void updateOutputs();
    void resetSession();
    double getDatasetAccuracy(NetworkHandler::DataType type);
    double getDatasetCost(NetworkHandler::DataType type);

    [[nodiscard]] bool isNetworkCreated() const;

};


#endif //NEURAL_NETWORK_NETWORKHANDLER_H
