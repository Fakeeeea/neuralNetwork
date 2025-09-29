#ifndef NEURAL_NETWORK_NEURALNETWORK_H
#define NEURAL_NETWORK_NEURALNETWORK_H

#include "Cost.h"
#include "ActivationFunction.h"
#include "TrainingData.h"
#include <random>
#include "Eigen/Dense"

class NeuralNetwork{
private:

    std::vector<Eigen::MatrixXd> _tempNablaWeights;
    std::vector<Eigen::VectorXd> _tempNablaBias;

    //Not batched
    std::vector<Eigen::VectorXd> _zs;
    std::vector<Eigen::VectorXd> _activations;

    //Batched
    Eigen::MatrixXd _inputBatch;
    Eigen::MatrixXd _targetBatch;
    std::vector<Eigen::MatrixXd> _activationsBatch;
    std::vector<Eigen::MatrixXd> _zsBatch;

    int numLayers = 0;
    int numConnections = 0;
    bool regularize = true;
    double regularizationFactor = 5;
    double learningRate = 0.5;

    std::default_random_engine rng;

    ActivationFunction act;
    Cost cost;

    std::vector<ActivationFunction> fn;

    std::vector<Eigen::MatrixXd> nablaWeights;
    std::vector<Eigen::VectorXd> nablaBias;

public:

    std::vector<int> layerSizes = {0,0,0};

    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::VectorXd> biases;

    NeuralNetwork();

    explicit NeuralNetwork(std::vector<int> layerSizes, bool regularize = false, double regularizationFactor = 5, double learningRate = 0.5, int costType = 1);

    void initialize();

    void stochasticGradientDescent(std::vector<TrainingData> trainingData, int miniBatchesSize, int epochs);

    std::vector<std::vector<TrainingData>> shuffleIntoTrainingBatches(std::vector<TrainingData> data, int miniBatchesSize);

     void allocateTrainingVariables(bool batched, int miniBatchesSize = 0);

    void trainNetwork(const std::vector<TrainingData>& trainingBatch, int dataSetSize);
    void trainNetworkBatch(const std::vector<TrainingData>& trainingBatch, int dataSetSize);

    void backpropagate(const TrainingData& trainingData);
    void backpropagateBatch(const std::vector<TrainingData>& trainingData);

    Eigen::VectorXd feedForward(Eigen::VectorXd input);
    Eigen::MatrixXd batchFeedForward(Eigen::MatrixXd inputs);

    int testAccuracy(const std::vector<TrainingData>& data);
    int testAccuracy(const std::vector<TrainingData>& data, const Eigen::MatrixXd& outputs);
    double calculateCost(const std::vector<TrainingData>& data);
    double calculateCost(const std::vector<TrainingData>& data, const Eigen::MatrixXd& outputs);

    void setLayerSizes(std::vector<int> newLayerSizes);
    void setRegularizationFactor(double newFactor);
    void setRegularize(bool newRegularize);
    void setLearningRate(double newLearningRate);
    void setCost(int newCostType);
};


#endif //NEURAL_NETWORK_NEURALNETWORK_H
