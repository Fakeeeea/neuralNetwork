#include "NeuralNetwork.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <utility>
#include "Eigen/Dense"

NeuralNetwork::NeuralNetwork() {};

NeuralNetwork::NeuralNetwork(std::vector<int> layerSizes, bool regularize, double regularizationFactor, double learningRate, int costType) : layerSizes(layerSizes), act(ActivationFunction::Sigmoid), cost((Cost::CostType)costType) {

    this->regularize = regularize;
    this->regularizationFactor = regularizationFactor;
    this->learningRate = learningRate;

    std::random_device rd;
    rng = std::default_random_engine { rd() };

    if(layerSizes.size() < 2) {
        throw std::invalid_argument("The neural network cant have less than an input and output layer.");
    }

    initialize();
}

void NeuralNetwork::initialize() {

    numLayers = layerSizes.size();
    numConnections = numLayers - 1;

    weights.resize(numConnections);
    biases.resize(numConnections);
    fn.resize(numConnections);

    for(int i = 0; i < numConnections; i++) {

        double limit = sqrt(2.0f / (layerSizes[i] + layerSizes[i+1]));
        std::normal_distribution <double> distBiases(0,0.001);
        std::uniform_real_distribution<double> distWeights(-limit, limit);

        weights[i] = Eigen::MatrixXd(layerSizes[i+1], layerSizes[i]);
        biases[i] = Eigen::VectorXd(layerSizes[i+1]);

        for(int j = 0; j < layerSizes[i+1]; j++) {
            for(int k = 0; k < layerSizes[i]; k++) {
                weights[i](j,k) = distWeights( rng );
            }
        }

        for(int j = 0; j < layerSizes[i+1]; j++) {
            biases[i][j] = distBiases( rng );
        }

        if(i != numConnections-1)
            fn[i] = ActivationFunction(ActivationFunction::ActivationType::Tanh);
    }
}

Eigen::VectorXd NeuralNetwork::feedForward(Eigen::VectorXd input) {

    if(input.size() != layerSizes[0]) {
        std::cerr << "Warning: input size is different from network expected input (" << input.size() << " ! = " << layerSizes[0]<< ")\n";
    }


    for(int i = 0; i < numConnections; i++) {
        Eigen::VectorXd z = (weights[i] * input) + biases[i];
        input = fn[i](z);
    }

    return input;
}


Eigen::MatrixXd NeuralNetwork::batchFeedForward(Eigen::MatrixXd inputs) {

    for (int i = 0; i < numConnections; ++i) {
        Eigen::MatrixXd batchZs = (weights[i] * inputs).colwise() + biases[i];
        inputs = fn[i](batchZs);
    }

    return inputs;
}

void NeuralNetwork::stochasticGradientDescent(std::vector<TrainingData> trainingData, int miniBatchesSize, int epochs) {

    int trainingDataNumber = trainingData.size();

    allocateTrainingVariables(false);

    for(int i = 0; i < epochs; ++i) {
        std::vector<std::vector<TrainingData>> miniBatches = shuffleIntoTrainingBatches(trainingData, miniBatchesSize);

        for(auto& miniBatch : miniBatches) {
            trainNetwork(miniBatch, trainingDataNumber);
        }
    }
}

void NeuralNetwork::allocateTrainingVariables(bool batched, int miniBatchesSize) {
    _tempNablaBias.resize(numConnections);
    _tempNablaWeights.resize(numConnections);

    if(!batched) {
        nablaWeights.resize(numConnections);
        nablaBias.resize(numConnections);
        _zs.resize(numConnections);
        _activations.resize(numConnections + 1);
    } else {
        _zsBatch.resize(numConnections);
        _activationsBatch.resize(numConnections + 1);

        _inputBatch.resize(layerSizes[0], miniBatchesSize);
        _targetBatch.resize(layerSizes.back(), miniBatchesSize);
    }

    for(int i = 0; i < numConnections; ++i) {
        _tempNablaWeights[i].resize(layerSizes[i + 1], layerSizes[i]);
        _tempNablaBias[i].resize(layerSizes[i + 1]);

        if(!batched) {
            nablaWeights[i].resize(layerSizes[i + 1], layerSizes[i]);
            nablaBias[i].resize(layerSizes[i + 1]);
            _zs[i].resize(layerSizes[i + 1]);
        } else {
            _zsBatch[i].resize(layerSizes[i + 1], miniBatchesSize);
        }
    }

    for(int i = 0; i < numLayers; ++i) {
        if(!batched)
            _activations[i].resize(layerSizes[i]);
        else
            _activationsBatch[i].resize(layerSizes[i], miniBatchesSize);
    }
}

std::vector<std::vector<TrainingData>> NeuralNetwork::shuffleIntoTrainingBatches(std::vector<TrainingData> data, int miniBatchesSize) {
    int dataSize = data.size();
    int miniBatchesNumber = (dataSize + miniBatchesSize - 1) / miniBatchesSize;

    std::shuffle(data.begin(), data.end(), rng);

    std::vector<std::vector<TrainingData>> miniBatches;
    miniBatches.reserve(miniBatchesNumber);

    for (int i = 0; i < miniBatchesNumber; i++) {
        int start = i * miniBatchesSize;
        int end = std::min(start + miniBatchesSize, dataSize);
        miniBatches.emplace_back(data.begin() + start, data.begin() + end);
    }

    return miniBatches;
}


void NeuralNetwork::trainNetwork(const std::vector<TrainingData>& trainingBatch, int dataSetSize) {

    const double learningRateFactor = learningRate / trainingBatch.size();
    const double regularization = regularize ? ( 1 - (learningRate * regularizationFactor) / dataSetSize) : 0;

    for(int i = 0; i < numConnections; ++i) {
        nablaWeights[i].setZero();
        nablaBias[i].setZero();
    }

    for(const TrainingData& data : trainingBatch) {

        backpropagate(data);

        for(int i = 0; i < numConnections; ++i) {
            nablaBias[i] += _tempNablaBias[i];
            nablaWeights[i] += _tempNablaWeights[i];
        }
    }

    if(regularize) {
        for(int i = 0; i < numConnections; ++i) {
            biases[i] -= learningRateFactor * nablaBias[i];
            weights[i] = regularization * weights[i] - learningRateFactor * nablaWeights[i];
        }
    } else {
        for(int i = 0; i < numConnections; ++i) {
            biases[i] -= learningRateFactor * nablaBias[i];
            weights[i] -= learningRateFactor * nablaWeights[i];
        }
    }
}

void NeuralNetwork::trainNetworkBatch(const std::vector<TrainingData>& trainingBatch, int dataSetSize) {
    const double learningRateFactor = learningRate / trainingBatch.size();
    const double regularization = regularize ? ( 1 - (learningRate * regularizationFactor) / dataSetSize) : 0;

    backpropagateBatch(trainingBatch);

    if(regularize) {
        for(int i = 0; i < numConnections; ++i) {
            biases[i] -= learningRateFactor * _tempNablaBias[i];
            weights[i] = regularization * weights[i] - learningRateFactor * _tempNablaWeights[i];
        }
    } else {
        for(int i = 0; i < numConnections; ++i) {
            biases[i] -= learningRateFactor * _tempNablaBias[i];
            weights[i] -= learningRateFactor * _tempNablaWeights[i];
        }
    }
}

void NeuralNetwork::backpropagate(const TrainingData& trainingData) {

    //feedforward

    Eigen::VectorXd activation = trainingData.input;
    _activations[0] = activation;
    for(int i = 0; i < numConnections; i++) {
        Eigen::VectorXd z = (weights[i] * activation) + biases[i];
        activation = fn[i](z);

        _zs[i] = z;
        _activations[i + 1] = activation;
    }

    //actual backpropagation start
    Eigen::VectorXd target = trainingData.target;
    Eigen::VectorXd sp;

    Eigen::VectorXd delta = cost.delta(_activations.back(), target, _zs.back(), fn.back());

    _tempNablaBias.back() = delta;
    _tempNablaWeights.back() = delta * _activations[numConnections - 1].transpose(); // nablaWeights = a(l-1) (x) delta

    for(int l = 2; l < numLayers; ++l) {
        int layer_idx = numConnections-l;

        sp = fn[layer_idx].derivativeFromOutput(_activations[layer_idx+1]);
        delta = (weights[layer_idx + 1].transpose() * delta);
        delta = delta.cwiseProduct(sp); // (weights(l+1)delta(l+1)) (.) sigmoid'(z(l))

        _tempNablaBias[layer_idx] = delta;
        _tempNablaWeights[layer_idx] = delta * _activations[layer_idx].transpose();
    }
}

void NeuralNetwork::backpropagateBatch(const std::vector<TrainingData>& trainingData) {
    const int batchSize = trainingData.size();
    const double invBatchSize = 1.0 / batchSize;

    for(int i = 0; i < batchSize; ++i) {
        _inputBatch.col(i)  = trainingData[i].input;
        _targetBatch.col(i) = trainingData[i].target;
    }

    const auto inputView  = _inputBatch.leftCols(batchSize);
    const auto targetView = _targetBatch.leftCols(batchSize);

    _activationsBatch[0].leftCols(batchSize) = inputView;

    for(int i = 0; i < numConnections; ++i) {
        const auto& currentActivation = _activationsBatch[i].leftCols(batchSize);

        _zsBatch[i].leftCols(batchSize).noalias() = weights[i] * currentActivation;
        _zsBatch[i].leftCols(batchSize).colwise() += biases[i];
        _activationsBatch[i+1].leftCols(batchSize) = fn[i]( (Eigen::MatrixXd) _zsBatch[i].leftCols(batchSize) );
    }

    Eigen::MatrixXd deltasBatch = cost.deltaBatch(_activationsBatch.back().leftCols(batchSize),
                                  _targetBatch.leftCols(batchSize),
                                  _zsBatch.back().leftCols(batchSize),
                                  fn.back());

    _tempNablaBias.back() = deltasBatch.rowwise().sum() * invBatchSize;
    _tempNablaWeights.back() = (deltasBatch * _activationsBatch[numConnections - 1].leftCols(batchSize).transpose()) * invBatchSize;

    for(int l = 2; l < numLayers; ++l) {
        int layer_idx = numConnections - l;

        auto sp = fn[layer_idx].derivativeFromOutput( (Eigen::MatrixXd) _activationsBatch[layer_idx+1].leftCols(batchSize) );

        deltasBatch = weights[layer_idx+1].transpose() * deltasBatch;
        deltasBatch = deltasBatch.cwiseProduct(sp);

        _tempNablaBias[layer_idx] = deltasBatch.rowwise().sum() * invBatchSize;
        _tempNablaWeights[layer_idx] = (deltasBatch * _activationsBatch[layer_idx].leftCols(batchSize).transpose()) * invBatchSize;
    }
}

int NeuralNetwork::testAccuracy(const std::vector<TrainingData>& data) {
    const int dataSize = data.size();

    Eigen::MatrixXd inputs(layerSizes[0], dataSize);

    for(int i = 0; i < dataSize; ++i) {
        inputs.col(i) = data[i].input;
    }

    Eigen::MatrixXd outputs = batchFeedForward(inputs);

    int correct = 0;

    for(int i = 0; i < outputs.cols(); ++i) {
        Eigen::Index predicted, actual;
        outputs.col(i).maxCoeff(&predicted);
        data[i].target.maxCoeff(&actual);

        if(predicted == actual) {
            correct++;
        }
    }

    return correct;
}

int NeuralNetwork::testAccuracy(const std::vector<TrainingData>& data, const Eigen::MatrixXd& outputs) {
    int correct = 0;

    for(int i = 0; i < outputs.cols(); ++i) {
        Eigen::Index predicted, actual;
        outputs.col(i).maxCoeff(&predicted);
        data[i].target.maxCoeff(&actual);

        if(predicted == actual) {
            correct++;
        }
    }

    return correct;
}

double NeuralNetwork::calculateCost(const std::vector<TrainingData>& data) {
    const int dataSize = data.size();

    Eigen::MatrixXd inputs(layerSizes[0], dataSize);

    for(int i = 0; i < dataSize; ++i) {
        inputs.col(i) = data[i].input;
    }

    Eigen::MatrixXd outputs = batchFeedForward(inputs);

    std::vector<Eigen::VectorXd> out(dataSize);
    std::vector<Eigen::VectorXd> target(dataSize);

    for (int i = 0; i < dataSize; ++i) {
        out[i] = outputs.col(i);
        target[i] = data[i].target;
    }

    return cost.totalCost(out, target);
}

double NeuralNetwork::calculateCost(const std::vector<TrainingData>& data, const Eigen::MatrixXd& outputs) {
    const int dataSize = data.size();

    std::vector<Eigen::VectorXd> out(dataSize);
    std::vector<Eigen::VectorXd> target(dataSize);

    for (int i = 0; i < dataSize; ++i) {
        out[i] = outputs.col(i);
        target[i] = data[i].target;
    }

    return cost.totalCost(out, target);
}

void NeuralNetwork::setLayerSizes(std::vector<int> newLayerSizes) {
    layerSizes = newLayerSizes;
    initialize();
}

void NeuralNetwork::setRegularizationFactor(double newFactor) {
    regularizationFactor = newFactor;
}
void NeuralNetwork::setRegularize(bool newRegularize) {
    regularize = newRegularize;
}
void NeuralNetwork::setLearningRate(double newLearningRate) {
    learningRate = newLearningRate;
}
void NeuralNetwork::setCost(int newCostType) {
    cost = Cost(static_cast<Cost::CostType>(newCostType));
}
