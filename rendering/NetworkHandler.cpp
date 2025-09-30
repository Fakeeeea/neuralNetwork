#include "NetworkHandler.h"

NetworkHandler::NetworkHandler() {
    nn = NeuralNetwork({2,3,2});
    status._networkCreated = true;
}

NetworkHandler::NetworkHandler(std::vector<int> startingSizes) {
    nn = NeuralNetwork(std::move(startingSizes));
    status._networkCreated = true;
}

void NetworkHandler::update(bool& updateGraphs) {
    if(status._training && session.imagesLoaded)
        updateGraphs = trainNetwork();
}

bool NetworkHandler::trainNetwork() {

    bool updateGraphs = false;
    if(session.miniBatches.empty())
        session.miniBatches = nn.shuffleIntoTrainingBatches(session.trainData, _miniBatchesSize), updateGraphs = true;
    if (session.epoch >= _epochsNum) {
        status._training = false;
    } else {
        if (session.minibatch >= session.miniBatches.size()) {

            updateGraphs = true;

            session.miniBatches = nn.shuffleIntoTrainingBatches(session.trainData, _miniBatchesSize);

            session.minibatch = 0;
            session.epoch++;
        } else {
            //nn.trainNetwork(session.miniBatches[session.minibatch], session.trainData.size());
            nn.trainNetworkBatch(session.miniBatches[session.minibatch], session.trainData.size());
            session.minibatch++;
        }
    }

    return updateGraphs;
}

void NetworkHandler::loadData(std::string trainImagesPath, std::string trainLabelsPath, std::string testImagesPath, std::string testLabelsPath, double validationAmount, bool MNIST) {
    resetSession();

    session.testData =  MNIST ? TrainingData::getMNISTdata(std::move(testImagesPath), std::move(testLabelsPath)) :
                                TrainingData::getTextData(std::move(testImagesPath), std::move(testLabelsPath));

    session.trainData = MNIST ? TrainingData::getMNISTdata(std::move(trainImagesPath), std::move(trainLabelsPath)) :
                                TrainingData::getTextData(std::move(trainImagesPath), std::move(trainLabelsPath));

    size_t trainSize = session.trainData.size();
    size_t validationSize = trainSize * validationAmount;

    session.validationData.reserve(validationSize);

    session.validationData.insert(session.validationData.end(),
                                  std::make_move_iterator(session.trainData.end() - validationSize),
                                  std::make_move_iterator(session.trainData.end()));

    session.trainData.erase(session.trainData.end() - validationSize,
                            session.trainData.end());

    session.imagesLoaded = true;
}

void NetworkHandler::loadData(int type) {
    resetSession();

    std::vector<TrainingData> fullData;

    const float validationAmount = 0.1, testAmount = 0.2;

    switch(type) {
        case XOR: fullData = TrainingData::generateXorData(100 + 100 * (validationAmount + testAmount)); break;
        case SPIRAL: fullData = TrainingData::generateSpiralData(400 + 400 * (validationAmount + testAmount)); break;
        case LINEAR: fullData = TrainingData::generateLinearData(400 + 400 * (validationAmount + testAmount)); break;
        case CIRCLE: fullData = TrainingData::generateCircleData(400 + 400 * (validationAmount + testAmount)); break;
        case MOONS: fullData = TrainingData::generateMoonsData(400 + 400 * (validationAmount + testAmount)); break;
        case CUBE: fullData = TrainingData::generateCubeData(1000 + 1000 * (validationAmount + testAmount)); break;
        case PYRAMID: fullData = TrainingData::generatePyramidData(1000 + 1000 * (validationAmount + testAmount)); break;
        default: fullData = TrainingData::generateXorData(400 + 400 * (validationAmount + testAmount)); break;
    }

    size_t totalSize = fullData.size();
    size_t testSize = totalSize * testAmount;
    size_t validationSize = totalSize * validationAmount;

    std::shuffle(fullData.begin(), fullData.end(), std::default_random_engine{std::random_device{}()});

    session.testData.insert(session.testData.end(),
                            std::make_move_iterator(fullData.end() - testSize),
                            std::make_move_iterator(fullData.end()));
    fullData.erase(fullData.end() - testSize, fullData.end());

    session.validationData.insert(session.validationData.end(),
                                  std::make_move_iterator(fullData.end() - validationSize),
                                  std::make_move_iterator(fullData.end()));
    fullData.erase(fullData.end() - validationSize, fullData.end());

    session.trainData = std::move(fullData);

    session.imagesLoaded = true;
}

void NetworkHandler::createNetwork(std::vector<int> layerSizes, int costType, double regularizationFactor, double learningRate, bool regularize) {
    nn = NeuralNetwork(std::move(layerSizes), regularize, regularizationFactor, learningRate, costType);
    nn.allocateTrainingVariables(true, _miniBatchesSize);
    status._networkCreated = true;

    testOutputs.resize(0, 0);
    trainOutputs.resize(0, 0);
    validationOutputs.resize(0, 0);
}

void NetworkHandler::updateSettings(int costType, double regularizationFactor, double learningRate, bool regularize) {
    if(isNetworkCreated()) {
        nn.setRegularizationFactor(regularizationFactor);
        nn.setCost(costType);
        nn.setLearningRate(learningRate);
        nn.setRegularize(regularize);
    }
}

void NetworkHandler::updateTrainingSessionSettings(int miniBatchesS, int epochN) {
    _epochsNum = epochN;
    _miniBatchesSize = miniBatchesS;

    nn.allocateTrainingVariables(true, _miniBatchesSize);

    session.epoch = session.minibatch = 0;
    session.miniBatches.clear();
}

bool NetworkHandler::isNetworkCreated() const {
    return status._networkCreated;
}

void NetworkHandler::toggleTraining() {
    if(isNetworkCreated())
        status._training = !status._training;
}

void NetworkHandler::updateOutputs() {

    const auto getOutput = [&](const std::vector<TrainingData>& data) {
        const int dataSize = data.size();

        Eigen::MatrixXd inputs(nn.layerSizes[0], dataSize);

        for(int i = 0; i < dataSize; ++i) {
            inputs.col(i) = data[i].input;
        }

        return nn.batchFeedForward(inputs);
    };

    trainOutputs = getOutput(session.trainData);
    testOutputs = getOutput(session.testData);
    validationOutputs = getOutput(session.validationData);
}

double NetworkHandler::getDatasetAccuracy(NetworkHandler::DataType type) {
    int correct, expected;
    switch(type) {
        case TRAIN:
            expected = session.trainData.size(), correct = nn.testAccuracy(session.trainData, trainOutputs);
            break;
        case TEST:
            expected = session.testData.size(), correct = nn.testAccuracy(session.testData, testOutputs);
            break;
        case VAL:
        default:
            expected = session.validationData.size(), correct = nn.testAccuracy(session.validationData, validationOutputs);
    }

    return ( (double) correct / expected) * 100;
}

double NetworkHandler::getDatasetCost(NetworkHandler::DataType type) {
    switch(type) {
        case TRAIN:
            return nn.calculateCost(session.trainData, trainOutputs);
        case TEST:
            return nn.calculateCost(session.testData, testOutputs);
        case VAL:
        default:
            return nn.calculateCost(session.validationData, validationOutputs);
    }
}

void NetworkHandler::setLayerSizes(std::vector<int> newLayerSizes) {
    nn.setLayerSizes(newLayerSizes);
    nn.allocateTrainingVariables(true, _miniBatchesSize);
}

void NetworkHandler::resetSession() {
    session.trainData.clear();
    session.testData.clear();
    session.validationData.clear();
    session.miniBatches.clear();
    session.minibatch = 0;
    session.epoch = 0;
}
