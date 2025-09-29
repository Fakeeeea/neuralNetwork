#include "TrainingData.h"

#include <random>

TrainingData::TrainingData(Eigen::VectorXd input, Eigen::VectorXd target) : input(std::move(input)) , target(std::move(target)) {}

TrainingData::TrainingData(const unsigned char* start, size_t size, unsigned char target) {
    this->input = Eigen::Map<const Eigen::Matrix<unsigned char, Eigen::Dynamic, 1>>(start, size).cast<double>() / 255.0;
    this->target = Eigen::VectorXd::Zero(10);
    this->target[target] = 1.0;
}

int TrainingData::reverseInt(int i) {
    unsigned char c1 = i & 0xFF;
    unsigned char c2 = (i >> 8) & 0xFF;
    unsigned char c3 = (i >> 16) & 0xFF;
    unsigned char c4 = (i >> 24) & 0xFF;

    return ((int)c1 << 24) | ((int)c2 << 16) | ((int)c3 << 8) | c4;
}

std::vector<unsigned char> TrainingData::readMNISTimages(std::string filePath, int& imageNumber, int& nRows, int &nCols) {
    std::ifstream file(filePath, std::ios::binary);

    if(!file.is_open())
        throw std::runtime_error("Could not open MNIST images file");

    int magicNumber = 0;

    file.read((char*)&magicNumber, sizeof(magicNumber));
    magicNumber = reverseInt(magicNumber);

    if(magicNumber != 2051)
        throw std::runtime_error("Invalid MNIST images file");

    file.read((char*)&imageNumber, sizeof(imageNumber));
    file.read((char*)&nRows, sizeof(nRows));
    file.read((char*)&nCols, sizeof(nCols));

    imageNumber = reverseInt(imageNumber), nRows = reverseInt(nRows), nCols = reverseInt(nCols);

    int imageSize = nRows * nCols;

    std::vector<unsigned char> images(imageNumber * imageSize);

    file.read(reinterpret_cast<char*>(images.data()), images.size());

    return images;
}

std::vector<unsigned char> TrainingData::readMNISTlabels(std::string filePath) {
    std::ifstream file(filePath, std::ios::binary);

    if(!file.is_open())
        throw std::runtime_error("Could not open MNIST labels file");

    int magicNumber = 0;
    int labelsNumber = 0;

    file.read((char*)&magicNumber, sizeof(magicNumber));
    magicNumber = reverseInt(magicNumber);

    if(magicNumber != 2049)
        throw std::runtime_error("Invalid MNIST images file");

    file.read((char*)&labelsNumber, sizeof(labelsNumber));
    labelsNumber = reverseInt(labelsNumber);

    std::vector<unsigned char> labels(labelsNumber);
    file.read((char*)labels.data(), labelsNumber);

    return labels;
}

std::vector<TrainingData> TrainingData::getMNISTdata(std::string imagesPath, std::string labelsPath) {
    int nImages, nRows, nCols;
    std::vector<unsigned char> flatImages = readMNISTimages(imagesPath, nImages, nRows, nCols);
    std::vector<unsigned char> labels = readMNISTlabels(labelsPath);

    int nLabels = labels.size();

    if(nImages != nLabels)
        throw std::invalid_argument("Images count does not match labels count");

    int imageSize = nRows * nCols;
    std::vector<TrainingData> data;
    data.reserve(nImages);

    for(int i = 0; i < nImages; ++i) {
        const unsigned char* imageStart = flatImages.data() + i * imageSize;
        data.emplace_back(imageStart, imageSize, labels[i]);
    }

    return data;
}

std::vector<TrainingData> TrainingData::getTextData(std::string dataPath, std::string labelsPath) {

    std::vector<TrainingData> data;
    std::ifstream dataFile(dataPath);
    std::ifstream labelsFile(labelsPath);

    if (!dataFile || !labelsFile) {
        throw std::runtime_error("Could not open data or labels file");
    }

    int nRows, nCols, nData, nLabels, nOutputs;

    dataFile >> nRows >> nCols >> nData;
    labelsFile >> nLabels >> nOutputs;
    if(nData != nLabels) {
        throw std::invalid_argument("Data count does not match labels count");
    }

    data.reserve(nData);


    int dataSize = nCols*nRows;

    for(int i = 0; i < nData; ++i) {
        Eigen::VectorXd singleData(dataSize);
        Eigen::VectorXd singleLabel(nOutputs);

        for(int d = 0; d < dataSize; ++d) {
            dataFile >> singleData[d];
        }

        for(int l = 0; l < nOutputs; ++l) {
            labelsFile >> singleLabel[l];
        }
        data.emplace_back(std::move(singleData), std::move(singleLabel));
    }
    return data;
}

std::vector<TrainingData> TrainingData::generateXorData(int dataSize) {
    std::vector<TrainingData> data;

    std::random_device rg;
    std::default_random_engine rng = std::default_random_engine{ rg() };
    std::normal_distribution<double> noise(0, 0.05);

    int pointsPerQuadrant = (dataSize / 4);
    data.reserve(pointsPerQuadrant*4);

    for(int i = 0; i < 4; ++i) {
        float x,y;
        switch(i) {
            case 0: x=0.25; y=0.25; break;
            case 1: x=0.75; y=0.25; break;
            case 2: x=0.75; y=0.75; break;
            case 3: default: x=0.25; y=0.75; break;
        }

        Eigen::VectorXd label(2);
        label.setZero();

        int bitX = x > 0.5;
        int bitY = y > 0.5;

        label[bitX^bitY] = 1;

        for(int j = 0; j < pointsPerQuadrant; ++j) {
            Eigen::VectorXd point(2);
            point[0] = x + noise(rng), point[1] = y + noise(rng);
            data.emplace_back(point,label);
        }
    }
    return data;
}

std::vector<TrainingData> TrainingData::generateSpiralData(int dataSize) {
    std::vector<TrainingData> data;

    std::random_device rg;
    std::default_random_engine rng = std::default_random_engine { rg() };
    std::uniform_real_distribution<double> dist(0.0, 2*M_PI);
    std::uniform_real_distribution<double> noise(0.0, 0.05);

    int pointsPerClass = dataSize * 0.5;
    data.reserve(pointsPerClass*2);

    for(int i = 0; i < 2; ++i) {

        Eigen::VectorXd label(2);
        label.setZero();
        label[i] = 1;

        for(int j = 0; j < pointsPerClass; ++j) {

            Eigen::VectorXd input(2);
            double t = dist( rng );

            input[0] = ((t * std::sin(t + M_PI*i)) / (2 * M_PI) + 1 + noise(rng)) * 0.45;
            input[1] = ((t * std::cos(t + M_PI*i)) / (2 * M_PI) + 1 + noise(rng)) * 0.45;

            data.emplace_back(input, label);
        }
    }
    return data;
}

std::vector<TrainingData> TrainingData::generateLinearData(int dataSize) {
    std::vector<TrainingData> data;

    std::random_device rg;
    std::default_random_engine rng = std::default_random_engine { rg() };
    std::uniform_real_distribution <double> dist(0, 1.0);

    int pointsPerClass = dataSize / 2;
    data.reserve(pointsPerClass*2);

    for(int i = 0; i < dataSize; ++i) {

        Eigen::VectorXd label(2);
        label.setZero();

        Eigen::VectorXd input(2);

        input[0] = dist(rng);
        input[1] = dist(rng);

        label[(input[1] > input[0]) ? 1 : 0] = 1;

        data.emplace_back(input,label);
    }

    return data;
}

std::vector<TrainingData> TrainingData::generateCircleData(int dataSize) {
    std::vector<TrainingData> data;

    std::random_device rg;
    std::default_random_engine rng = std::default_random_engine { rg() };
    std::uniform_real_distribution<double> noise(-0.02, 0.02);

    int pointsPerClass = dataSize * 0.5;
    data.reserve(pointsPerClass*2);

    Eigen::Vector2d center = {0.5,0.5};
    double radius = 0.2;

    for(int i = 0; i < 2; ++i) {
        Eigen::VectorXd label(2);
        label.setZero(), label[i] = 1;

        radius = radius * (i+1);

        for(int j = 0; j < pointsPerClass; ++j) {
            Eigen::VectorXd input(2);
            float angle = (2 * M_PI * j) / pointsPerClass;

            input[0] = center[0] + sin(angle) * radius + noise(rng) * radius;
            input[1] = center[1] + cos(angle) * radius + noise(rng) * radius;

            data.emplace_back(input, label);
        }
    }

    return data;
}

std::vector<TrainingData> TrainingData::generateMoonsData(int dataSize) {
    std::vector<TrainingData> data;

    std::random_device rg;
    std::default_random_engine rng = std::default_random_engine { rg() };
    std::uniform_real_distribution<double> dist(0.0, M_PI);
    std::normal_distribution<double> noise(0.0, 0.005);

    int pointsPerClass = dataSize / 2;
    data.reserve(pointsPerClass*2);

    for(int i = 0; i < pointsPerClass; ++i) {
        double t = dist(rng);

        Eigen::VectorXd input(2);
        float x = (cos(t) - 0.5) * 0.33 + 0.5;
        float y = (sin(t) - 0.3) * 0.5 + 0.5;
        input[0] = std::min(1.0, std::max(x + noise(rng), 0.0));
        input[1] = std::min(1.0, std::max(y + noise(rng), 0.0));

        Eigen::VectorXd label(2);
        label[0] = 1; label[1] = 0;

        data.emplace_back(input, label);
    }

    for(int i = 0; i < (dataSize - pointsPerClass); ++i) {
        double t = dist(rng);

        Eigen::VectorXd input(2);
        float x = (cos(t) + 0.5) * 0.33 + 0.5;
        float y = (-sin(t) + 0.3) * 0.5 + 0.5;
        input[0] = std::min(1.0, std::max(x + noise(rng), 0.0));
        input[1] = std::min(1.0, std::max(y + noise(rng), 0.0));

        Eigen::VectorXd label(2);
        label[0] = 0; label[1] = 1;

        data.emplace_back(input, label);
    }

    return data;
}

std::vector<TrainingData> TrainingData::generateCubeData(int dataSize) {
    std::vector<TrainingData> data;

    int pointsPerEdge = dataSize / 12;
    data.reserve(pointsPerEdge * 12);

    std::vector<Eigen::Vector3d> vertices = {
            {0.2,0.2,0.2}, {0.2,0.2,0.8}, {0.2,0.8,0.2}, {0.2,0.8,0.8},
            {0.8,0.2,0.2}, {0.8,0.2,0.8}, {0.8,0.8,0.2}, {0.8,0.8,0.8}
    };

    Eigen::Vector3d center = {0.5, 0.5, 0.5};

    std::vector<std::pair<int,int>> edges = {
            {0,1}, {0,2}, {0,4},
            {1,3}, {1,5},
            {2,3}, {2,6},
            {3,7},
            {4,5}, {4,6},
            {5,7},
            {6,7}
    };

    std::random_device rd;
    std::default_random_engine rng{ rd() };
    std::uniform_real_distribution<double> rotation(M_PI / 16, M_PI_4);
    std::normal_distribution<double> noise(0.0, 0.001);

    double xAngle = rotation(rng);
    double yAngle = rotation(rng);

    Eigen::Matrix3d rotX, rotY;

    rotX << 1, 0, 0,
            0, cos(xAngle), -sin(xAngle),
            0, sin(xAngle), cos(xAngle);

    rotY << cos(yAngle), 0, sin(yAngle),
            0, 1, 0,
            -sin(yAngle), 0, cos(yAngle);

    Eigen::Matrix3d rotationMatrix = rotY * rotX;

    for(auto& v : vertices) {
        v -= center;
        v = rotationMatrix * v;
        v += center;
    }

    for(int i = 0; i < 12; ++i) {

        Eigen::VectorXd label(12);
        label.setZero();
        label[i] = 1;

        Eigen::Vector3d start = vertices[edges[i].first];
        Eigen::Vector3d end = vertices[edges[i].second];

        for(int j = 0; j < pointsPerEdge; ++j) {
            Eigen::VectorXd input(3);

            double t = (double) j / (pointsPerEdge - 1);

            Eigen::Vector3d point = start + t * (end - start);

            input[0] = point[0] + noise(rng);
            input[1] = point[1] + noise(rng);
            input[2] = point[2] + noise(rng);

            data.emplace_back(input, label);
        }
    }
    return data;
}
