#ifndef NEURAL_NETWORK_COST_H
#define NEURAL_NETWORK_COST_H

#include "TrainingData.h"
#include "ActivationFunction.h"
#include "Eigen/Dense"
#include <vector>

class Cost {
public:

    enum CostType {
        QUADRATIC,
        CROSSENTROPY
    };

    Cost();
    Cost(CostType type);

    double cost(const Eigen::VectorXd& out, const Eigen::VectorXd& target);
    Eigen::VectorXd delta(const Eigen::VectorXd& out, const Eigen::VectorXd& target, const Eigen::VectorXd& zs, ActivationFunction& fn);
    Eigen::MatrixXd deltaBatch(const Eigen::MatrixXd& out, const Eigen::MatrixXd& target, const Eigen::MatrixXd& zs, ActivationFunction& fn);
    double totalCost(const std::vector<Eigen::VectorXd>& out, const std::vector<Eigen::VectorXd>& target);

private:

    CostType costType;

    double totalCostQuadratic(const std::vector<Eigen::VectorXd>& out, const std::vector<Eigen::VectorXd>& target);
    double quadratic(const Eigen::VectorXd& out, const Eigen::VectorXd& target);
    Eigen::VectorXd quadraticDelta(const Eigen::VectorXd& out, const Eigen::VectorXd& target, const Eigen::VectorXd& zs, ActivationFunction& fn);
    Eigen::MatrixXd quadraticDelta(const Eigen::MatrixXd& out, const Eigen::MatrixXd& target, const Eigen::MatrixXd& zs, ActivationFunction& fn);

    inline Eigen::ArrayXd nanToNum(const Eigen::ArrayXd& x);
    double totalCostCrossEntropy(const std::vector<Eigen::VectorXd>& out, const std::vector<Eigen::VectorXd>& target);
    double crossEntropy(const Eigen::VectorXd& out, const Eigen::VectorXd& target);
    Eigen::VectorXd crossEntropyDelta(const Eigen::VectorXd& out, const Eigen::VectorXd& target);
    Eigen::MatrixXd crossEntropyDelta(const Eigen::MatrixXd& out, const Eigen::MatrixXd& target);
};


#endif //NEURAL_NETWORK_COST_H
