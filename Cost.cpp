#include "Cost.h"

Cost::Cost() {
    costType = CostType::CROSSENTROPY;
}

Cost::Cost(CostType type) : costType(type) {};

double Cost::cost(const Eigen::VectorXd& out, const Eigen::VectorXd& target) {
    switch(costType) {
        case QUADRATIC: return quadratic(out, target);
        case CROSSENTROPY: return crossEntropy(out, target);
    }
    return crossEntropy(out, target);
}

Eigen::VectorXd Cost::delta(const Eigen::VectorXd& out, const Eigen::VectorXd& target, const Eigen::VectorXd& zs, ActivationFunction& fn) {
    switch(costType) {
        case QUADRATIC: return quadraticDelta(out, target, zs, fn);
        case CROSSENTROPY: return crossEntropyDelta(out, target);
    }
    return crossEntropyDelta(out, target);
}

Eigen::MatrixXd Cost::deltaBatch(const Eigen::MatrixXd& out, const Eigen::MatrixXd& target, const Eigen::MatrixXd& zs, ActivationFunction& fn) {

    switch(costType) {
        case QUADRATIC: return quadraticDelta(out, target, zs, fn);
        case CROSSENTROPY: return crossEntropyDelta(out, target);
    }
    return crossEntropyDelta(out, target);
}

double Cost::totalCost(const std::vector<Eigen::VectorXd>& out, const std::vector<Eigen::VectorXd>& target) {
    switch(costType) {
        case QUADRATIC: return totalCostQuadratic(out, target);
        case CROSSENTROPY: return totalCostCrossEntropy(out, target);
    }
    return totalCostCrossEntropy(out, target);
}

double Cost::quadratic(const Eigen::VectorXd& out, const Eigen::VectorXd& target) {
    return (out-target).squaredNorm();
}

Eigen::VectorXd Cost::quadraticDelta(const Eigen::VectorXd& out, const Eigen::VectorXd& target, const Eigen::VectorXd& zs, ActivationFunction& fn) {
    return (out - target).cwiseProduct(fn.derivative(zs));
}

Eigen::MatrixXd Cost::quadraticDelta(const Eigen::MatrixXd& out, const Eigen::MatrixXd& target, const Eigen::MatrixXd& zs, ActivationFunction& fn) {
    return (out-target).cwiseProduct(fn.derivative(zs));
}

inline Eigen::ArrayXd Cost::nanToNum(const Eigen::ArrayXd& x) {
    Eigen::ArrayXd out = x;
    out = out.isNaN().select(0.0, out);
    out = out.isInf().select(0.0, out);
    return out;
}

double Cost::crossEntropy(const Eigen::VectorXd& out, const Eigen::VectorXd& target) {
    Eigen::ArrayXd a = out.array();
    Eigen::ArrayXd y = target.array();
    Eigen::ArrayXd loss = -y * (a.log()) - (1.0 - y) * ((1.0 - a).log());
    loss = Cost::nanToNum(loss);
    return loss.sum();
}

Eigen::VectorXd Cost::crossEntropyDelta(const Eigen::VectorXd& out, const Eigen::VectorXd& target) {
    return out - target;
}

Eigen::MatrixXd Cost::crossEntropyDelta(const Eigen::MatrixXd& out, const Eigen::MatrixXd& target) {
    return out - target;
}

double Cost::totalCostQuadratic(const std::vector<Eigen::VectorXd>& out, const std::vector<Eigen::VectorXd>& target) {
    double cost = 0;
    int nOut = out.size();

    for(int i = 0; i < nOut; ++i) {
        cost += quadratic(out[i], target[i]);
    }
    cost *= 1./ (2*nOut);

    return cost;
}

double Cost::totalCostCrossEntropy(const std::vector<Eigen::VectorXd>& out, const std::vector<Eigen::VectorXd>& target) {
    double cost = 0;
    int nOut = out.size();

    for(int i = 0; i < nOut; ++i) {
        cost += crossEntropy(out[i], target[i]);
    }

    cost *= 1./nOut;
    return cost;
}