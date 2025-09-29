#include "ActivationFunction.h"
#include <cmath>
#include <iostream>

ActivationFunction::ActivationFunction() : type(ActivationType::Sigmoid) {};
ActivationFunction::ActivationFunction(ActivationFunction::ActivationType f) : type(f) {}

double ActivationFunction::operator()(double z) const {
    switch(type) {
        case Sigmoid: return sigmoid(z);
        case ReLU: return reLU(z);
        case Tanh: return tanhAct(z);
    }
    return sigmoid(z);
}

Eigen::VectorXd ActivationFunction::operator()(Eigen::VectorXd zs) const {
    switch(type) {
        case Sigmoid: return sigmoid(zs);
        case ReLU: return reLU(zs);
        case Tanh: return tanhAct(zs);
    }
    return sigmoid(zs);
}

Eigen::MatrixXd ActivationFunction::operator()(Eigen::MatrixXd zs) const {
    switch(type) {
        case Sigmoid: return sigmoid(zs);
        case ReLU: return reLU(zs);
        case Tanh: return tanhAct(zs);
    }
    return sigmoid(zs);
}

double ActivationFunction::derivative(double z) const {
    switch(type) {
        case Sigmoid: return sigmoidDerivative(z);
        case ReLU: return reLUDerivative(z);
        case Tanh: return tanhDerivative(z);
    }
    return sigmoidDerivative(z);
}

Eigen::VectorXd ActivationFunction::derivative(Eigen::VectorXd zs) const {
    switch(type) {
        case Sigmoid: return sigmoidDerivative(zs);
        case ReLU: return reLUDerivative(zs);
        case Tanh: return tanhDerivative(zs);
    }
    return sigmoidDerivative(zs);
}

Eigen::MatrixXd ActivationFunction::derivative(Eigen::MatrixXd zs) const {
    switch(type) {
        case Sigmoid: return sigmoidDerivative(zs);
        case ReLU: return reLUDerivative(zs);
        case Tanh: return tanhDerivative(zs);
    }
    return sigmoidDerivative(zs);
}

double ActivationFunction::derivativeFromOutput(double z) const {
    switch(type) {
        case Sigmoid: return sigmoidDerivativeFromOutput(z);
        case ReLU: return reLUDerivative(z);
        case Tanh: return tanhDerivativeFromOutput(z);
    }
    return sigmoidDerivative(z);
}

Eigen::VectorXd ActivationFunction::derivativeFromOutput(Eigen::VectorXd zs) const {
    switch(type) {
        case Sigmoid: return sigmoidDerivativeFromOutput(zs);
        case ReLU: return reLUDerivative(zs);
        case Tanh: return tanhDerivativeFromOutput(zs);
    }
    return sigmoidDerivative(zs);
}

Eigen::MatrixXd ActivationFunction::derivativeFromOutput(Eigen::MatrixXd zs) const {
    switch(type) {
        case Sigmoid: return sigmoidDerivativeFromOutput(zs);
        case ReLU: return reLUDerivative(zs);
        case Tanh: return tanhDerivativeFromOutput(zs);
    }
    return sigmoidDerivative(zs);
}

double ActivationFunction::sigmoid(double z) {
    if(z < -600.0) return 0.0;
    if(z > 600.0) return 1.0;
    return 1.0 / (1.0 + std::exp(-z));
}

Eigen::VectorXd ActivationFunction::sigmoid(Eigen::VectorXd& zs) {
    return zs.unaryExpr([](double x) {
        if(x < -600.0) return 0.0;
        if(x > 600.0) return 1.0;
        return 1.0 / (1.0 + std::exp(-x));
    });
}

Eigen::MatrixXd ActivationFunction::sigmoid(Eigen::MatrixXd& zs) {
    return zs.unaryExpr([](double x) {
        if(x < -600.0) return 0.0;
        if(x > 600.0) return 1.0;
        return 1.0 / (1.0 + std::exp(-x));
    });
}

double ActivationFunction::reLU(double z) {
    return z > 0 ? z : 0;
}

Eigen::VectorXd ActivationFunction::reLU(Eigen::VectorXd& zs) {
    return zs.unaryExpr([](double x) {return std::max(x,0.0);} );
}

Eigen::MatrixXd ActivationFunction::reLU(Eigen::MatrixXd& zs) {
    return zs.unaryExpr([](double x) {return std::max(x,0.0);} );
}

double ActivationFunction::sigmoidDerivative(double z) {
    double s = sigmoid(z);
    return s * (1-s);
}

double ActivationFunction::reLUDerivative(double z) {
    return z > 0 ? 1 : 0;
}

Eigen::VectorXd ActivationFunction::sigmoidDerivative(Eigen::VectorXd &zs) {
    Eigen::VectorXd sig = sigmoid(zs);
    return (sig.array() * (1.0 - sig.array())).matrix();
}

Eigen::VectorXd ActivationFunction::reLUDerivative(Eigen::VectorXd &zs) {
    return zs.unaryExpr([](double x) {return x > 0 ? 1.0 : 0;} );
}

Eigen::MatrixXd ActivationFunction::sigmoidDerivative(Eigen::MatrixXd &zs) {
    Eigen::MatrixXd sig = sigmoid(zs);
    return (sig.array() * (1.0 - sig.array())).matrix();
}

Eigen::MatrixXd ActivationFunction::reLUDerivative(Eigen::MatrixXd &zs) {
    return zs.unaryExpr([](double x) { return x > 0 ? 1.0 : 0; });
}

double ActivationFunction::tanhAct(double z) {
    return tanh(z);
}

Eigen::VectorXd ActivationFunction::tanhAct(Eigen::VectorXd& zs) {
    return zs.unaryExpr([](double x) {return tanh(x); });
}

Eigen::MatrixXd ActivationFunction::tanhAct(Eigen::MatrixXd& zs) {
    return zs.unaryExpr([](double x) {return tanh(x); });
}

double ActivationFunction::tanhDerivative(double z) {
    double t = tanh(z);
    return 1 - t*t;
}

Eigen::VectorXd ActivationFunction::tanhDerivative(Eigen::VectorXd& zs) {
    return zs.unaryExpr([](double x) {
        double t = tanh(x);
        return 1-t*t;
    });
}

Eigen::MatrixXd ActivationFunction::tanhDerivative(Eigen::MatrixXd& zs) {
    return zs.unaryExpr([](double x) {
        double t = tanh(x);
        return 1-t*t;
    });
}

double ActivationFunction::sigmoidDerivativeFromOutput(double z) {
    return z * (1-z);
}

Eigen::VectorXd ActivationFunction::sigmoidDerivativeFromOutput(Eigen::VectorXd& zs) {
    return (zs.array() * (1.0 - zs.array())).matrix();
}

Eigen::MatrixXd ActivationFunction::sigmoidDerivativeFromOutput(Eigen::MatrixXd& zs) {
    return (zs.array() * (1.0 - zs.array())).matrix();
}

double ActivationFunction::tanhDerivativeFromOutput(double z) {
    return 1 - z*z;
}

Eigen::VectorXd ActivationFunction::tanhDerivativeFromOutput(Eigen::VectorXd& zs) {
    return zs.unaryExpr([](double x) { return 1-x*x; });
}

Eigen::MatrixXd ActivationFunction::tanhDerivativeFromOutput(Eigen::MatrixXd& zs) {
    return zs.unaryExpr([](double x) { return 1-x*x; });
}