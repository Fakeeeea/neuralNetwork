#ifndef NEURAL_NETWORK_ACTIVATIONFUNCTION_H
#define NEURAL_NETWORK_ACTIVATIONFUNCTION_H

#include "Eigen/Dense"

class ActivationFunction {
public:
    enum ActivationType{
        Sigmoid,
        ReLU,
        Tanh,
    };

private:

    ActivationType type;

public:

    ActivationFunction();
    ActivationFunction(ActivationType f);

    double operator()(double z) const;
    Eigen::VectorXd operator()(Eigen::VectorXd zs) const;
    Eigen::MatrixXd operator()(Eigen::MatrixXd zs) const;

    double derivative(double z) const;
    Eigen::VectorXd derivative(Eigen::VectorXd zs) const;
    Eigen::MatrixXd derivative(Eigen::MatrixXd zs) const;

    double derivativeFromOutput(double z) const;
    Eigen::VectorXd derivativeFromOutput(Eigen::VectorXd zs) const;
    Eigen::MatrixXd derivativeFromOutput(Eigen::MatrixXd zs) const;

    static double sigmoid(double z);
    static Eigen::VectorXd sigmoid(Eigen::VectorXd& zs);
    static Eigen::MatrixXd sigmoid(Eigen::MatrixXd& zs);

    static double reLU(double z);
    static Eigen::VectorXd reLU(Eigen::VectorXd& zs);
    static Eigen::MatrixXd reLU(Eigen::MatrixXd& zs);

    static double tanhAct(double z);
    static Eigen::VectorXd tanhAct(Eigen::VectorXd& zs);
    static Eigen::MatrixXd tanhAct(Eigen::MatrixXd& zs);

    //derivatives
    static double sigmoidDerivative(double z);
    static Eigen::VectorXd sigmoidDerivative(Eigen::VectorXd& zs);
    static Eigen::MatrixXd sigmoidDerivative(Eigen::MatrixXd& zs);

    static double sigmoidDerivativeFromOutput(double z);
    static Eigen::VectorXd sigmoidDerivativeFromOutput(Eigen::VectorXd& zs);
    static Eigen::MatrixXd sigmoidDerivativeFromOutput(Eigen::MatrixXd& zs);

    static double reLUDerivative(double z);
    static Eigen::VectorXd reLUDerivative(Eigen::VectorXd& zs);
    static Eigen::MatrixXd reLUDerivative(Eigen::MatrixXd& zs);

    static double tanhDerivative(double z);
    static Eigen::VectorXd tanhDerivative(Eigen::VectorXd& zs);
    static Eigen::MatrixXd tanhDerivative(Eigen::MatrixXd& zs);

    static double tanhDerivativeFromOutput(double z);
    static Eigen::VectorXd tanhDerivativeFromOutput(Eigen::VectorXd& zs);
    static Eigen::MatrixXd tanhDerivativeFromOutput(Eigen::MatrixXd& zs);

};


#endif //NEURAL_NETWORK_ACTIVATIONFUNCTION_H
