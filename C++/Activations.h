#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <iostream>
#include <string>
#include <cmath>

namespace neuralnetwork {
	
	double ReLU(double Z) {
		return Z > 0 ? Z : 0;
	}
	
	double dReLU(double Z) {
		return Z > 0;
	}
	
	double Sigmoid(double Z) {
		return 1 / (1 + exp(-1 * Z));
	}
	
	double dSigmoid(double Z) {
		return Z * (1 - Z);
	}

	double Tanh(double Z) {
		return tanh(Z);
	}

	double dTanh(double Z) {
		return 1 - (tanh(Z) * tanh(Z));
	}

}

#endif