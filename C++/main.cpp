#include <iostream>
#include <string>
#include <ctime>

#include "Matriz.h"
#include "Dados.h"
#include "NN.h"
#include "Activations.h"

using namespace neuralnetwork;

int main() {
	srand(time(NULL));

	NN Net(2, 10, 2);
	Net.trainNN("xor.txt", ReLU, 0.008);
	Net.testNN("xor2.txt", "wandb.txt");

	return 0;
}