#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include "Matriz.h"
#include "Dados.h"
#include "NN.h"

int main() {
	srand(time(NULL));

	/* Neural Network */
	NN Net = initNN(2, 5, 2); // taminput, numhidden, numoutput
	
	//Net = loadNN(wandb, Net);

	trainNN("xor.txt", Net, 0.008);
		
	testNN("xor2.txt", Net, "wandb.txt");
	
	/* Guess */
	/*
	char *guess = (char *) malloc(sizeof(char) * (taminput + 1));
	
	printf("\nInput: ");
	scanf("%s", guess);
	
	while (strcmp(guess, "q")) {
		Net = guessNN(guess, Net);

		printf("Prediction: %d", maxMatriz(Net->Output));
		printf(" [%s]\n", (maxMatriz(Net->Output) == chartoint(guess[strlen(guess)])) ? "Errado" : "Correto");
		printf("Confidence: %.2lf\n", Net->Output->matriz[maxMatriz(Net->Output)][0]);
		
		printf("\nInput: ");
		scanf("%s", guess);
	}

	free(guess);
	*/

	free(Net);
	Net = NULL;

	return 0;
}