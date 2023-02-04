#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#include "Matriz.h"
#include "Dados.h"

typedef struct nn *NN;
struct nn {
	Matriz Hidden; 	// 10x1
	Matriz Output; 	// 2x1
	Matriz Wi; 			// 10x50
	Matriz Bi; 			// 10x1
	Matriz Wh; 			// 2x10
	Matriz Bh; 			// 2x1
};

double randfrom(double min, double max) {
  double range = (max - min); 
  double div = RAND_MAX / range;
	
  return min + (rand() / div);
}

void initParams(Matriz M) {

	for (int i = 0; i < M->m; i++) {
		for (int j = 0; j < M->n; j++) {
			M->matriz[i][j] = randfrom(-0.5, 0.5);
		}
	}

}

NN initNN(int taminput, int numhidden, int numoutput) {

	NN Net = (NN) malloc(sizeof(NN));

	Net->Hidden = criaMatriz(numhidden, 1); // 10x1
	Net->Output = criaMatriz(numoutput, 1); // 2x1

	Net->Wi = criaMatriz(numhidden, taminput); // 10x50
	Net->Bi = criaMatriz(numhidden, 1); // 10x1
	
	Net->Wh = criaMatriz(numoutput, numhidden); // 2x10
	Net->Bh = criaMatriz(numoutput, 1); // 2x1

	initParams(Net->Wi); // 10x50
	initParams(Net->Bi); // 10x1
	
	initParams(Net->Wh); // 2x10
	initParams(Net->Bh); // 2x1
	
	return Net;
}

double Sigmoid(double Z) {
	return 1 / (1 + exp(-1 * Z));
}

double ReLU(double Z) {
	return Z > 0 ? Z : 0; // ou Z * (Z > 0)
}

double derivSigmoid(double Z) {
	return Z * (1 - Z);
}

double derivReLU(double Z) {
	return Z > 0;
}

double derivTanh(double Z) {
	return 1 - (tanh(Z) * tanh(Z));
}
	
Matriz ActivationFunction(Matriz Z) {
	for (int i = 0; i < Z->m; i++) {
		for (int j = 0; j < Z->n; j++) {
			Z->matriz[i][j] = ReLU(Z->matriz[i][j]);
		}
	}

	return Z;
}

Matriz derivActivationFunction(Matriz Z) {

	Matriz M = copiaMatriz(Z);
	
	for (int i = 0; i < Z->m; i++) {
		for (int j = 0; j < Z->n; j++) {
			M->matriz[i][j] = derivReLU(M->matriz[i][j]);
		}
	}

	return M;
}

Matriz Softmax(Matriz M) {
	double soma = 0;
	
	for (int i = 0; i < M->m; i++) {
		for (int j = 0; j < M->n; j++) {
			soma += exp(M->matriz[i][j]);
		}
	}

	for (int i = 0; i < M->m; i++) {
		for (int j = 0; j < M->n; j++) {
			M->matriz[i][j] = exp(M->matriz[i][j]) / soma;
		}
	}

	return M;
}

Matriz OneHotEncode(Matriz M, int label) {

	for (int i = 0; i < M->m; i++) {
		for (int j = 0; j < M->n; j++) {
			if (i == label) M->matriz[i][j] = 1; 
			else M->matriz[i][j] = 0;
		}
	}
	
	return M;
}

void FeedForward(Data Input, NN Net) {
	/*
	LAYER:   M             N 
	-------------------------------------
	Input:   50   linhas   1    coluna
	Hidden:  10   linhas   1    coluna
	Output:  2    linhas   1    coluna
	*/

	Input->Dados = divEscalar(Input->Dados, 255);
	printf("\nHidden: %d %d\n", Net->Hidden->m, Net->Hidden->n);

	Net->Hidden = prodMatrizes(Net->Wi, Input->Dados); // Wi.X

	Net->Hidden = addMatrizes(Net->Hidden, Net->Bi); // Wi.X + Bi
	Net->Hidden = ActivationFunction(Net->Hidden); // o(Wi.X + Bi)

	Net->Output = prodMatrizes(Net->Wh, Net->Hidden); // Wh.H
	Net->Output = addMatrizes(Net->Output, Net->Bh); // Wh.H + Bh
	Net->Output = Softmax(Net->Output); // sm(Wh.H + Bh)

	//return Net;
}

void BackPropagation(Data Input, NN Net, double learningRate) {
	Matriz OneHot = copiaMatriz(Net->Output);
	OneHot = OneHotEncode(OneHot, Input->label);

	Matriz dWi = copiaMatriz(Net->Wi); // 10x50
	Matriz dBi = copiaMatriz(Net->Bi); // 10x1
	
	Matriz dWh = copiaMatriz(Net->Wh); // 2x10
	Matriz dBh = copiaMatriz(Net->Bh); // 2x1

	Matriz Erro = copiaMatriz(Net->Output);
	Matriz Gradiente = copiaMatriz(Net->Output);
	Matriz HiddenT = copiaMatriz(Net->Hidden);

	/* dWh e dBh */
	Erro = subMatrizes(Net->Output, OneHot); // Calcula o erro
	Gradiente = derivActivationFunction(Net->Output);
	Gradiente = prodElementos(Gradiente, Erro); // Calcula o gradiente
	
	HiddenT = transporMatriz(HiddenT); // Transpoe a matriz Hidden

	dWh = prodMatrizes(Gradiente, HiddenT); // Calcula o delta de Wh
	dBh = copiaMatriz(Gradiente); //delta Bh é o gradiente

	/* dWi e dBi */
		// Gradiente Hidden
	Matriz WhT = copiaMatriz(Net->Wh);
	WhT = transporMatriz(WhT);
	
	Matriz ErroH = copiaMatriz(Net->Hidden);
	ErroH = prodMatrizes(WhT, Erro);
	
	Gradiente = copiaMatriz(Net->Hidden); // Reutilizando a matriz Gradiente
	Gradiente = derivActivationFunction(Net->Hidden);
	Gradiente = prodElementos(Gradiente, ErroH);

	Matriz InputT = copiaMatriz(Input->Dados);
	InputT = transporMatriz(InputT);
	
	dWi = prodMatrizes(Gradiente, InputT); // Calcula o delta de Wi
	dBi = copiaMatriz(Gradiente); // delta Bi é o gradiente

	/* Atualizar os parâmetros */
	dWi = prodEscalar(dWi, learningRate);
	dBi = prodEscalar(dBi, learningRate);
	dWh = prodEscalar(dWh, learningRate);
	dBh = prodEscalar(dBh, learningRate);

	Net->Wi = subMatrizes(Net->Wi, dWi);
	Net->Bi = subMatrizes(Net->Bi, dBi);
	Net->Wh = subMatrizes(Net->Wh, dWh);
	Net->Bh = subMatrizes(Net->Bh, dBh);
	
	//return Net;
}

void saveNN(FILE *fp, NN Net) {
	salvaMatriz(fp, Net->Wi);
	salvaMatriz(fp, Net->Bi);
	salvaMatriz(fp, Net->Wh);
	salvaMatriz(fp, Net->Bh);
}

NN loadNN(char *filename, NN Net) {

	FILE *fp = fopen(filename, "a+");
	
	Net->Wi = loadMatriz(fp);
	Net->Bi = loadMatriz(fp);
	Net->Wh = loadMatriz(fp);
	Net->Bh = loadMatriz(fp);

	return Net;
}

void trainNN(char *filename, NN Net, double learningRate) {
	int qtd, m, n;

	FILE *inputfile = safeOpen(filename);
	
	fscanf(inputfile, "%d", &qtd); // Quantidade de dados
	fscanf(inputfile, "%d %d", &m, &n); // Tamanho da matriz

	Data Input = criaDado(m, n); // Cria o dado
	
	for (int i = 0; i < qtd; i++) {
		Input = getDados(inputfile, Input); // Recebe o dado
		
		FeedForward(Input, Net);
		BackPropagation(Input, Net, learningRate);
	}

	fclose(inputfile);
	freeMatriz(Input->Dados);
	free(Input);
	
	//return Net;
}

void testNN(char *filename, NN Net, char *wandbfile) {
	double acc;
	int media = 0;
	int numtestes, m, n;

	FILE *testfile = safeOpen(filename);
	FILE *wandb = fopen(wandbfile, "a+");
	
	fscanf(testfile, "%d", &numtestes); // Quantidade de dados
	fscanf(testfile, "%d %d", &m, &n); // Tamanho da matriz

	Data Teste = criaDado(m, n);
	
	for (int i = 0; i < numtestes; i++) {
		Teste = getDados(testfile, Teste); // Recebe o dado
		FeedForward(Teste, Net);

		if (maxMatriz(Net->Output) == Teste->label) {
			media++;
		}
		acc = (double) media / numtestes;
		
		if ((i+1) % (numtestes/20) == 0) {
			usleep(25000);
			printf("[%d] Accuracy: %.2lf%%\n", i+1, 100 * acc);
		}
		
	}
	
	if (acc >= 0.95) {
		printf("\n[%.2lf%%] Neural Network Salva!\n\n", acc * 100);
		saveNN(wandb, Net);
	}

	fclose(testfile);
	freeMatriz(Teste->Dados);
	free(Teste);
	
	//return Net;
}
/*
NN guessNN(FILE *fp, char *Input, NN Net) {
	Data Guess = criaDado();
	Guess->Dados = criaMatriz(strlen(Input)-1, 1);
	
	for (int j = 0; j < strlen(Input)-1; j++) {
		Guess->Dados->matriz[j][0] = chartodouble(Input[j]); // Copia a guess
	}
	Guess->label = chartoint(Input[strlen(Input)-1]); // Copia a label

	Net = FeedForward(Guess, Net); // Passa pela NN
	
	free(Guess); // Libera vetor

	return Net;
}
*/