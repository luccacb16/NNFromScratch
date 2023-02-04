#ifndef NN_H
#define NN_H

#include <iostream>
#include <string>
#include <unistd.h>
#include <ctime>
#include <math.h>

#include "Matriz.h"
#include "Dados.h"
#include "Activations.h"

namespace neuralnetwork {

	class NN {
		public:
			Matriz Hidden;
			Matriz Output;
			Matriz Wi;
			Matriz Bi;
			Matriz Wh;
			Matriz Bh;
		 	
			double (*func) (double);
 			double (*dfunc) (double);

			// Construtor e Destrutor
			NN(int taminput, int numhidden, int numoutput);
			~NN();

			void FeedForward(Data Input);
			void BackPropagation(Data Input, double learningrate);

			void trainNN(const char *filename, double (*func) (double), double learningrate);
			void testNN(const char *filename, const char *savefile);

			// Auxiliares
			void saveNN(FILE *fp);
			 
	};

	// Auxiliar
	double randfrom(double min, double max) {
	  double range = (max - min); 
	  double div = RAND_MAX / range;
		
	  return min + (rand() / div);
	}

	void NN::saveNN(FILE *fp) {
		this->Wi.salvar(fp);
		this->Bi.salvar(fp);
		this->Wh.salvar(fp);
		this->Bh.salvar(fp);
	}

	void initParams(Matriz M) {

		for (int i = 0; i < M.m; i++) {
			for (int j = 0; j < M.n; j++) {
				M.matriz[i][j] = randfrom(-0.5, 0.5);
			}
		}

	}

	// Construtor
	NN::NN(int taminput, int numhidden, int numoutput) :
		Hidden(numhidden, 1), Output(numoutput, 1), Wi(numhidden, taminput), Bi(numhidden, 1), Wh(numoutput, numhidden), Bh(numoutput, 1) {	
		initParams(Wi);
		initParams(Bi);
		initParams(Wh);
		initParams(Bh);
	}

	// Destrutor
	NN::~NN() {}

	// FeedForward
	void NN::FeedForward(Data Input) {

		//Input.Dado = Input.Dado / 255;

		this->Hidden = this->Wi * Input.Dado + this->Bi; // Wi.X + Bi
		this->Hidden.Activation(this->func); // o(Wi.X + Bi)

		this->Output = this->Wh * this->Hidden + this->Bh; // Wh.H + Bh
		this->Output.Softmax(); // sm(Wh.H + Bh)

		//return *this;
	}

	// BackPropagation
	void NN::BackPropagation(Data Input, double learningrate) {
		Matriz OneHot(this->Output.m, this->Output.n);
		Matriz dWi(this->Wi.m, this->Wi.n);
		Matriz dBi(this->Bi.m, this->Bi.n);
		Matriz dWh(this->Wh.m, this->Wh.n);
		Matriz dBh(this->Bh.m, this->Bh.n);
		Matriz Erro, Erro2;

		// dWh e dBh
		OneHot.OneHotEncode(Input.label);
		Erro = (this->Output - OneHot) * 2;
		
		dWh = Erro * Hidden.transpor();
		dBh = Erro;

		// dWi, dBi
		Erro2 = (this->Wh.transpor() * Erro).dot(this->Hidden.Activation(this->dfunc));

		dWi = Erro2 * Input.Dado.transpor();
		dBi = Erro2;

		// Update parameters
		this->Wi = this->Wi - dWi * learningrate;
		this->Bi = this->Bi - dBi * learningrate;
		this->Wh = this->Wh - dWh * learningrate;
		this->Bh = this->Bh - dBh * learningrate;
		
		//return *this;
	}

	// Treina a NN
	void NN::trainNN(const char *filename, double (*func) (double), double learningrate) {
		int qtd, m, n;
		FILE *input = safeOpen(filename, "r");

		// Funções de ativação
		this->func = func;
		if (func == ReLU) {
			this->dfunc = dReLU;
		} else if (func == Tanh) {
			dfunc = dTanh;
		} else if (func == Sigmoid) {
			dfunc = dSigmoid;
		}

		fscanf(input, "%d", &qtd);
		fscanf(input, "%d %d", &m, &n);
		
		Data Input(m, n);

		for (int i = 0; i < qtd; i++) {
			Input.getData(input);

			this->FeedForward(Input);
			this->BackPropagation(Input, learningrate);

			if ((i+1) % (qtd/20) == 0) {
				printf("%.2lf%% do treinamento concluido\n", (double) i/qtd * 100);
				usleep(50000);
			}
		}
		
	}

	void NN::testNN(const char *filename, const char *savefile) {
		double acc;
		int media = 0;
		int numtestes, m, n;

		FILE *testfile = safeOpen(filename, "r");
		
		fscanf(testfile, "%d", &numtestes); // Quantidade de dados
		fscanf(testfile, "%d %d", &m, &n); // Tamanho da matriz
	
		Data Teste(m, n);
		
		for (int i = 0; i < numtestes; i++) {
			Teste.getData(testfile); // Recebe o dado
			this->FeedForward(Teste);
	
			if (this->Output.max() == Teste.label) {
				media++;
			}
			acc = (double) media / numtestes;
			
			if ((i+1) % (numtestes/20) == 0) {
				usleep(25000);
				printf("[%d] Accuracy: %.2lf%%\n", i+1, 100 * acc);
			}
			
		}
		
		if (acc >= 0.90) {
			FILE *wandb = safeOpen(savefile, "a+");
			printf("\n[%.2lf%%] Neural Network Salva!\n\n", acc * 100);
			this->saveNN(wandb);
		}
	}
	
}

#endif