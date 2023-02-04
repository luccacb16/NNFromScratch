#ifndef DADOS_H
#define DADOS_H

#include <iostream>
#include <string>

#include "Matriz.h"

namespace neuralnetwork {

	class Data {
		public:
			Matriz Dado;
			int label;
			int qtd;
			
			Data(int m, int n);
			~Data();

			Data *getData(FILE *fp);

	};

	// Auxiliar
	FILE *safeOpen(const char *file, const char *modo) {

		FILE *fp = fopen(file, modo);
		if (fp == NULL) {
			printf("Erro ao abrir o arquivo '%s'\n", file);
			exit(-1);
		}

		return fp;
	}

	// Construtor
	Data::Data(int m, int n) : Dado(m, n) {}

	// Destrutor
	Data::~Data() {}

	// Pega o dado do arquivo
	Data *Data::getData(FILE *fp) {

			
		for (int i = 0; i < this->Dado.m; i++) {
			for (int j = 0; j < this->Dado.n; j++) {
				fscanf(fp, "%lf", &this->Dado.matriz[i][j]);
			}
		}

		fscanf(fp, "%d", &this->label); // Label
	
		return this;
	}

}

#endif