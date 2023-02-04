#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "Matriz.h"

// PROCESSAR OS DADOS
typedef struct data *Data;

struct data {
	Matriz Dados;
	int label;
};

int chartoint(char x) {
	return (int)x - (int)'0';
}

double chartodouble(char x) {
	return (double)x - (double)'0';
}

Data getDados(FILE *fp, Data Input) {
	//fseek(fp, ftell(fp), SEEK_SET);
	
	fscanf(fp, "%d", &Input->label); // Label

	for (int i = 0; i < Input->Dados->m; i++) {
		for (int j = 0; j < Input->Dados->n; j++) {
			fscanf(fp, "%lf", &Input->Dados->matriz[i][j]); // Coloca os dados na matriz
		}
	}
	
	return Input;
}

Data criaDado(int m, int n) {
	Data D = (Data) malloc(sizeof(Data));
	D->label = -1;

	D->Dados = criaMatriz(m, n);
	
	return D;
}

FILE *safeOpen(char *filename) {
	FILE *fp = fopen(filename, "r");
	if (fp == NULL) {
		printf("Erro ao abrir o arquivo '%s'\n", filename);
		exit(-1);
	}

	return fp;
}