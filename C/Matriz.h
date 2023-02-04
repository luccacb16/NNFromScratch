#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct matriz *Matriz;

struct matriz {
	double **matriz;
	int m, n;
};

void printaMatriz(Matriz M) {
	
	for (int i = 0; i < M->m; i++) {
		for (int j = 0; j < M->n; j++) {
			printf("%.0lf ", M->matriz[i][j]);
		}
		printf("\n\n");
	}

}

Matriz criaMatriz(int m, int n) {
	Matriz M = (Matriz) malloc(sizeof(Matriz));
	M->matriz = (double **) malloc(sizeof(double) * m);

	for (int i = 0; i < m; i++) {
		M->matriz[i] = (double *) malloc(sizeof(double) * n);
	}

	M->m = m;
	M->n = n;
	
	return M;
}

Matriz freeMatriz(Matriz M) {
	for (int i = 0; i < M->m; i++) {
		free(M->matriz[i]);
	}
	
	free(M->matriz);
	free(M);

	return NULL;
}

Matriz transporMatriz(Matriz M) {
	Matriz T = criaMatriz(M->n, M->m);

	for (int i = 0; i < M->m; i++) {
		for (int j = 0; j < M->n; j++) {
			T->matriz[j][i] = M->matriz[i][j];
		}
	}

	return T;
}

Matriz addMatrizes(Matriz m1, Matriz m2) {
	if (m1->m != m2->m || m1->n != m2->n) {
		printf("Tamanhos das matrizes diferentes para a adição!\n");
		exit(-1);
	}

	Matriz M = criaMatriz(m1->m, m1->n);

	for (int i = 0; i < m1->m; i++) {
		for (int j = 0; j < m1->n; j++) {
			M->matriz[i][j] = m1->matriz[i][j] + m2->matriz[i][j];
		}
	}
	
	return M;
}

Matriz subMatrizes(Matriz m1, Matriz m2) {
	if (m1->m != m2->m || m1->n != m2->n) {
		printf("Tamanhos das matrizes diferentes para a subtração!\n");
		exit(-1);
	}

	Matriz M = criaMatriz(m1->m, m1->n);

	for (int i = 0; i < m1->m; i++) {
		for (int j = 0; j < m1->n; j++) {
			M->matriz[i][j] = m1->matriz[i][j] - m2->matriz[i][j];
		}
	}
	
	return M;
}

Matriz addEscalar(Matriz M, double N) {
	for (int i = 0; i < M->m; i++) {
		for (int j = 0; j < M->n; j++) {
			 M->matriz[i][j] += N;
		}
	}

	return M;
}

Matriz subEscalar(Matriz M, double N) {
	for (int i = 0; i < M->m; i++) {
		for (int j = 0; j < M->n; j++) {
			M->matriz[i][j] -= N;
		}
	}

	return M;
}

Matriz prodEscalar(Matriz M, double N) {
	for (int i = 0; i < M->m; i++) {
		for (int j = 0; j < M->n; j++) {
			M->matriz[i][j] *= N;
		}
	}

	return M;
}

Matriz divEscalar(Matriz M, double N) {
	for (int i = 0; i < M->m; i++) {
		for (int j = 0; j < M->n; j++) {
			M->matriz[i][j] /= N;
		}
	}

	return M;
}

Matriz prodMatrizes(Matriz m1, Matriz m2) {
	if (m1->n != m2->m) {
		printf("Dimensões não equivalentes para o produto de matrizes!\n");
		exit(-1);
	}

	Matriz M = criaMatriz(m1->m, m2->n);

	for (int i = 0; i < m1->m; i++) {
		for (int j = 0; j < m2->n; j++) {

			for (int k = 0; k < m2->m; k++) {
				M->matriz[i][j] += m1->matriz[i][k] * m2->matriz[k][j];
			}
			
		}
	}

	return M;
}

Matriz prodElementos(Matriz m1, Matriz m2) {
	if (m1->m != m2->m || m1->n != m2->n) {
		printf("Dimensões não equivalentes para o produto de elementos!\n");
		exit(-1);
	}

	Matriz M = criaMatriz(m1->m, m1->n);

	for (int i = 0; i < m1->m; i++) {
		for (int j = 0; j < m2->n; j++) {
			M->matriz[i][j] = m1->matriz[i][j] * m2->matriz[i][j]; 
		}
	}

	return M;
}

void salvaMatriz(FILE *fp, Matriz M) {
	if (fp == NULL) {
		printf("Erro no arquivo ao salvar matriz!\n");
	}

	fprintf(fp, "%d %d\n", M->m, M->n);

	for (int i = 0; i < M->m; i++) {
		for (int j = 0; j < M->n; j++) {
			fprintf(fp, "%lf\n", M->matriz[i][j]);
		}
	}

	printf("Matriz salva!\n");
}

Matriz loadMatriz(FILE *fp) {
	int m, n;
	fseek(fp, ftell(fp), SEEK_SET);
	
	if (fp == NULL) {
		printf("Erro no arquivo ao carregar matriz!\n");
	}

	//rewind(fp); // FSEEK (byte offset volta pro começo)
	fscanf(fp, "%d %d", &m, &n);

	Matriz M = criaMatriz(m, n);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			fscanf(fp, "%lf", &M->matriz[i][j]);
		}
	}

	printf("Matriz carregada!\n");
	return M;
}

int maxMatriz(Matriz M) {
	int indexMax = 0;
	double max = -999999;

	for (int i = 0; i < M->m; i++) {
		for (int j = 0; j < M->n; j++) {
			if (M->matriz[i][j] >= max) {
				max = M->matriz[i][j];
				indexMax = i;
			}
			
		}
	}
	
	return indexMax;
}

Matriz copiaMatriz(Matriz A) {
	Matriz M = criaMatriz(A->m, A->n);

	for (int i = 0; i < A->m; i++) {
		for (int j = 0; j < A->n; j++) {
			M->matriz[i][j] = A->matriz[i][j];
		}
	}

	return M;
}

double somaMatriz(Matriz M) {
	double soma = 0;
	
	for (int i =0; i < M->m; i++) {
		for (int j = 0; j < M->n; j++) {
			soma += M->matriz[i][j];
		}
	}

	return soma;
}