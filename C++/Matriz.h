#ifndef MATRIZ_H
#define MATRIZ_H

#include <iostream>
#include <string>

#include "Activations.h"

namespace neuralnetwork {

	class Matriz {
		public:
			int m, n;
			double **matriz; 

			// Construtor e Destrutor
			Matriz(int m, int n);
			Matriz(); // Sobrecarga de construtor
			~Matriz();

			// Operações
			Matriz dot(Matriz M);
			Matriz transpor();

			Matriz operator+(Matriz M); // Soma de matrizes
			Matriz operator+(double x); // Soma escalar
			
			Matriz operator-(Matriz M); // Subtração de matrizes
			Matriz operator-(double x); //Subtração escalar

			Matriz operator*(Matriz M); // Multiplicação element-wise
			Matriz operator*(double x); // Multiplicação escalar

			Matriz operator/(Matriz M); // Divisão element-wise
			Matriz operator/(double x); // Divisão escalar

			// Auxiliares
			friend std::ostream &operator<<(std::ostream &os, Matriz &M); // Printar a matriz
			int max();
			double somatorio();
			void salvar(FILE *fp);
			Matriz load(FILE *fp);
			Matriz copia(Matriz M);

			// NN
			Matriz Activation(double (*function) (double));
			Matriz Softmax();
			Matriz OneHotEncode(int label);

	};

	// Construtor
	Matriz::Matriz(int m, int n) {
		this->m = m;
		this->n = n;
		
		this->matriz = new double*[m];
		for (int i = 0; i < m; i++) {
			this->matriz[i] = new double[n];
		}
	
	}

	// Construtor sobrecarregado
	Matriz::Matriz() {}

	// Destrutor
	Matriz::~Matriz() {}

	// Dot Product
	Matriz Matriz::dot(Matriz M) {
		if (this->m != M.m || this->n != M.n) {
			std::cout << "Dimensoes nao compativeis para o dot product" << std::endl;
			std::cout << "M1: " << this->m << " " << this->n << std::endl;
			std::cout << "M2: " << M.m << " " << M.n << std::endl;
		}

		Matriz m = Matriz(this->m, this->n);

		for (int i = 0; i < this->m; i++) {
			for (int j = 0; j < this->n; j++) {
				m.matriz[i][j] = this->matriz[i][j] * M.matriz[i][j];
			}
		}

		return m;
	}

	// Transpor matriz
	Matriz Matriz::transpor() {
		Matriz M(this->n, this->m);

		for (int i = 0; i < this->m; i++) {
			for (int j = 0; j < this->n; j++) {
				M.matriz[j][i] = this->matriz[i][j];
			}
		}
		
		return M;
	}

	// Soma de matrizes
	Matriz Matriz::operator+(Matriz M) {
		if (this->m != M.m || this->n != M.n) {
			std::cout << "Dimensoes nao compativeis para a soma de matrizes" << std::endl;
			std::cout << "M1: " << this->m << " " << this->n << std::endl;
			std::cout << "M2: " << M.m << " " << M.n << std::endl;
		}

		Matriz m = Matriz(this->m, this->n);

		for (int i = 0; i < this->m; i++) {
			for (int j = 0; j < this->n; j++) {
				m.matriz[i][j] = this->matriz[i][j] + M.matriz[i][j];
			}
		}

		return m;
	}

	// Soma escalar
	Matriz Matriz::operator+(double x) {
		Matriz m = Matriz(this->m, this->n);

		for (int i = 0; i < this->m; i++) {
			for (int j = 0; j < this->n; j++) {
				m.matriz[i][j] = this->matriz[i][j] + x;
			}
		}

		return m;
	}

	// Subtração de matrizes
	Matriz Matriz::operator-(Matriz M) {
		if (this->m != M.m || this->n != M.n) {
			std::cout << "Dimensoes nao compativeis para a substracao de matrizes" << std::endl;
			std::cout << "M1: " << this->m << " " << this->n << std::endl;
			std::cout << "M2: " << M.m << " " << M.n << std::endl;
		}

		Matriz m = Matriz(this->m, this->n);

		for (int i = 0; i < this->m; i++) {
			for (int j = 0; j < this->n; j++) {
				m.matriz[i][j] = this->matriz[i][j] - M.matriz[i][j];
			}
		}

		return m;
	}

	// Subtração escalar
	Matriz Matriz::operator-(double x) {
		Matriz m = Matriz(this->m, this->n);

		for (int i = 0; i < this->m; i++) {
			for (int j = 0; j < this->n; j++) {
				m.matriz[i][j] = this->matriz[i][j] - x;
			}
		}

		return m;
	}

	// Multiplicação de matrizes
	Matriz Matriz::operator*(Matriz M) {
		if (this->n != M.m) {
			std::cout << "Dimensoes nao compativeis para a multiplicacao de matrizes" << std::endl;
			std::cout << "M1: " << this->m << " " << this->n << std::endl;
			std::cout << "M2: " << M.m << " " << M.n << std::endl;
			
			exit(-1);
		}

		Matriz M2(this->m, M.n);
		
		for (int i = 0; i < this->m; i++) {
			for (int j = 0; j < M.n; j++) {
				
				for (int k = 0; k < M.m; k++) {
					M2.matriz[i][j] += this->matriz[i][k] * M.matriz[k][j];
				}
				
			}
		}
			
		return M2;
	}

	// Multiplicação escalar
	Matriz Matriz::operator*(double x) {
		Matriz m = Matriz(this->m, this->n);

		for (int i = 0; i < this->m; i++) {
			for (int j = 0; j < this->n; j++) {
				m.matriz[i][j] = this->matriz[i][j] * x;
			}
		}

		return m;
	}

	// Divisão element-wise
	Matriz Matriz::operator/(Matriz M) {
		if (this->m != M.m || this->n != M.n) {
			std::cout << "Dimensoes nao compativeis para a divisao element-wise" << std::endl;
			std::cout << "M1: " << this->m << " " << this->n << std::endl;
			std::cout << "M2: " << M.m << " " << M.n << std::endl;
		}

		Matriz m = Matriz(this->m, this->n);

		for (int i = 0; i < this->m; i++) {
			for (int j = 0; j < this->n; j++) {
				m.matriz[i][j] = this->matriz[i][j] / M.matriz[i][j];
			}
		}

		return m;
	}

	// Divisão escalar
	Matriz Matriz::operator/(double x) {
		if (x == 0) {
			std::cout << "Erro: Divisão por 0!" << std::endl;
			exit(-1);
		}
		
		Matriz m = Matriz(this->m, this->n);

		for (int i = 0; i < this->m; i++) {
			for (int j = 0; j < this->n; j++) {
				m.matriz[i][j] = this->matriz[i][j] / x;
			}
		}

		return m;
	}

	// Printar a matriz
	std::ostream &operator<<(std::ostream &os, Matriz &M) {
		for (int i = 0; i < M.m; i++) {
			for (int j = 0; j < M.n; j++) {
				os << M.matriz[i][j] << " ";
			}
			os << std::endl;
		}
		
		return os;
	}

	// Retorna a posição de maior valor da matriz
	int Matriz::max() {
		int indexMax = 0;
		double max = -999999;

		for (int i = 0; i < this->m; i++) {
			for (int j = 0; j < this->n; j++) {
				if (this->matriz[i][j] >= max) {
					max = this->matriz[i][j];
					indexMax = i;
				}
				
			}
		}
		
		return indexMax;
	}

	// Retorna a soma de todos os elementos da matriz
	double Matriz::somatorio() {
		double soma = 0;
	
		for (int i =0; i < this->m; i++) {
			for (int j = 0; j < this->n; j++) {
				soma += this->matriz[i][j];
			}
		}
	
		return soma;
	}

	// Salva a matriz no arquivo passado em fp
	void Matriz::salvar(FILE *fp) {
		if (fp == NULL) {
			std::cout<< "Erro no arquivo ao salvar matriz!" << std::endl;
			return;
		}

		fprintf(fp, "%d %d\n", this->m, this->n);

		for (int i = 0; i < this->m; i++) {
			for (int j = 0; j < this->n; j++) {
				fprintf(fp, "%lf\n", this->matriz[i][j]);
			}
		}

		std::cout << "Matriz salva!" << std::endl;
	}	

	// Carrega a matriz no arquivo passado em fp
	Matriz Matriz::load(FILE *fp) {
		int m, n;
		
		if (fp == NULL) {
			std::cout << "Erro no arquivo ao carregar matriz!" << std::endl;
			exit(-1);
		}
		fseek(fp, ftell(fp), SEEK_SET);
	
		fscanf(fp, "%d %d", &m, &n);
	
		Matriz M = Matriz(m, n);
	
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				fscanf(fp, "%lf", &this->matriz[i][j]);
			}
		}
	
		std::cout << "Matriz carregada!" << std::endl;
		return M;
	}

	// Copia a matriz
	Matriz Matriz::copia(Matriz M) {

		Matriz C(M.m, M.n);

		for (int i = 0; i < M.m; i++) {
			for (int j = 0; j < M.n; j++) {
				C.matriz[i][j] = M.matriz[i][j];
			}
		}
		*this = C;
		
		return *this;
	}

	// Activation functions
	Matriz Matriz::Activation(double (*func)(double)) {

		Matriz M(this->m, this->n);

		for (int i = 0; i < this->m; i++) {
			for (int j; j < this->n; j++) {
				M.matriz[i][j] = func(this->matriz[i][j]);
			}
		}

		return M;
	}

	Matriz Matriz::Softmax() {
		double soma = 0;
		
		for (int i = 0; i < this->m; i++) {
			for (int j = 0; j < this->n; j++) {
				soma += exp(this->matriz[i][j]);
			}
		}
	
		for (int i = 0; i < this->m; i++) {
			for (int j = 0; j < this->n; j++) {
				this->matriz[i][j] = exp(this->matriz[i][j]) / soma;
			}
		}
	
		return *this;
	}

	Matriz Matriz::OneHotEncode(int label) {
	
		for (int i = 0; i < this->m; i++) {
			for (int j = 0; j < this->n; j++) {
				
				if (i == label)    this->matriz[i][j] = 1; 
				else 						   this->matriz[i][j] = 0;
			
			}
		}
		
		return *this;
	}

}

#endif