#pragma once

#include "Matrix.h"

#include <string>
#include <fstream>

class Training {
	std::string fileName = "training.txt";
	std::ifstream ifs;

	Matrix* inputBatch;
	Matrix* outputBatch;

	int trainingBatchSize; // columns
	int inputNeuronCount; // rows
	int outputNeuronCount; // rows
public:
	Training(int trainingBatchSize, int inputNeuronCount, int outputNeuronCount) {
		this->inputBatch = new Matrix(inputNeuronCount, trainingBatchSize);
		this->outputBatch = new Matrix(outputNeuronCount, trainingBatchSize);

		this->trainingBatchSize = trainingBatchSize;
		this->inputNeuronCount = inputNeuronCount;
		this->outputNeuronCount = outputNeuronCount;
		this->ifs.open(this->fileName);
	}

	~Training() {
		delete this->inputBatch;
		delete this->outputBatch;

		this->ifs.close();
	}

	void GetNextBatch() {
		for (int c = 0; c < this->trainingBatchSize; ++c) {
			for (int r = 0; r < inputNeuronCount; ++r) {
				float input;
				ifs >> input;
				inputBatch->operator()(r, c) = input;
			}
			
			for (int i = 0; i < outputNeuronCount; ++i) {
				float output;
				ifs >> output;
				outputBatch->operator()(i, c) = output;
			}
		}
	}

	Matrix& GetInputBatch() {
		return *this->inputBatch;
	}

	Matrix& GetOutputBatch() {
		return *this->outputBatch;
	}

	void PrintInputBatch() {
		this->inputBatch->PrintMatrix();
	}

	void PrintOutputBatch() {
		this->outputBatch->PrintMatrix();
	}
};