#pragma once

#include "Matrix.h"

class Layer {
	Matrix* activations;
	Matrix* sums;
	Matrix* weights;
	Matrix* biases;

	Matrix* deltaActivations;
	Matrix* deltaSums;
	Matrix* deltaWeights;

	Matrix* deltaWeightsTmp;

	Matrix* deltaBiases;

	int neuronCount;

	Layer* previousLayer;
	Layer* nextLayer;

	float(*activationFunction)(float);
	float(*derivative)(float);

	float learningRate = 0.5f;
public:
	Layer(int neuronCount) {
		this->neuronCount = neuronCount;
	}

	~Layer() {
		delete this->activations;
		delete this->sums;
		delete this->weights;
		delete this->biases;

		delete this->deltaActivations;
		delete this->deltaSums;
		delete this->deltaWeights;
		delete this->deltaWeightsTmp;
		delete this->deltaBiases;

	}

	void SetInputLayerActivationMatrix(int neuronCount, int trainingBatchSize) {
		this->activations = new Matrix(neuronCount, trainingBatchSize);
	}

	void CopyInputLayerActivationMatrix(Matrix& from) {
		this->activations->COPY(from);
	}

	void SetPreviousLayer(Layer* previousLayer, int trainingBatchSize) {
		this->previousLayer = previousLayer;
		previousLayer->nextLayer = this;

		this->activations = new Matrix(neuronCount, trainingBatchSize);
		this->deltaActivations = new Matrix(neuronCount, trainingBatchSize);

		this->sums = new Matrix(neuronCount, trainingBatchSize);
		this->deltaSums = new Matrix(neuronCount, trainingBatchSize);

		this->weights = new Matrix(neuronCount, this->previousLayer->neuronCount);
		weights->Randomize();
		this->deltaWeights = new Matrix(neuronCount, this->previousLayer->neuronCount);
		this->deltaWeightsTmp = new Matrix(neuronCount, this->previousLayer->neuronCount);

		this->biases = new Matrix(neuronCount, 1);
		biases->Randomize();
		this->deltaBiases = new Matrix(neuronCount, trainingBatchSize);
	}

	void FeedForward() {
		if (this->previousLayer) {
			this->sums->MUL(*this->weights, *this->previousLayer->activations); // multiply previous activations by connection weights of this layer
			this->sums->AddColumnVector(this->biases); // add bias

			this->sums->ApplyFunction(this->activations, activationFunction); // apply sigmoid
		}

		if (this->nextLayer) {
			this->nextLayer->FeedForward(); // continue feeding
		}
	}

	void FeedBack(Matrix& targets, int trainingBatchSize) {
		if (this->previousLayer) { // if not input layer
			this->deltaBiases->COPY(*this->activations); // get this layers activations
			this->sums->ApplyFunction(this->deltaSums, this->derivative); // sums = σ′(z^L) = delta
			
			if (!this->nextLayer) { // output layer get error
				this->deltaBiases->SUB(targets); // deltaBiases = (a^L − y) subtract output layer activations from target activations
				//3b1b says multiply deltabiases by scalar 2 at this point
			}
			else { // hidden layer get error
				this->deltaWeightsTmp->Transpose(*this->nextLayer->weights); // deltaWeightsTmp = (w^(l+1))^T
				this->deltaBiases->MUL(*this->deltaWeightsTmp, *this->nextLayer->deltaBiases); // deltaBiases = (w^(l + 1))^T * δ^(l + 1)
			} // w^l_jk = the weight from the k'th neuron in the (l-1)'th layer to the j'th in the l'th layer

			this->deltaBiases->Hadamard(*this->deltaSums);
			this->deltaActivations->Transpose(*this->previousLayer->activations); // the description in the book didn't say to tranpose again, but the python code he provided did, and from testing it seems necessary
			this->deltaWeights->MUL(*this->deltaBiases, *this->deltaActivations); // deltaWeights = a^(l−1)_k * δ^l_j.

			//this->deltaWeights->MUL(*this->previousLayer->activations, *this->errors); this is what the book says but not what the code he distributed actually does

			Matrix averageDeltaBias = Matrix(this->neuronCount, 1);
			Matrix averageDeltaWeight = Matrix(this->neuronCount, 1);

			this->deltaBiases->GetAverageColumnVector(averageDeltaBias);
			this->deltaWeights->GetAverageColumnVector(averageDeltaWeight);

			averageDeltaBias.MUL(this->learningRate / trainingBatchSize);
			averageDeltaWeight.MUL(this->learningRate / trainingBatchSize);

			this->biases->SubColumnVector(averageDeltaBias);
			this->weights->SubColumnVector(averageDeltaWeight);

			this->previousLayer->FeedBack(targets, trainingBatchSize);
		}
	}

	void SetActivationFunction(float(*activationFunction)(float), float(*derivative)(float)) {
		this->activationFunction = activationFunction;
		this->derivative = derivative;
	}

	void PrintActivations() {
		this->activations->PrintMatrix();
	}
};
