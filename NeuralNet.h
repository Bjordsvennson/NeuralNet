#pragma once

#include "Layer.h"
#include "Training.h"
#include "Activations.h"

#include <vector>

class NeuralNet {
	std::vector<Layer*> layers;
	Training* set;

	int trainingBatchSize;

	float error;
public:
	NeuralNet() {}

	~NeuralNet() {
		delete this->set;

		for (Layer* layer : this->layers) {
			delete layer;
		}

		this->layers.clear();
	}

	void Init(int inputNeuronCount, int trainingBatchSize, int outputNeuronCount) {
		if (layers.size() != 0)
			throw "Neural net has input layer already";

		this->trainingBatchSize = trainingBatchSize;

		this->set = new Training(trainingBatchSize, inputNeuronCount, outputNeuronCount);

		Layer* inputLayer = new Layer(inputNeuronCount);

		inputLayer->SetInputLayerActivationMatrix(inputNeuronCount, trainingBatchSize);

		this->layers.push_back(inputLayer);
	}

	void AddLayer(int neuronCount) {
		if (layers.size() < 1)
			throw "Neural net requires an input layer before adding layers";

		this->layers.back()->SetActivationFunction(ReLU, ReLUDerivative);

		Layer* hiddenLayer = new Layer(neuronCount);

		hiddenLayer->SetPreviousLayer(layers.back(), this->trainingBatchSize);

		layers.push_back(hiddenLayer);

		this->layers.back()->SetActivationFunction(LogisticFunction, LogisticFunctionDerivative);
	}

	void FeedForward() {
		if (this->layers.size() <= 1)
			throw "Net cannot feed forward with this few layers";

		this->set->GetNextBatch(); // probably should go in main

		Matrix& inputBatch = this->set->GetInputBatch();

		layers[0]->CopyInputLayerActivationMatrix(inputBatch);

		layers[0]->FeedForward();
	}

	void FeedBack() {
		if (this->layers.size() <= 1)
			throw "Net cannot feed back with this few layers";

		Matrix& targets = this->set->GetOutputBatch();

		this->layers.back()->FeedBack(targets, this->trainingBatchSize);
	}

	void PrintInputLayer() {
		this->layers.front()->PrintActivations();
	}

	void PrintOutputLayer() {
		this->layers.back()->PrintActivations();
	}
};