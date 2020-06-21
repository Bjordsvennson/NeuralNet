#include <iostream>

#include "NeuralNet.h"

int main() {

	NeuralNet net = NeuralNet();

	net.Init(2, 3, 1);
	net.AddLayer(10);
	net.AddLayer(5);
	net.AddLayer(1);

	net.FeedForward();

	net.PrintInputLayer();
	std::cout << '\n';
	net.PrintOutputLayer();
	std::cout << '\n';

	for (int i = 0; i < 3000; ++i) {
		net.FeedBack();
		net.FeedForward();

		net.PrintInputLayer();
		std::cout << '\n';
		net.PrintOutputLayer();
		std::cout << '\n';
	}

	system("PAUSE");

	return 0;
}