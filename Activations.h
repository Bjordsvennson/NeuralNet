#pragma once
#include <iostream>

#define E 2.71828182845904523536

float LogisticFunction(float x) {
	return 1.f / (1.f + pow(E, -x));
}

float LogisticFunctionDerivative(float x) {
	return LogisticFunction(x) * (1.f - LogisticFunction(x));
}

float TanH(float x) {
	return tanh(x);
}

float TanHDerivative(float x) {
	return 1.f - (TanH(x) * TanH(x));
}

float ReLU(float x) {
	if (x < 0.f)
		return 0.f;

	return x;
}

float ReLUDerivative(float x) {
	if (x < 0.f)
		return 0.f;

	return 1.f;
}