#pragma once

class Matrix {
	float* data;
	int rows, columns;
public:
	Matrix(int rows, int columns) {
		this->rows = rows;
		this->columns = columns;

		this->data = (float*)_aligned_malloc(sizeof(float) * (rows * columns), 64);
	}

	~Matrix() {
		_aligned_free(this->data);
	}

	float& operator()(int row, int col) {
		return this->data[row + col * this->rows];
	}

	void COPY(Matrix& from) {
		if (this->columns != from.columns || this->rows != from.rows)
			throw "matrices must have the same dimension to copy";

		if (this == &from)
			throw "cannot copy a matrix to itself, thats retarded";

		for (int r = 0; r < from.rows; ++r) {
			for (int c = 0; c < from.columns; ++c) {
				this->operator()(r, c) = from(r, c);
			}
		}
	}
	
	void Transpose(Matrix& src) {
		_aligned_free(this->data);
		this->data = (float*)_aligned_malloc(sizeof(float) * (src.rows * src.columns), 64);

		this->rows = src.columns;
		this->columns = src.rows;
		for (int r = 0; r < src.rows; ++r) {
			for (int c = 0; c < src.columns; ++c) {
				this->operator()(c, r) = src(r, c);
			}
		}
	}

	void MUL(Matrix& lhs, Matrix& rhs) {
		if (lhs.columns != rhs.rows)
			throw "Columns of lhs matrix and rows of lhs matrix must be equal";

		if (this == &lhs || this == &rhs)
			throw "Destination matrix must be distinct from lhs and rhs";

		int sharedDim = lhs.columns;

		for (int r = 0; r < this->rows; ++r) {
			for (int c = 0; c < rhs.columns; ++c) {
				float total = 0.f;
				for (int d = 0; d < sharedDim; ++d) {
					total += lhs(r, d) * rhs(d, c);
				}

				this->operator()(r, c) = total;
			}
		}
	}

	void MUL(float scale) {
		for (int i = 0; i < this->rows * this->columns; ++i) {
			data[i] *= scale;
		}
	}

	void Hadamard(Matrix& rhs) {
		if (this->rows != rhs.rows || this->columns != rhs.columns)
			throw "matrices must be of same dimension";

		for (int i = 0; i < this->rows * this->columns; ++i) {
			this->data[i] *= rhs.data[i];
		}
	}

	void ADD(Matrix& rhs) {
		if (this->rows != rhs.rows || this->columns != rhs.columns)
			throw "matrices must be of same dimension";

		for (int i = 0; i < this->rows * this->columns; ++i) {
			this->data[i] += rhs.data[i];
		}
	}

	void SUB(Matrix& rhs) {
		if (this->rows != rhs.rows || this->columns != rhs.columns)
			throw "matrices must be of same dimension";

		for (int i = 0; i < this->rows * this->columns; ++i) {
			this->data[i] -= rhs.data[i];
		}
	}

	void GetAverageColumnVector(Matrix& dest) {
		for (int r = 0; r < this->rows; ++r) {
			float sum = 0.f;
			for (int c = 0; c < this->columns; ++c) {
				sum += this->operator()(r, c);
			}
			dest(r, 0) = sum / this->columns;
		}
	}

	void Randomize() {
		for (int i = 0; i < this->rows * this->columns; ++i) {
			data[i] = (float)rand() / (float)RAND_MAX;
		}
	}

	void PrintMatrix() {
		for (int r = 0; r < this->rows; ++r) {
			for (int c = 0; c < this->columns; ++c) {
				std::cout << this->operator()(r, c) << '\t';
			}
			std::cout << '\n';
		}
	}

	void AddColumnVector(Matrix* columnVector) {
		if (this->rows != columnVector->rows || columnVector->columns != 1)
			throw "Columns vector must have the same number of rows and a single column";

		for (int c = 0; c < this->columns; ++c) {
			for (int r = 0; r < this->rows; ++r) {
				this->operator()(r, c) += columnVector->operator()(r, 0);
			}
		}
	}

	void SubColumnVector(Matrix& columnVector) {
		if (this->rows != columnVector.rows || columnVector.columns != 1)
			throw "sub matrix must have the same number of rows and the column vector a single column";

		for (int c = 0; c < this->columns; ++c) {
			for (int r = 0; r < this->rows; ++r) {
				//this->operator()(r, 0) -= sub(r, c);
				this->operator()(r, c) -= columnVector(r, 0);
			}
		}
	}

	void ApplyFunction(Matrix* dest, float(*function)(float)) {
		for (int i = 0; i < this->rows * this->columns; ++i) {
			dest->data[i] = function(this->data[i]);
		}
	}
};