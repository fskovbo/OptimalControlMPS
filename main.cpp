#include"Amoeba.h"

int main()
{

	// Rosenbrock function
	double a = 1.0;
	double b = 100;

	std::size_t dimension = 2;

	auto rosen = [a,b,dimension](std::valarray<double> input) {
		double val = 0;
		for (std::size_t i = 0; i < dimension - 1; ++i) {
			double xip1 = input[i + 1];
			double xi = input[i];
			val = val + (a - xi)*(a - xi) + b * (xip1 - xi * xi)*(xip1 - xi * xi);
		}
		return val;
	};

	
	std::valarray<double> initialPoint(dimension);
	for (std::size_t i = 0; i < dimension; ++i) {
		initialPoint[i] = 1.5;
	}
	Amoeba optimizer(dimension);
	
	auto result = optimizer.optimize(initialPoint,rosen);

	std::cout << "Printing Result" << std::endl;
	std::cout << "Best cost found: " << std::get<0>(result) << std::endl;

	std::cout << "Best solution found" << std::endl;
	for (auto& xi : std::get<1>(result)) {
		std::cout << xi << std::endl;
	}
	return 0;
	std::cout << "costHistory" << std::endl;
	for (auto& xi : std::get<2>(result)) {
		std::cout << xi << std::endl;
	}
	std::cout << "func_evalsHistory" << std::endl;
	for (auto& xi : std::get<3>(result)) {
		std::cout << xi << std::endl;
	}
}
