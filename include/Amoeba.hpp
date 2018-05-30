#pragma once
#include<valarray>
#include<vector>
#include<string>
#include<algorithm>
#include<tuple>
#include<iostream>

struct Member {
	std::valarray<double> x;
	double fx;

	Member(std::valarray<double> x, double fx) : x(x), fx(fx) {

	}

	friend bool operator<(const Member& lhs, const Member& rhs) {
		return lhs.fx < rhs.fx;
	}
	friend bool operator>(const Member& lhs, const Member& rhs) {
		return lhs.fx > rhs.fx;
	}
};

class Amoeba {
	private:
		std::size_t dimension;
		std::vector<Member> v;
		bool display;

		// Stopping criteria
		unsigned int maxFun = 5000;
		unsigned int maxIter = 5000;
		double tolFun = 1e-6;

		// Optimization parameters
		double usual_delta = 0.05;
		double zero_term_delta = 0.00025;
		double rho = 1.0;
		double chi = 2.0;
		double psi = 0.5;
		double sigma = 0.5;

		std::valarray<double> const calcXBar() {
			std::valarray<double> xBar = v.at(0).x;
			for (std::size_t i = 1; i < dimension; ++i) {
				xBar = xBar + v.at(i).x;
			}
			return 1.0 / static_cast<double>(dimension) * xBar;
		}

		void const displayStatus(const unsigned int iter,const unsigned int func_evals,const double best, const std::string& what){
			std::cout << "    ";
			std::cout << iter << "\t" << "\t" << "    ";
			std::cout << func_evals << "\t" << "      ";
			std::cout << best << "\t" << "\t";
			std::cout << what << std::endl;
		}

		void const displayNames() {
			std::cout << "Iteration" << "\t";
			std::cout << "Func_evals" << "\t";
			std::cout << "Best" << "\t" << "\t" << "  ";
			std::cout << "Action" << std::endl;
		}

		template<typename F>
		void initializeSimplex(std::valarray<double>& x0,double fx0, F& foo) {
			Member nextMem = Member(x0, fx0);
			v.push_back(nextMem);

			for (std::size_t i = 0; i < dimension; ++i) {
				std::valarray<double> nextvec = x0;
				if (nextvec[i] != 0) {
					nextvec[i] = (1 + usual_delta)*nextvec[i];
				}
				else {
					nextvec[i] = zero_term_delta;
				}
				nextMem.x = nextvec; nextMem.fx = foo(nextvec);
				v.push_back(nextMem);
			}
		}
		template<typename F>
		void shrinkSimplex(F& foo) {
			for (std::size_t i = 1; i < dimension + 1; ++i) {
				std::valarray<double> nextvec = v.at(i).x;
				nextvec = v.at(1).x + sigma * (v.at(i).x - v.at(1).x);
				v.at(i) = Member(nextvec, foo(nextvec));
			}
		}

		bool const shouldStop(const unsigned int func_evals,const unsigned int iter) {
			bool stop = false;
			if (func_evals >= maxFun) stop = true;
			if (iter >= maxIter) stop = true;

			std::valarray<double> diff_fx(dimension);
			for (std::size_t i = 0; i < dimension; ++i) {
				diff_fx[i] = std::abs(v.at(0).fx - v.at(i + 1).fx);
			}
			if (diff_fx.max() <= tolFun) stop = true;
			return stop;
		}

	public:
		Amoeba(std::size_t dimension) : dimension(dimension) {
			v.reserve(dimension + 1);
			display = true;
		}

		template<typename F>
		std::tuple<double,std::valarray<double>,std::valarray<double>,std::valarray<unsigned int>> optimize(std::valarray<double> x0, F& rawFoo) {
			if (x0.size() != dimension) {
				throw std::invalid_argument("x0 does not have the correct size.");
			}
			// Wrap the function so we can keep track of function evaluations
			unsigned int func_evals = 0;
			unsigned int iter = 0;
			std::string what = "Start";
			std::valarray<double> costHistory(maxIter+1);
			std::valarray<unsigned int> func_evalsHistory(maxIter+1);

			auto foo = [&func_evals, &rawFoo](std::valarray<double> input) {
				func_evals++;
				return rawFoo(input);
			};

			// Evaluate start point
			double fx0 = foo(x0);
			if (display) displayNames();
			if (display) displayStatus(iter, func_evals, fx0, what);
			costHistory[iter] = fx0;

			// Initialize the simplex
			initializeSimplex(x0, fx0, foo);
			iter = iter + 1;
			std::sort(v.begin(), v.end());
			what = "Initialize";
			if (display) displayStatus(iter, func_evals, v.at(0).fx, what);
			costHistory[iter] = v.at(0).fx;
			func_evalsHistory[iter] = func_evals;

			// Main loop
			while (!shouldStop(func_evals,iter)) {
				// Calculate average of the n points
				std::valarray<double> xBar = calcXBar();
				// Calculate the reflection point 
				std::valarray<double> xr = (1.0 + rho)*xBar - rho * v.at(dimension).x;
				double fxr = foo(xr);

				if (fxr < v.at(0).fx) {
					// Calculate the expansion point
					std::valarray<double> xe = (1 + rho * chi)*xBar - rho * chi*v.at(dimension).x;
					double fxe = foo(xe);

					if (fxe < fxr) {
						v.at(dimension) = Member(xe, fxe);
						what = "Expand";
					}
					else {
						v.at(dimension) = Member(xr, fxr);
						what = "Reflect";
					}
				}
				// fxr >= v.at(0).fx
				else {
					if (fxr < v.at(dimension-1).fx) {
						v.at(dimension) = Member(xr, fxr);
						what = "Reflect";
					}
					// fxr >= v.at(n).fx
					else {
						// Perform contraction
						if (fxr < v.at(dimension).fx) {
							std::valarray<double> xc = (1 + psi * rho)*xBar - psi * rho*v.at(dimension).x;
							double fxc = foo(xc);
							if (fxc <= fxr) {
								v.at(dimension) = Member(xc, fxc);
								what = "Contract outside";
							}
							else {
								// Perform a shrink
								shrinkSimplex(foo);
								what = "Shrink";
							}
						}
						else {
							// Perform an inside contraction 
							std::valarray<double> xcc = (1 - psi)*xBar + psi * v.at(dimension).x;
							double fxcc = foo(xcc);
							if (fxcc < v.at(dimension).fx) {
								v.at(dimension) = Member(xcc, fxcc);
								what = "Contract inside";
							}
							else {
								// Perform a shrink
								shrinkSimplex(foo);
								what = "Shrink";
							}
						}
					}
				}
				std::sort(v.begin(), v.end());
				iter = iter + 1;
				if (display) displayStatus(iter, func_evals, v.at(0).fx, what);
				costHistory[iter] = v.at(0).fx;
				func_evalsHistory[iter] = func_evals;
			}
			// Fill in values missing if needed
			for (std::size_t i = iter; i <= maxIter; ++i) {
				costHistory[i] = v.at(0).fx;
				func_evalsHistory[i] = func_evals;
			}

			return std::make_tuple(v.at(0).fx, v.at(0).x, costHistory, func_evalsHistory);
		}		
};