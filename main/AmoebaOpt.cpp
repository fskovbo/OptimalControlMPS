﻿#include "OptimalControl.hpp"
#include "ControlBasisFactory.hpp"
#include "SeedGenerator.hpp"
#include "itensor/all.h"
#include "BH_sites.h"
#include "BH_tDMRG.hpp"
#include "InitializeState.hpp"
#include "Amoeba.hpp"
#include <fstream>
#include <stdlib.h>


template<typename OC>
class OCWrapper
{
private:
    OC& optObj;

    double calcPenalty(std::vector<double> input){
        std::vector<double> control = optObj.getControl(input);

        double above = 0;
        double below = 0;

        for(std::size_t i = 0; i< control.size(); ++i){
            if(control[i]>uMax){
                above = above + (control[i]-uMax)*(control[i]-uMax);
            }
            if(control[i]<uMin){
                below = below + (control[i]-uMin)*(control[i]-uMin);
            }
        }
        return gammaBound*(above+below);
    }

public:
    double gammaBound;
    double uMin;
    double uMax;

    OCWrapper(OC& optObj,double uMin, double uMax,double gamma = 100):optObj(optObj),uMin(uMin),uMax(uMax){
        gammaBound = gamma;
    }


    double operator ()(std::valarray<double> input){
        std::vector<double> vecInput(input.size());
        vecInput.assign(std::begin(input),std::end(input));

        return optObj.getCost(vecInput) + calcPenalty(vecInput);
    }
};


int main(int argc, char* argv[])
{
	if(argc < 2) {
		printfln("Usage: %s InputFile_BHcontrol",argv[0]);
		return 0;
	}

	auto input      = InputGroup(argv[1],"input");

	double tstep    = input.getReal("tstep",1e-2);
	double T        = input.getReal("T");

	int N           = input.getInt("N");
	int Npart       = input.getInt("Npart");
	int locDim      = input.getInt("d");

	double J        = 1.0;
	double U_i      = 2.5;
	double U_f      = 50;

	int M           = input.getInt("M");
	double gamma    = input.getReal("gamma",0);
	bool cache      = input.getYesNo("cacheProgress",false);
	int maxBondDim  = input.getInt("maxBondDim",100);
	double optTol   = input.getReal("optTol",1e-7);
	double threshold= input.getReal("threshold",1e-7);
    double gammaBound= input.getReal("gammaBound",100);
	
	int seed      = 1;

	if(argc > 2) seed = std::stoi(argv[2]);
	else printfln("Default seed used");

	srand ((unsigned) seed*time(NULL));


	std::cout << "Performing optimal control of Bose-Hubbard model ... \n\n";
	std::cout << " ******* Parameters used ******* \n";
	std::cout << "Number of sites ................ " << N << "\n";
	std::cout << "Number of particles ............ " << Npart << "\n";
	std::cout << "Local Fock space dimension ..... " << locDim << "\n";
	std::cout << "Control duration ............... " << T << "\n";
	std::cout << "Time-step size ................. " << tstep << "\n";
	std::cout << "GROUP dimension ................ " << M << "\n";
	std::cout << "Gamma (regularisation) ......... " << gamma << "\n";
	std::cout << "Maximum bond dimension (MPS).... " << maxBondDim << "\n";
	std::cout << "Truncation threshold (MPS) ..... " << threshold << "\n";
	std::cout << "Optimization tolerance (IPOPT).. " << optTol << "\n";
	std::cout << "Seed  .......................... " << seed << "\n\n\n";


	auto sites    = BoseHubbard(N,locDim);
	auto u0       = SeedGenerator::linsigmoidSeed(U_i,U_f,T/tstep+1);
	auto basis    = ControlBasisFactory::buildChoppedSineBasis(u0,tstep,T,M);
    auto psi_i    = InitializeState(sites,Npart,J,u0.front(),maxBondDim,threshold);
    auto psi_f    = InitializeState(sites,Npart,J,u0.back(),maxBondDim,threshold);

	auto stepper  = BH_tDMRG(sites,J,tstep,{"Cutoff=",threshold,"Maxm=",maxBondDim});
	OptimalControl<BH_tDMRG> OC(psi_f,psi_i,stepper,basis,gamma);

	// Cost function encapsulated wrapper

    auto costFunc = OCWrapper<decltype(OC)>(OC,2,100,gammaBound);

	std::valarray<double> initialPoint(M);
	for (std::size_t i = 0; i < M; ++i) {
		initialPoint[i] = 0;
	}
	Amoeba optimizer(M);
	
	// Optimize cost
	auto result = optimizer.optimize(initialPoint,costFunc);

	std::cout << "Printing Result" << std::endl;
	std::cout << "Best cost found: " << std::get<0>(result) << std::endl;

	std::cout << "Best solution found" << std::endl;
	for (auto& xi : std::get<1>(result)) {
		std::cout << xi << std::endl;
	}
	
	// std::cout << "costHistory" << std::endl;
	// for (auto& xi : std::get<2>(result)) {
	// 	std::cout << xi << std::endl;
	// }
	// std::cout << "func_evalsHistory" << std::endl;
	// for (auto& xi : std::get<3>(result)) {
	// 	std::cout << xi << std::endl;
	// }

	// Save results
	std::vector<double> initCont(M);
	std::copy(begin(initialPoint), end(initialPoint), begin(initCont));
	auto initialFidelities = OC.getFidelityForAllT(initCont);

	std::vector<double> finalCont(M);
	std::copy(begin(std::get<1>(result)), end(std::get<1>(result)), begin(finalCont));
	auto finalFidelities = OC.getFidelityForAllT(finalCont);
	auto finalControl = OC.getControl(finalCont);

	auto times = OC.getTimeAxis();
	std::ofstream myfile1 ("BHrampInitialFinal.txt");
	if (myfile1.is_open())
	{
		for (int i = 0; i < OC.getN(); i++) {
			myfile1 << times.at(i) << "\t";
			myfile1 << u0.at(i) << "\t";
			myfile1 << initialFidelities.at(i) << "\t";
			myfile1 << finalControl.at(i) << "\t";
			myfile1 << finalFidelities.at(i) << "\n";
		}
		myfile1.close();
	}
	else std::cout << "Unable to open file\n";

	size_t iter = 0;
    std::ofstream myfile2 ("ProgressCache.txt");
    if (myfile2.is_open())
    {
		for (int i = 0; i < std::get<3>(result).size(); i++) {
			myfile2 << iter++ << "\t";
			myfile2 << std::get<2>(result)[i] << "\t";
			myfile2 << times.back() << "\t";
			myfile2 << std::get<3>(result)[i] << "\t";
			myfile2 << OC.getM() << "\n";
		}
    }
    else std::cout << "Unable to open file\n";

	return 0;
}
