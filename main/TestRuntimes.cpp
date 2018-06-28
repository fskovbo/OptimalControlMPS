#include "itensor/all.h"
#include "BH_sites.h"
#include "InitializeState.hpp"
#include "BH_tDMRG.hpp"
#include "OptimalControl.hpp"
#include "SeedGenerator.hpp"
#include "correlations.hpp"
#include <string>
#include <iomanip>
#include <time.h>


using namespace itensor;


int main()
{
    std::vector<int> durations = {1 , 2 , 3  };

    double tstep    = 1e-2;

    int N           = 5;
    int Npart       = 5;
    int locDim      = 5;
   
    double J        = 1.0;
    double U_i      = 2.5;
    double U_f      = 50;

    auto sites      = BoseHubbard(N,locDim);
    auto psi_i      = InitializeState(sites,Npart,J,U_i);
    auto psi_f      = InitializeState(sites,Npart,J,U_f);
    auto stepper    = BH_tDMRG(sites,J,tstep,{"Cutoff=",1E-7,"Maxm=",40});

    // test sequencial runtimes
    std::vector<double> gradRuntimesSeq, hessRuntimesSeq, costsSeq;
    for (double T : durations)
    {
        std::cout << "Calculating sequencial time-evolution for T = " << T << "\n";

        auto control    = SeedGenerator::adiabaticSeed(U_i,U_f,T/tstep+1);
        bool useBFGS    = false;
        OptimalControl<BH_tDMRG> OC(psi_f,psi_i,stepper,control.size(),0,useBFGS);
        
        // run time evolution and get fidelity
        clock_t begin   = clock();
        auto grad       = OC.getAnalyticGradient(control);
        auto cost       = OC.getCost(control,false);
        clock_t end     = clock();
        gradRuntimesSeq.push_back(double(end - begin) / CLOCKS_PER_SEC);    

        begin           = clock();
        auto hess       = OC.getHessian(control);
        end             = clock();
        hessRuntimesSeq.push_back(double(end - begin) / CLOCKS_PER_SEC);    

        costsSeq.push_back(cost);    
    } 

    // test parallel runtimes for 2 threads
    std::vector<double> gradRuntimesPar2, hessRuntimesPar2, costsPar;
    for (double T : durations)
    {
        std::cout << "Calculating parallel(2) time-evolution for T = " << T << "\n";

        auto control    = SeedGenerator::adiabaticSeed(U_i,U_f,T/tstep+1);
        bool useBFGS    = true;
        OptimalControl<BH_tDMRG> OC(psi_f,psi_i,stepper,control.size(),0,useBFGS);
        OC.setThreadCount(2);
        
        // run time evolution and get fidelity
        clock_t begin   = clock();
        auto grad       = OC.getAnalyticGradient(control);
        auto cost       = OC.getCost(control,false);
        clock_t end     = clock();
        gradRuntimesPar2.push_back(double(end - begin) / CLOCKS_PER_SEC);    

        begin           = clock();
        auto hess       = OC.getHessian(control);
        end             = clock();
        hessRuntimesPar2.push_back(double(end - begin) / CLOCKS_PER_SEC);    

        costsPar.push_back(cost);    
    }

    // test parallel runtimes for 4 threads
    std::vector<double> gradRuntimesPar4, hessRuntimesPar4;
    for (double T : durations)
    {
        std::cout << "Calculating parallel(4) time-evolution for T = " << T << "\n";

        auto control    = SeedGenerator::adiabaticSeed(U_i,U_f,T/tstep+1);
        bool useBFGS    = true;
        OptimalControl<BH_tDMRG> OC(psi_f,psi_i,stepper,control.size(),0,useBFGS);
        OC.setThreadCount(4);
        
        // run time evolution and get fidelity
        clock_t begin   = clock();
        auto grad       = OC.getAnalyticGradient(control);
        auto cost       = OC.getCost(control,false);
        clock_t end     = clock();
        gradRuntimesPar4.push_back(double(end - begin) / CLOCKS_PER_SEC);    

        begin           = clock();
        auto hess       = OC.getHessian(control);
        end             = clock();
        hessRuntimesPar4.push_back(double(end - begin) / CLOCKS_PER_SEC);    
    }  

    // test parallel runtimes for 8 threads
    std::vector<double> gradRuntimesPar8, hessRuntimesPar8;
    for (double T : durations)
    {
        std::cout << "Calculating parallel(8) time-evolution for T = " << T << "\n";

        auto control    = SeedGenerator::adiabaticSeed(U_i,U_f,T/tstep+1);
        bool useBFGS    = true;
        OptimalControl<BH_tDMRG> OC(psi_f,psi_i,stepper,control.size(),0,useBFGS);
        OC.setThreadCount(8);
        
        // run time evolution and get fidelity
        clock_t begin   = clock();
        auto grad       = OC.getAnalyticGradient(control);
        auto cost       = OC.getCost(control,false);
        clock_t end     = clock();
        gradRuntimesPar8.push_back(double(end - begin) / CLOCKS_PER_SEC);    

        begin           = clock();
        auto hess       = OC.getHessian(control);
        end             = clock();
        hessRuntimesPar8.push_back(double(end - begin) / CLOCKS_PER_SEC);    
    }
    std::cout << "\n\n";

    std::cout << "Printing gradient runtimes .... " << "\n";
    std::cout << "Duration\t Runtimes Seq\t Runtimes Par(2)\t Runtimes Par(4)\t Runtimes Par(8)\n";
    for(size_t i = 0; i < durations.size(); i++)
    {
        std::cout << durations.at(i) << "\t";
        std::cout << gradRuntimesSeq.at(i) << "\t";
        std::cout << gradRuntimesPar2.at(i) << "\t";
        std::cout << gradRuntimesPar4.at(i) << "\t";
        std::cout << gradRuntimesPar8.at(i) << "\n";
    }
    std::cout << "\n\n";

    std::cout << "Printing Hessian runtimes .... " << "\n";
    std::cout << "Duration\t Runtimes Seq\t Runtimes Par(2)\t Runtimes Par(4)\t Runtimes Par(8)\n";
    for(size_t i = 0; i < durations.size(); i++)
    {
        std::cout << durations.at(i) << "\t";
        std::cout << hessRuntimesSeq.at(i) << "\t";
        std::cout << hessRuntimesPar2.at(i) << "\t";
        std::cout << hessRuntimesPar4.at(i) << "\t";
        std::cout << hessRuntimesPar8.at(i) << "\n";
    }
    std::cout << "\n\n";

    std::cout << "Printing costs .... " << "\n";
    std::cout << "Duration\t Cost (sequencial)\t Cost (parallel)\n";
    for(size_t i = 0; i < durations.size(); i++)
    {
        std::cout << durations.at(i) << "\t";
        std::cout << costsSeq.at(i) << "\t";
        std::cout << costsPar.at(i) << "\n";
    }

    
    return 0;
}
