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
#include <sys/time.h>


using namespace itensor;

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

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
    std::vector<double> gradRuntimesCPUSeq, hessRuntimesCPUSeq, gradRuntimesWallSeq, hessRuntimesWallSeq, costsSeq;
    for (double T : durations)
    {
        std::cout << "Calculating sequencial time-evolution for T = " << T << "\n";

        auto control    = SeedGenerator::adiabaticSeed(U_i,U_f,T/tstep+1);
        bool useBFGS    = false;
        OptimalControl<BH_tDMRG> OC(psi_f,psi_i,stepper,control.size(),0,useBFGS);
        
        // run time evolution and get fidelity
        clock_t begin   = clock();
        auto t1         = get_wall_time();
        auto grad       = OC.getAnalyticGradient(control);
        auto cost       = OC.getCost(control,false);
        clock_t end     = clock();
        auto t2         = get_wall_time();

        gradRuntimesCPUSeq.push_back(double(end - begin) / CLOCKS_PER_SEC); 
        gradRuntimesWallSeq.push_back(t2-t1);    

        t1              = get_wall_time();
        begin           = clock();
        auto hess       = OC.getHessian(control);
        end             = clock();
        t2              = get_wall_time();
        hessRuntimesCPUSeq.push_back(double(end - begin) / CLOCKS_PER_SEC); 
        hessRuntimesWallSeq.push_back(t2-t1);    


        costsSeq.push_back(cost);    
    } 

    // test parallel runtimes for 2 threads
    std::vector<double> gradRuntimesCPUPar2, hessRuntimesCPUPar2, gradRuntimesWallPar2, hessRuntimesWallPar2, costsPar;
    for (double T : durations)
    {
        std::cout << "Calculating parallel(2) time-evolution for T = " << T << "\n";

        auto control    = SeedGenerator::adiabaticSeed(U_i,U_f,T/tstep+1);
        bool useBFGS    = false;
        OptimalControl<BH_tDMRG> OC(psi_f,psi_i,stepper,control.size(),0,useBFGS);
        OC.setThreadCount(2);
        
        // run time evolution and get fidelity
        auto t1         = get_wall_time();
        clock_t begin   = clock();
        auto grad       = OC.getAnalyticGradient(control);
        auto cost       = OC.getCost(control,false);
        clock_t end     = clock();
        auto t2         = get_wall_time();
        gradRuntimesCPUPar2.push_back(double(end - begin) / CLOCKS_PER_SEC);    
        gradRuntimesWallPar2.push_back(t2-t1);    

        t1              = get_wall_time();
        begin           = clock();
        auto hess       = OC.getHessian(control);
        end             = clock();
        t2              = get_wall_time();
        hessRuntimesCPUPar2.push_back(double(end - begin) / CLOCKS_PER_SEC); 
        hessRuntimesWallPar2.push_back(t2-t1);    

        costsPar.push_back(cost);    
    }

    // test parallel runtimes for 4 threads
    std::vector<double> gradRuntimesCPUPar4, hessRuntimesCPUPar4, gradRuntimesWallPar4, hessRuntimesWallPar4;
    for (double T : durations)
    {
        std::cout << "Calculating parallel(4) time-evolution for T = " << T << "\n";

        auto control    = SeedGenerator::adiabaticSeed(U_i,U_f,T/tstep+1);
        bool useBFGS    = false;
        OptimalControl<BH_tDMRG> OC(psi_f,psi_i,stepper,control.size(),0,useBFGS);
        OC.setThreadCount(4);
        
        // run time evolution and get fidelity
        auto t1         = get_wall_time();
        clock_t begin   = clock();
        auto grad       = OC.getAnalyticGradient(control);
        auto cost       = OC.getCost(control,false);
        clock_t end     = clock();
        auto t2         = get_wall_time();
        gradRuntimesCPUPar4.push_back(double(end - begin) / CLOCKS_PER_SEC);    
        gradRuntimesWallPar4.push_back(t2-t1);    

        t1              = get_wall_time();
        begin           = clock();
        auto hess       = OC.getHessian(control);
        end             = clock();
        t2              = get_wall_time();
        hessRuntimesCPUPar4.push_back(double(end - begin) / CLOCKS_PER_SEC); 
        hessRuntimesWallPar4.push_back(t2-t1); 
    }  

    // test parallel runtimes for 8 threads
    std::vector<double> gradRuntimesCPUPar8, hessRuntimesCPUPar8, gradRuntimesWallPar8, hessRuntimesWallPar8;
    for (double T : durations)
    {
        std::cout << "Calculating parallel(8) time-evolution for T = " << T << "\n";

        auto control    = SeedGenerator::adiabaticSeed(U_i,U_f,T/tstep+1);
        bool useBFGS    = false;
        OptimalControl<BH_tDMRG> OC(psi_f,psi_i,stepper,control.size(),0,useBFGS);
        OC.setThreadCount(8);
        
        // run time evolution and get fidelity
        auto t1         = get_wall_time();
        clock_t begin   = clock();
        auto grad       = OC.getAnalyticGradient(control);
        auto cost       = OC.getCost(control,false);
        clock_t end     = clock();
        auto t2         = get_wall_time();
        gradRuntimesCPUPar8.push_back(double(end - begin) / CLOCKS_PER_SEC);    
        gradRuntimesWallPar8.push_back(t2-t1);    

        t1              = get_wall_time();
        begin           = clock();
        auto hess       = OC.getHessian(control);
        end             = clock();
        t2              = get_wall_time();
        hessRuntimesCPUPar8.push_back(double(end - begin) / CLOCKS_PER_SEC); 
        hessRuntimesWallPar8.push_back(t2-t1);    
    }
    std::cout << "\n\n";

    std::cout << "Printing gradient CPU runtimes .... " << "\n";
    std::cout << "Duration\t Runtimes Seq\t Runtimes Par(2)\t Runtimes Par(4)\t Runtimes Par(8)\n";
    for(size_t i = 0; i < durations.size(); i++)
    {
        std::cout << durations.at(i) << "\t";
        std::cout << gradRuntimesCPUSeq.at(i) << "\t";
        std::cout << gradRuntimesCPUPar2.at(i) << "\t";
        std::cout << gradRuntimesCPUPar4.at(i) << "\t";
        std::cout << gradRuntimesCPUPar8.at(i) << "\n";
    }
    std::cout << "\n\n";

    std::cout << "Printing gradient Wall runtimes .... " << "\n";
    std::cout << "Duration\t Runtimes Seq\t Runtimes Par(2)\t Runtimes Par(4)\t Runtimes Par(8)\n";
    for(size_t i = 0; i < durations.size(); i++)
    {
        std::cout << durations.at(i) << "\t";
        std::cout << gradRuntimesWallSeq.at(i) << "\t";
        std::cout << gradRuntimesWallPar2.at(i) << "\t";
        std::cout << gradRuntimesWallPar4.at(i) << "\t";
        std::cout << gradRuntimesWallPar8.at(i) << "\n";
    }
    std::cout << "\n\n";

    std::cout << "Printing Hessian CPU runtimes .... " << "\n";
    std::cout << "Duration\t Runtimes Seq\t Runtimes Par(2)\t Runtimes Par(4)\t Runtimes Par(8)\n";
    for(size_t i = 0; i < durations.size(); i++)
    {
        std::cout << durations.at(i) << "\t";
        std::cout << hessRuntimesCPUSeq.at(i) << "\t";
        std::cout << hessRuntimesCPUPar2.at(i) << "\t";
        std::cout << hessRuntimesCPUPar4.at(i) << "\t";
        std::cout << hessRuntimesCPUPar8.at(i) << "\n";
    }
    std::cout << "\n\n";

    std::cout << "Printing Hessian Wall runtimes .... " << "\n";
    std::cout << "Duration\t Runtimes Seq\t Runtimes Par(2)\t Runtimes Par(4)\t Runtimes Par(8)\n";
    for(size_t i = 0; i < durations.size(); i++)
    {
        std::cout << durations.at(i) << "\t";
        std::cout << hessRuntimesWallSeq.at(i) << "\t";
        std::cout << hessRuntimesWallPar2.at(i) << "\t";
        std::cout << hessRuntimesWallPar4.at(i) << "\t";
        std::cout << hessRuntimesWallPar8.at(i) << "\n";
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
