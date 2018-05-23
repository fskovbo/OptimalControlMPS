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


int main(int argc, char* argv[])
{
    std::vector<int> maxBondDim = {10 , 20, 30, 40 , 50 , 500 };

    if (argc < 3)
    {
        printfln("Usage: %s InputFile_BHcontrol BHrampInitialFinal.txt",argv[0]);
        printfln("No input detected ... using standard parameters");
    }

    double tstep  = 1e-2;
    double T      = 6;

    int N         = 15;
    int Npart     = 15;
    int locDim    = 10;

    // load InputFile_BHcontrol.txt
    if (argc >= 2)
    {
        auto input  = InputGroup(argv[1],"input");
        tstep       = input.getReal("tstep",1e-2);
        T           = input.getReal("T",6);

        N           = input.getInt("N",8);
        Npart       = input.getInt("Npart",8);
        locDim      = input.getInt("d",8);    
    }


    double J      = 1.0;
    double U_i    = 2.0;
    double U_f    = 50;


    // load BHrampInitialFinal.txt
    std::vector<double> control, times;
    if (argc >= 3)
    {
        std::string RampDataFile(argv[2]);
        double tmp, t, u;
        std::ifstream in(RampDataFile);
        while(!in.eof())
        {
            in >> t >> tmp >> tmp >> u >> tmp;
            times.push_back(t);
            control.push_back(u);
        }
    } 
    else
    {
        times   = SeedGenerator::generateRange(0,tstep,T);
        control = SeedGenerator::adiabaticSeed(U_i,U_f,times.size());
    }

    auto sites      = BoseHubbard(N,locDim);
    auto psi_i      = InitializeState(sites,Npart,J,control.front());
    auto psi_f      = InitializeState(sites,Npart,J,control.back());

    // extend duration and controls
    for (size_t i = 1; i <= 50; i++)
    {
        times.push_back(T + i*tstep);
        control.push_back( control.back() );
    }

    // Create an output string stream
    std::ostringstream streamObj1;
    // Set Fixed -Point Notation
    streamObj1 << std::fixed;
    // Set precision to 1 digits
    streamObj1 << std::setprecision(1);
    //Add double to stream
    streamObj1 << T;

    std::vector<double> runtimes;

    for (double maxM : maxBondDim)
    {
        std::cout << "Calculating time-evolution for maxM = " << maxM << "\n";

        auto stepper    = BH_tDMRG(sites,J,tstep,{"Cutoff=",1E-8,"Maxm=",maxM});
        OptimalControl<BH_tDMRG> OC(psi_f,psi_i,stepper,0);
        
        // run time evolution and get fidelity
        clock_t begin   = clock();
        auto fid        = OC.getFidelityForAllT(control);
        auto psi        = OC.getPsit();
        clock_t end     = clock();
        auto grad       = OC.getAnalyticGradient(control,false);


        // Save extended Data

        // Create an output string stream
        std::ostringstream streamObj2;
        // Set Fixed -Point Notation
        streamObj2 << std::fixed;
        // Set precision to 1 digits
        streamObj2 << std::setprecision(0);
        //Add double to stream
        streamObj2 << maxM;

        std::string filename = "TimeEvolBondDimT" + streamObj1.str() + "maxM" + streamObj2.str() + ".txt";
        std::ofstream myfile (filename);
        size_t ind = 0;

        if (myfile.is_open())
        {
            for (auto& pt : psi)
            {
                myfile << times.at(ind)     << "\t";
                myfile << control.at(ind)   << "\t";
                myfile << fid.at(ind)       << "\t";
                myfile << grad.at(ind)      << "\t";
                
                for(int b = 1; b < pt.N(); ++b) 
                {
                    myfile << linkInd(pt,b).m() << "\t"; 
                }

                ind++;
                myfile << "\n";
            }
            myfile.close();
        }
        else std::cout << "Unable to open file\n";

        double rt = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "Runtime = " << rt << "\n";
        runtimes.push_back(rt);        
    } // foreach maxBondDim


    // save runtimes
    std::string filename = "TimeEvolBondDimT" + streamObj1.str() + "runtimes.txt";
    std::ofstream myfile (filename);

    if (myfile.is_open())
    {
        for(size_t i = 0; i < runtimes.size(); i++)
        {
            myfile << maxBondDim.at(i)  << "\t";
            myfile << runtimes.at(i)    << "\n";
        }
        myfile.close();
    }
    else std::cout << "Unable to open file\n";


    // bond dims of initial and target state
    std::ofstream myfile2 ("DMRGstateBondDim.txt");
    if (myfile2.is_open())
    { 
        for(int b = 1; b < psi_i.N(); ++b) 
        {
            myfile2 << linkInd(psi_i,b).m() << "\t";
            myfile2 << linkInd(psi_f,b).m() << "\n"; 
        }
        myfile2.close();
    }
    else std::cout << "Unable to open file\n";

    
    return 0;
}
