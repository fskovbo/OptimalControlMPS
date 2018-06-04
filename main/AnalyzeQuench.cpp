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
#include <math.h>


using namespace itensor;
typedef std::vector<double> stdvec;
typedef std::vector< std::vector<double> > rowmat;

stdvec quenchRamp(double Ui, double Uf, size_t length)
{
    stdvec ramp(length, Uf);
    ramp.front() = Ui;
    return ramp;
}

stdvec expRamp(double Ui, double Uf, size_t length)
{
    double a = Ui;
    double b = log(Uf/Ui)/length;
    stdvec ramp;
    
    for(size_t i = 0; i < length; i++)
    {
        ramp.push_back( a*exp(b*i) );
    }
    return ramp;
}

stdvec optRamp(char* filename)
{
    stdvec ramp;
    std::string RampDataFile(filename);
    double tmp, u;
    std::ifstream in(RampDataFile);
    while(!in.eof())
    {
        in >> tmp >> tmp >> tmp >> u >> tmp;
        ramp.push_back(u);
    }
    return ramp;
}

void saveRowmat(rowmat data, std::string filename)
{
    std::ofstream myfile (filename);
    if (myfile.is_open())
    {
        for (auto& row : data)
        {
            for (auto& val : row)
            {
                myfile << val << "\t";
            }
            myfile << "\n";
        }
        myfile.close();
    }
    else std::cout << "Unable to open file\n";
}


int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        printfln("Usage: %s InputFile_BHcontrol",argv[0]);
        printfln("No input detected ... using standard parameters");
    }

    double tstep  = 5e-3;
    double T      = 3;

    int N         = 20;
    int Npart     = 20;
    int locDim    = 7;

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


    double J        = 1.0;
    double U_i      = 2.5;
    double U_f      = 50;
    size_t timesteps= T/tstep + 1;

    auto sites      = BoseHubbard(N,locDim);
    auto psi_i      = InitializeState(sites,Npart,J,U_i);
    auto psi_f      = InitializeState(sites,Npart,J,U_f);
    auto stepper    = BH_tDMRG(sites,J,tstep,{"Cutoff=",1E-8,"Maxm=",1000});


    //
    // set ramp type
    //
    auto ramp             = quenchRamp(U_i,U_f,timesteps);
    std::string filename1 = "EntanglementEntropies_Quench.txt";
    std::string filename2 = "SingleParticleCorr_Quench.txt";
    std::string filename3 = "DensityDensityCorr_Quench.txt";


    //
    // calculate entanglement entropy and correlations
    //
    rowmat entropies;
    rowmat SPcorrelations;
    rowmat DDcorrelations;

    // examing correlations from row 5-11
    size_t startpoint = 7;
    size_t endpoint = 13;


    //
    // run simulation
    //
    auto psi0 = psi_i;
    auto S = entanglementEntropy(sites,psi_i);
        
    stdvec SP, DD;
    for(size_t i = startpoint+1; i <= endpoint; i++)
    {
        SP.push_back( correlationFunction(sites,psi_i,"Adag",startpoint,"A",i).real() );
        DD.push_back( correlationFunction(sites,psi_i,"N",startpoint,"N",i).real() );
    }

    entropies.push_back(S);
    SPcorrelations.push_back(SP);
    DDcorrelations.push_back(DD);   

    for (size_t i = 0; i < ramp.size()-1; i++)
    {
        stepper.step(psi0,ramp[i],ramp[i+1],true);
        auto psi = psi0; // make copy to make sure psi0 is not altered
        
        //calculate correlations
        auto S = entanglementEntropy(sites,psi);
        
        stdvec SP, DD;
        for(size_t i = startpoint+1; i <= endpoint; i++)
        {
            SP.push_back( correlationFunction(sites,psi,"Adag",startpoint,"A",i).real() );
            DD.push_back( correlationFunction(sites,psi,"N",startpoint,"N",i).real() );
        }

        entropies.push_back(S);
        SPcorrelations.push_back(SP);
        DDcorrelations.push_back(DD); 

        std::cout << "Step " << i+1 << " done.\n";
    }

  

    //
    // save data
    //
    saveRowmat(entropies,filename1);
    saveRowmat(SPcorrelations,filename2);
    saveRowmat(DDcorrelations,filename3);
    
    return 0;
}
