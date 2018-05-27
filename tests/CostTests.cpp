#include <gtest/gtest.h>
#include "itensor/all.h"

#include "../include/BH_sites.h"
#include "../include/BH_tDMRG.hpp"
#include "../include/OptimalControl.hpp"
#include "../include/InitializeState.hpp"
#include "../include/ControlBasisFactory.hpp"


using namespace itensor;


struct CostTest : testing::Test
{   
    OptimalControl<BH_tDMRG>* OC_GRAPE;
    OptimalControl<BH_tDMRG>* OC_GROUP;
    
    int N, M;

    CostTest()
    {
        int L           = 5;
        int Npart       = 5;
        int locDim      = 5;

        double J        = 1.0;
        double cstart   = 2.0;
        double cend     = 50.0;
        double T        = 0.1;
        double tstep    = 1e-2;
        N               = T/tstep + 1;
        M               = 5;

        srand((unsigned)time(NULL));


        auto sites      = BoseHubbard(L,locDim);
        std::cout.setstate(std::ios_base::failbit); // silences the DMRG info
        auto psi_i      = InitializeState(sites,Npart,J,cstart);
        auto psi_f      = InitializeState(sites,Npart,J,cend);
        std::cout.clear(); // clears silence
        auto tDMRG      = BH_tDMRG(sites,J,tstep,{"Cutoff=",1E-8});
        auto u0         = linspace(cstart,cend,N);
        auto basis      = ControlBasisFactory::buildChoppedSineBasis(u0,tstep,T,M);  

        OC_GRAPE        = new OptimalControl<BH_tDMRG>(psi_f,psi_i,tDMRG,N,0);
        OC_GROUP        = new OptimalControl<BH_tDMRG>(psi_f,psi_i,tDMRG,basis,0);
    }
    ~CostTest()
    {
        delete OC_GRAPE;
        delete OC_GROUP;        
    }

    std::vector<double> linspace(double a, double b, int n)
    {
        std::vector<double> array;
        double step = (b-a) / (n-1);

        while(a <= b + 1e-7) {
            array.push_back(a);
            a += step;           // could recode to better handle rounding errors
        }
        return array;
    }
};


TEST_F(CostTest, testGRAPEfidelities)
{
    auto control    = linspace(2.0,50.0,N);
    auto cost       = OC_GRAPE->getCost(control);
    auto fidelities = OC_GRAPE->getFidelityForAllT(control,false);

    // calculated using old version of program
    std::vector<double> fidres = {0.214338,	0.214325,	0.215126,	0.217281,	0.221019,	0.22621,	0.232328,	0.238484,	0.243617,	0.246862,	0.24801};

    ASSERT_EQ( fidelities.size() , N ); 
    EXPECT_NEAR(cost , 0.375995 , 1e-6);
    
    for(size_t i = 0; i < fidelities.size()-1; i++)
    {
        EXPECT_NEAR(fidelities.at(i) , fidres.at(i), 1e-6);
    }

    std::vector<double> control2(N, 1);
    auto cost2       = OC_GRAPE->getCost(control2);
    auto fidelities2 = OC_GRAPE->getFidelityForAllT(control2,false);

    // calculated using old version of program
    std::vector<double> fidres2 = {0.214338,	0.214233,	0.213919,	0.213398,	0.212672,	0.211744,	0.210618,	0.2093,	0.207796,	0.206112,	0.204256};

    ASSERT_EQ( fidelities2.size() , N ); 
    EXPECT_NEAR(cost2 , 0.397872 , 1e-6);
    
    for(size_t i = 0; i < fidelities2.size()-1; i++)
    {
        EXPECT_NEAR(fidelities2.at(i) , fidres2.at(i), 1e-6);
    }
}


TEST_F(CostTest, testGROUPfidelities)
{
    std::vector<double> control(M, 0);
    auto cost       = OC_GROUP->getCost(control);
    auto fidelities = OC_GROUP->getFidelityForAllT(control,false);

    // calculated using old version of program
    std::vector<double> fidres = {0.214338,	0.214325,	0.215126,	0.217281,	0.221019,	0.22621,	0.232328,	0.238484,	0.243617,	0.246862,	0.24801};

    ASSERT_EQ( fidelities.size() , N ); 
    EXPECT_NEAR(cost , 0.375995 , 1e-6);
    
    for(size_t i = 0; i < fidelities.size()-1; i++)
    {
        EXPECT_NEAR(fidelities.at(i) , fidres.at(i), 1e-6);
    }

    auto control2    = linspace(0,7,M);
    auto cost2       = OC_GROUP->getCost(control2);
    auto fidelities2 = OC_GROUP->getFidelityForAllT(control2,false);

    // calculated using old version of program
    std::vector<double> fidres2 = {0.214338,	0.21411,	0.216706,	0.222581,	0.229759,	0.23623,	0.242512,	0.249913,	0.256515,	0.259334,	0.259687};

    ASSERT_EQ( fidelities2.size() , N ); 
    EXPECT_NEAR(cost2 , 0.370157 , 1e-6);
    
    for(size_t i = 0; i < fidelities2.size()-1; i++)
    {
        EXPECT_NEAR(fidelities2.at(i) , fidres2.at(i), 1e-6);
    }
}


TEST_F(CostTest, testGRAPEregularization)
{
    OC_GRAPE->setGamma(1); // sets gamma = 1 (VERY HIGH)
    auto control    = linspace(2.0,50.0,N);
    auto cost       = OC_GRAPE->getCost(control);
    auto fidelities = OC_GRAPE->getFidelityForAllT(control,false);

    // calculated using old version of program
    std::vector<double> fidres = {0.214338,	0.214325,	0.215126,	0.217281,	0.221019,	0.22621,	0.232328,	0.238484,	0.243617,	0.246862,	0.24801};

    ASSERT_EQ( fidelities.size() , N ); 
    EXPECT_NEAR(cost , 11520.4 , 1e-1);
    
    for(size_t i = 0; i < fidelities.size()-1; i++)
    {
        EXPECT_NEAR(fidelities.at(i) , fidres.at(i), 1e-6);
    }

    std::vector<double> control2(N, 1);
    auto cost2       = OC_GRAPE->getCost(control2);
    auto fidelities2 = OC_GRAPE->getFidelityForAllT(control2,false);

    // calculated using old version of program
    std::vector<double> fidres2 = {0.214338,	0.214233,	0.213919,	0.213398,	0.212672,	0.211744,	0.210618,	0.2093,	0.207796,	0.206112,	0.204256};

    ASSERT_EQ( fidelities2.size() , N ); 
    EXPECT_NEAR(cost2 , 0.397872 , 1e-6);
    
    for(size_t i = 0; i < fidelities2.size()-1; i++)
    {
        EXPECT_NEAR(fidelities2.at(i) , fidres2.at(i), 1e-6);
    }
}


TEST_F(CostTest, testGROUPregularization)
{
    OC_GROUP->setGamma(1); // sets gamma = 1 (VERY HIGH)
    std::vector<double> control(M, 0);
    auto cost       = OC_GROUP->getCost(control);
    auto fidelities = OC_GROUP->getFidelityForAllT(control,false);

    // calculated using old version of program
    std::vector<double> fidres = {0.214338,	0.214325,	0.215126,	0.217281,	0.221019,	0.22621,	0.232328,	0.238484,	0.243617,	0.246862,	0.24801};

    ASSERT_EQ( fidelities.size() , N ); 
    EXPECT_NEAR(cost , 11520.4 , 1e-1);
    
    for(size_t i = 0; i < fidelities.size()-1; i++)
    {
        EXPECT_NEAR(fidelities.at(i) , fidres.at(i), 1e-6);
    }

    auto control2    = linspace(0,7,M);
    auto cost2       = OC_GROUP->getCost(control2);
    auto fidelities2 = OC_GROUP->getFidelityForAllT(control2,false);

    // calculated using old version of program
    std::vector<double> fidres2 = {0.214338,	0.21411,	0.216706,	0.222581,	0.229759,	0.23623,	0.242512,	0.249913,	0.256515,	0.259334,	0.259687};

    ASSERT_EQ( fidelities2.size() , N ); 
    EXPECT_NEAR(cost2 , 48360.2 , 1e-1
    );
    
    for(size_t i = 0; i < fidelities2.size()-1; i++)
    {
        EXPECT_NEAR(fidelities2.at(i) , fidres2.at(i), 1e-6);
    }
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}