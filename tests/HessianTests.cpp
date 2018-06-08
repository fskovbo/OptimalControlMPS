#include <gtest/gtest.h>
#include<fstream>
#include<iostream>
#include "itensor/all.h"

#include "../include/BH_sites.h"
#include "../include/BH_tDMRG.hpp"
#include "../include/OptimalControl.hpp"
#include "../include/InitializeState.hpp"
#include "../include/ControlBasisFactory.hpp"


using namespace itensor;

using rowmat = std::vector< std::vector<double>>;

struct HessianTest : testing::Test
{   
    OptimalControl<BH_tDMRG>* OC_GRAPE;
    OptimalControl<BH_tDMRG>* OC_GROUP;
    
    int N, M;

    HessianTest()
    {
        int L           = 5;
        int Npart       = 5;
        int locDim      = 5;

        double J        = 1.0;
        double cstart   = 2.0;
        double cend     = 12.0;
        double T        = 0.15;
        double tstep    = 1e-2;
        N               = T/tstep + 1;
        M               = 10;

        srand((unsigned)time(NULL));


        auto sites      = BoseHubbard(L,locDim);
        auto psi_i      = InitializeState(sites,Npart,J,cstart);
        auto psi_f      = InitializeState(sites,Npart,J,cend);
        auto tDMRG      = BH_tDMRG(sites,J,tstep,{"Cutoff=",1E-8});
        auto u0         = linseed(cstart,cend,N);
        auto basis      = ControlBasisFactory::buildChoppedSineBasis(u0,tstep,T,M);  

        OC_GRAPE        = new OptimalControl<BH_tDMRG>(psi_f,psi_i,tDMRG,N,0);
        OC_GROUP        = new OptimalControl<BH_tDMRG>(psi_f,psi_i,tDMRG,basis,0);
    }
    ~HessianTest()
    {
        delete OC_GRAPE;
        delete OC_GROUP;        
    }

    std::vector<double> linseed(double a, double b, int n)
    {
        std::vector<double> array;
        double step = (b-a) / (n-1);

        while(a <= b + 1e-7) {
            array.push_back(a);
            a += step;           // could recode to better handle rounding errors
        }
        return array;
    }

    std::vector<double> sinseed(double a, double b, int n)
    {
        auto lin = linseed(a,b,n);
        
        for (auto& val : lin){
            val = sin(val);
        }
        
        return lin;
    }

    std::vector<double> expseed(double a, double b, int n)
    {
        auto lin = linseed(a,b,n);
        
        for (auto& val : lin){
            val = exp(val);
        }
        
        return lin;
    }

    double fRand(double fMin, double fMax)
    {
        double f = (double)rand() / RAND_MAX;
        return fMin + f * (fMax - fMin);
    }

    std::vector<double> randseed(double min, double max, int n)
    {
        std::vector<double> vec;

        for(size_t i = 0; i < n; i++)
        {
            vec.push_back(fRand(min, max));
        }
        
        return vec;
    }

    std::vector<double> getNumericGrad(std::vector<double>& control, OptimalControl<BH_tDMRG>& OCBH)
    {
        double Jp, Jm, epsilon = 1e-5;
        std::vector<double> g;
        g.reserve(control.size());

        size_t count = 0;

        for (auto& ui : control){
            ui        += epsilon;
            Jp         = OCBH.getCost(control);

            ui        -= 2.0*epsilon;
            Jm         = OCBH.getCost(control);

            ui        += epsilon;
            g.push_back((Jp-Jm)/(2.0*epsilon));
        }
        
        return g;
    }

    rowmat getNumericHessianForward(std::vector<double>& control, OptimalControl<BH_tDMRG>& OCBH)
    {
        rowmat numHessian(N, std::vector<double>(N, 0));

        double epsilon = 1e-3;

        double fx = OCBH.getCost(control);
        std::vector<double> feps;

        for(size_t i=0;i<N;++i){
            control[i] += epsilon;
            feps.push_back(OCBH.getCost(control));
            control[i] -= epsilon;
        }

        for(size_t i=0;i<N;++i){
            for(size_t j=i;j<N;++j){
                control[i] += epsilon;
                control[j] += epsilon;
                double fepsiepsj = OCBH.getCost(control);

                numHessian[i][j]=(fepsiepsj-feps[i]-feps[j]+fx)/(epsilon*epsilon);
                numHessian[j][i] = numHessian[i][j];
                control[i] -= epsilon;
                control[j] -= epsilon;
            }
        }
        return numHessian;
    }
  
};


TEST_F(HessianTest, testGRAPE)
{
    // Test GRAPE fidelity gradient
    auto control            = randseed(2,10,N);
    auto analyticHessian    = OC_GRAPE->getHessian(control);
    auto numHessian         = getNumericHessianForward(control,*OC_GRAPE);

    ASSERT_EQ( numHessian.size() , analyticHessian.size() );
    ASSERT_EQ( numHessian.front().size() , analyticHessian.front().size() );
    ASSERT_EQ( analyticHessian.size() , N );
    ASSERT_EQ( analyticHessian.front().size() , N );
    
    
    for(size_t i = 1; i < N-1; i++)
    {
        for(size_t j = 1; j < N-1; j++)
        {
            EXPECT_NEAR(numHessian[i][j], analyticHessian[i][j], fabs(numHessian[i][j])*5e-3); 
        }
    }
    


//    // Test GRAPE regularization gradient
//    OC_GRAPE->setGamma(1);

//    analytic   = OC_GRAPE->getAnalyticGradient(control,false);
//    numeric    = getNumericGrad(control,*OC_GRAPE);

//    ASSERT_EQ( analytic.size() , numeric.size() );
    
//    for(size_t i = 1; i < analytic.size()-1; i++)
//    {
//        // expected to vary by max 0.001%
//        EXPECT_NEAR(analytic.at(i) , numeric.at(i), fabs(numeric.at(i)*1e-5));
//    }
}


//TEST_F(gradientTest, testGROUP)
//{
//    // Test GROUP fidelity gradient
//    auto control    = randseed(-4,4,M);
//    auto numeric    = getNumericGrad(control,*OC_GROUP);
//    auto analytic   = OC_GROUP->getAnalyticGradient(control);

//    ASSERT_EQ( analytic.size() , numeric.size() );
    
//    for(size_t i = 1; i < analytic.size()-1; i++)
//    {
//        // expected to vary by max 0.1%
//        EXPECT_NEAR(analytic.at(i) , numeric.at(i), fabs(numeric.at(i)*1e-3));
//    }

//    // Test GROUP regularization gradient
//    OC_GROUP->setGamma(1);

//    analytic   = OC_GROUP->getAnalyticGradient(control,false);
//    numeric    = getNumericGrad(control,*OC_GROUP);

//    ASSERT_EQ( analytic.size() , numeric.size() );
    
//    for(size_t i = 1; i < analytic.size()-1; i++)
//    {
//        // expected to vary by max 0.001%
//        EXPECT_NEAR(analytic.at(i) , numeric.at(i), fabs(numeric.at(i)*1e-5));
//    }
//}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
