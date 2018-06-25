#include <gtest/gtest.h>
#include "itensor/all.h"

#include "../include/BH_sites.h"
#include "../include/BH_tDMRG.hpp"
#include "../include/OptimalControl.hpp"
#include "../include/InitializeState.hpp"
#include "../include/ControlBasisFactory.hpp"


using namespace itensor;


struct SequencingTest : testing::Test
{   
    OptimalControl<BH_tDMRG>* OC_GRAPE;
    
    int N;
    double init_cost;
    std::vector<double> init_control, init_grad;
    std::vector< std::vector<double> > init_Hess;

    SequencingTest()
    {
        int L           = 3;
        int Npart       = 3;
        int locDim      = 3;

        double J        = 2.0;
        double cstart   = 2.0;
        double cend     = 12.0;
        double T        = 0.5;
        double tstep    = 1e-2;
        N               = T/tstep + 1;

        auto sites      = BoseHubbard(L,locDim);
        auto psi_i      = InitializeState(sites,Npart,J,cstart);
        auto psi_f      = InitializeState(sites,Npart,J,cend);
        auto tDMRG      = BH_tDMRG(sites,J,tstep,{"Cutoff=",1E-7}); 

        OC_GRAPE        = new OptimalControl<BH_tDMRG>(psi_f,psi_i,tDMRG,N,0);

        srand((unsigned)time(NULL));
        
        init_control    = randseed(5,15,N);
        init_cost       = OC_GRAPE->getCost(init_control,true);
        init_grad       = OC_GRAPE->getAnalyticGradient(init_control,true);
        init_Hess       = OC_GRAPE->getHessian(init_control,true);
    }
    ~SequencingTest()
    {
        delete OC_GRAPE;
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

    bool compareCosts(const double& cost1, const double& cost2 )
    {
        return fabs(cost1-cost2) < 1e-10;
    }

    bool compareGradients(const std::vector<double>& grad1, const std::vector<double>& grad2)
    {        
        for(size_t i = 0; i < grad1.size(); i++)
        {
            if (fabs(grad1[i]-grad2[i]) > 1e-10) return false;
        }
        
        return true;
    }

    bool compareHessians(const std::vector< std::vector<double> >& hess1, const std::vector< std::vector<double> >& hess2)
    {
        for(size_t i = 0; i < hess1.size(); i++)
        {
            for(size_t j = 0; j < hess1.front().size(); j++)
            {
                if (fabs(hess1[i][j]-hess2[i][j]) > 1e-10) return false;
            }
        }
        
        return true;
    }
    
};


TEST_F(SequencingTest, testSameControl_CostGradHess)
{
    auto same_cost  = OC_GRAPE->getCost(init_control,true);
    auto same_grad  = OC_GRAPE->getAnalyticGradient(init_control,false);
    auto same_Hess  = OC_GRAPE->getHessian(init_control,false);

    ASSERT_TRUE( compareCosts(init_cost,same_cost) );
    ASSERT_TRUE( compareGradients(init_grad,same_grad) );
    ASSERT_TRUE( compareHessians(init_Hess,same_Hess) );

    // Toggle BFGS-mode ON
    OC_GRAPE->setBFGS(true);

    same_cost  = OC_GRAPE->getCost(init_control,true);
    same_grad  = OC_GRAPE->getAnalyticGradient(init_control,false);

    ASSERT_TRUE( compareCosts(init_cost,same_cost) );
    ASSERT_TRUE( compareGradients(init_grad,same_grad) );
}

TEST_F(SequencingTest, testSameControl_GradCostHess)
{
    auto same_grad  = OC_GRAPE->getAnalyticGradient(init_control,true);
    auto same_cost  = OC_GRAPE->getCost(init_control,false);
    auto same_Hess  = OC_GRAPE->getHessian(init_control,false);

    ASSERT_TRUE( compareCosts(init_cost,same_cost) );
    ASSERT_TRUE( compareGradients(init_grad,same_grad) );
    ASSERT_TRUE( compareHessians(init_Hess,same_Hess) );

    // Toggle BFGS-mode ON
    OC_GRAPE->setBFGS(true);

    same_grad  = OC_GRAPE->getAnalyticGradient(init_control,true);
    same_cost  = OC_GRAPE->getCost(init_control,false);

    ASSERT_TRUE( compareCosts(init_cost,same_cost) );
    ASSERT_TRUE( compareGradients(init_grad,same_grad) );
}

TEST_F(SequencingTest, testSameControl_CostHessGrad)
{
    auto same_cost  = OC_GRAPE->getCost(init_control,true);
    auto same_Hess  = OC_GRAPE->getHessian(init_control,false);
    auto same_grad  = OC_GRAPE->getAnalyticGradient(init_control,false);

    ASSERT_TRUE( compareCosts(init_cost,same_cost) );
    ASSERT_TRUE( compareGradients(init_grad,same_grad) );
    ASSERT_TRUE( compareHessians(init_Hess,same_Hess) );
}

TEST_F(SequencingTest, testSameControl_GradHessCost)
{
    auto same_grad  = OC_GRAPE->getAnalyticGradient(init_control,true);
    auto same_Hess  = OC_GRAPE->getHessian(init_control,false);
    auto same_cost  = OC_GRAPE->getCost(init_control,false);

    ASSERT_TRUE( compareCosts(init_cost,same_cost) );
    ASSERT_TRUE( compareGradients(init_grad,same_grad) );
    ASSERT_TRUE( compareHessians(init_Hess,same_Hess) );
}

TEST_F(SequencingTest, testSameControl_HessGradCost)
{
    auto same_Hess  = OC_GRAPE->getHessian(init_control,true);
    auto same_grad  = OC_GRAPE->getAnalyticGradient(init_control,false);
    auto same_cost  = OC_GRAPE->getCost(init_control,false);

    ASSERT_TRUE( compareCosts(init_cost,same_cost) );
    ASSERT_TRUE( compareGradients(init_grad,same_grad) );
    ASSERT_TRUE( compareHessians(init_Hess,same_Hess) );
}

TEST_F(SequencingTest, testSameControl_HessCostGrad)
{
    auto same_Hess  = OC_GRAPE->getHessian(init_control,true);
    auto same_cost  = OC_GRAPE->getCost(init_control,false);
    auto same_grad  = OC_GRAPE->getAnalyticGradient(init_control,false);

    ASSERT_TRUE( compareCosts(init_cost,same_cost) );
    ASSERT_TRUE( compareGradients(init_grad,same_grad) );
    ASSERT_TRUE( compareHessians(init_Hess,same_Hess) );
}

TEST_F(SequencingTest, testNewControl_Cost)
{
    auto new_control = randseed(2,20,N);
    auto new_cost    = OC_GRAPE->getCost(new_control,true);
    
    ASSERT_FALSE( compareCosts(init_cost,new_cost) );

    // Toggle BFGS-mode ON
    OC_GRAPE->setBFGS(true);

    auto new_cost2   = OC_GRAPE->getCost(new_control,true);
    ASSERT_FALSE( compareCosts(init_cost,new_cost2) );
    ASSERT_TRUE( compareCosts(new_cost,new_cost2) );
}

TEST_F(SequencingTest, testNewControl_Grad)
{
    auto new_control = randseed(2,20,N);
    auto new_grad    = OC_GRAPE->getAnalyticGradient(new_control,true);
    
    ASSERT_FALSE( compareGradients(init_grad,new_grad) );

    // Toggle BFGS-mode ON
    OC_GRAPE->setBFGS(true);

    auto new_grad2   = OC_GRAPE->getAnalyticGradient(new_control,true);
    ASSERT_FALSE( compareGradients(init_grad,new_grad2) );
    ASSERT_TRUE( compareGradients(new_grad,new_grad2) );
}

TEST_F(SequencingTest, testNewControl_Hess)
{
    auto new_control = randseed(2,20,N);
    auto new_Hess    = OC_GRAPE->getHessian(new_control,true);
    
    ASSERT_FALSE( compareHessians(init_Hess,new_Hess) );
}

TEST_F(SequencingTest, testNewControl_CostCost)
{
    auto new_control  = randseed(2,20,N);    
    auto same_costF   = OC_GRAPE->getCost(new_control,false);
    auto same_costT   = OC_GRAPE->getCost(new_control,true);

    ASSERT_TRUE( compareCosts(init_cost,same_costF) );
    ASSERT_FALSE( compareCosts(same_costT,same_costF) );
}

TEST_F(SequencingTest, testNewControl_GradGrad)
{
    auto new_control  = randseed(2,20,N);    
    auto same_gradF   = OC_GRAPE->getAnalyticGradient(new_control,false);
    auto same_gradT   = OC_GRAPE->getAnalyticGradient(new_control,true);

    ASSERT_TRUE( compareGradients(init_grad,same_gradF) );
    ASSERT_FALSE( compareGradients(same_gradT,same_gradF) );
}

TEST_F(SequencingTest, testNewControl_HessHess)
{
    auto new_control  = randseed(1,4,N);    
    auto same_hessF   = OC_GRAPE->getHessian(new_control,false);
    auto same_hessT   = OC_GRAPE->getHessian(new_control,true);

    ASSERT_FALSE( compareHessians(init_Hess,same_hessF) );
    ASSERT_FALSE( compareHessians(same_hessT,same_hessF) );
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}