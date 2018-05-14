#include <gtest/gtest.h>
#include "itensor/all.h"

#include "../include/BH_sites.h"
#include "../include/correlations.hpp"


using namespace itensor;

struct twoSiteCorrelationTest : testing::Test {
  SiteSet* sites;
  IQMPS* psi;

  twoSiteCorrelationTest() {
    int N = 2;
    int d = 5;

    sites = new BoseHubbard(N,d);
    auto state = InitState(*sites);
    state.set(1,"Occ1");
    state.set(2,"Occ2");

    psi = new IQMPS(state);
  }

  ~twoSiteCorrelationTest() {
    delete sites;
    delete psi;
  }
};

struct operatorTest : testing::Test {
  SiteSet* sites;
  IQMPS* psi;
  IQMPO* N_opp;
  IQMPO* A_opp;
  IQMPO* Adag_opp;

  operatorTest() {
    int N = 2;
    int d = 5;

    sites = new BoseHubbard(N,d);
    auto state = InitState(*sites);
    state.set(1,"Occ2");
    state.set(2,"Occ2");

    psi = new IQMPS(state);

    auto ampoN = AutoMPO(*sites);
    ampoN += 1,"N",1;
    N_opp = new IQMPO(ampoN);

    auto ampoAdag = AutoMPO(*sites);
    ampoAdag += 1,"Adag",1,"A",2;
    Adag_opp = new IQMPO(ampoAdag);

    auto ampoA = AutoMPO(*sites);
    ampoA += 1,"A",1,"Adag",2;
    A_opp = new IQMPO(ampoA);

  }

  ~operatorTest() {
    delete sites;
    delete psi;
    delete N_opp;
    delete A_opp;
    delete Adag_opp;
  }
};


struct singleParticleDensityMatrixTest : testing::Test {
  SiteSet* sites;
  IQMPS* psi;

  singleParticleDensityMatrixTest(){ }
  ~singleParticleDensityMatrixTest() {
    delete sites;
    delete psi;
  }

  void setupMott(int fillingFraction, int N){
    int Npart = fillingFraction*N;
    int d = fillingFraction*2;

    sites = new BoseHubbard(N,d);
    auto state = InitState(*sites);

    for (size_t i = 1; i <= N; i++) {
        state.set(i,nameint("Occ",fillingFraction));
    }

    psi = new IQMPS(state);

  }

};

TEST_F(operatorTest, testParticleNumber){
  auto args = Args("Cutoff=",1E-9,"Maxm=",10);

  ASSERT_DOUBLE_EQ( 2.0, overlap(*psi,*N_opp,*psi));

  // raise particle number by 1, EVAL N, lower by 1
  *psi = exactApplyMPO(*Adag_opp,*psi,args);
  normalize(*psi);
  ASSERT_DOUBLE_EQ( 3.0, overlap(*psi,*N_opp,*psi));
  *psi = exactApplyMPO(*A_opp,*psi,args);
  normalize(*psi);

  // lower particle number by 1, EVAL N, raise by 1
  *psi = exactApplyMPO(*A_opp,*psi,args);
  normalize(*psi);
  ASSERT_DOUBLE_EQ( 1.0, overlap(*psi,*N_opp,*psi));
  *psi = exactApplyMPO(*Adag_opp,*psi,args);
  normalize(*psi);
}


TEST_F(twoSiteCorrelationTest, testBosonSiteWithOcc1) {
    int site1 = 1;
    int site2 = 1;

    ASSERT_EQ( (Cplx) 1, correlationFunction(*sites,*psi,"N",site1,"N",site2));
    ASSERT_EQ( (Cplx) 0, correlationFunction(*sites,*psi,"Id",site1,"N(N-1)",site2));
    ASSERT_EQ( (Cplx) 0, correlationFunction(*sites,*psi,"N(N-1)",site1,"Id",site2));

    ASSERT_EQ( (Cplx) 0, correlationFunction(*sites,*psi,"Adag",site1,"Adag",site2));
    ASSERT_EQ( (Cplx) 0, correlationFunction(*sites,*psi,"A",site1,"A",site2));
    ASSERT_EQ( (Cplx) 1, correlationFunction(*sites,*psi,"Adag",site1,"A",site2));
    ASSERT_EQ( (Cplx) 2, correlationFunction(*sites,*psi,"A",site1,"Adag",site2));

    ASSERT_EQ( (Cplx) 0, correlationFunction(*sites,*psi,"Adag",site1,"N",site2));
    ASSERT_EQ( (Cplx) 0, correlationFunction(*sites,*psi,"A",site1,"N",site2));
}

TEST_F(twoSiteCorrelationTest, testBosonSiteWithOcc2) {
    int site1 = 2;
    int site2 = 2;

    ASSERT_EQ( (Cplx) 4.0, correlationFunction(*sites,*psi,"N",site1,"N",site2));
    ASSERT_EQ( (Cplx) 2.0, correlationFunction(*sites,*psi,"Id",site1,"N(N-1)",site2));
    ASSERT_EQ( (Cplx) 2.0, correlationFunction(*sites,*psi,"N(N-1)",site1,"Id",site2));

    ASSERT_EQ( (Cplx) 0.0, correlationFunction(*sites,*psi,"Adag",site1,"Adag",site2));
    ASSERT_EQ( (Cplx) 0.0, correlationFunction(*sites,*psi,"A",site1,"A",site2));
    ASSERT_EQ( (Cplx) 2.0, correlationFunction(*sites,*psi,"Adag",site1,"A",site2));
    ASSERT_DOUBLE_EQ( 3, real(correlationFunction(*sites,*psi,"A",site1,"Adag",site2)));

    ASSERT_EQ( (Cplx) 0.0, correlationFunction(*sites,*psi,"Adag",site1,"N",site2));
    ASSERT_EQ( (Cplx) 0.0, correlationFunction(*sites,*psi,"A",site1,"N",site2));
}

TEST_F(twoSiteCorrelationTest, testBosonSitesWithOcc1andOcc2) {
    int site1 = 1;
    int site2 = 2;

    ASSERT_EQ( (Cplx) 2, correlationFunction(*sites,*psi,"N",site1,"N",site2));
    ASSERT_EQ( (Cplx) 2, correlationFunction(*sites,*psi,"Id",site1,"N(N-1)",site2));
    ASSERT_EQ( (Cplx) 0, correlationFunction(*sites,*psi,"N(N-1)",site1,"Id",site2));

    ASSERT_EQ( (Cplx) 0.0, correlationFunction(*sites,*psi,"Adag",site1,"Adag",site2));
    ASSERT_EQ( (Cplx) 0.0, correlationFunction(*sites,*psi,"A",site1,"A",site2));
    ASSERT_EQ( (Cplx) 0, correlationFunction(*sites,*psi,"Adag",site1,"A",site2));
    ASSERT_EQ( (Cplx) 0, correlationFunction(*sites,*psi,"A",site1,"Adag",site2));

    ASSERT_EQ( (Cplx) 0.0, correlationFunction(*sites,*psi,"Adag",site1,"N",site2));
    ASSERT_EQ( (Cplx) 0.0, correlationFunction(*sites,*psi,"A",site1,"N",site2));
}

TEST_F(singleParticleDensityMatrixTest, testMott){
  for (size_t i = 1; i < 5; i++) {
    for (size_t j = 5; j < 25; j=j+5) {
      setupMott(i,j);
      ASSERT_DOUBLE_EQ(i, correlationTerm(*sites,*psi,"Adag","A"));
    }
  }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
