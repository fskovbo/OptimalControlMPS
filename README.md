Notes regarding the structure of the code:
This library uses functions from the ITensor library for tensor computations.

The core functionality lies in the class OptimalControl, which calculates a cost function along with its first-order derivatives (gradient) and second-order derivatives (Hessian).
The OptimalControl class has a BFGS-setting which is toggled 'false' as default. If BFGS=true, a 3rd-party approximation of the Hessian will be made, and the getHessian() method should not be called, hence certain vectors are not allocated to save memory.
Furthermore, if threadCount>1, parallel computations of states Psi and Xi along with the Hessian rows will be used. Currently the parallel computations are slow due to a large overhead. 
When enabling BFGS=true, only sequencial computations will be performed.
The methods of the OptimalControl class take a boolean value 'new_control', which indicates whether any of the other methods have been called using the same control. This is very useful, as getCost(), getAnalyticalGradient(), and getHessian() all utilize data from the vector psi_t, whereby the vector should only be computed once for a given control vector. If new_control=true, no other methods have been called for the given control, whereby vectors such as psi_t should be calculated.

The OptimalControl class utilize two other classes:
  1. A TimeStepper for propagating the states for a given Hamiltonian.
  2. A ControlBasis for translating the control and derivatives into another basis.
  
These 3 classes are unit tested:
  ControlBasisTest tests that ControlBasis translates between bases correctly.
  CostTests test that the costs and fidelities are correctly calculated (this is also a test of TimeStepper).
  SequencingTests test that the right data is calculating when utilizing the 'new_control' flag
  GradientTests test the gradients for both BFGS-settings and GRAPE and GROUP parameterization.
  HessianTests test the Hessian for both GRAPE and GROUP parameterization.
  
NOTE: the gradient and Hessian tests compare numerical and analytical derivatives for randomly generated controls. As these are often very sensitive, the tests may fail due to the numerical approximation being poor. TYPICALLY ~4 ENTRIES IN A SINGLE TEST ARE OFF. Nevertheless, the test can still be considered passed. For good measure test twice, to see if the entries keep failing. Tests may take 5 minutes to complete.


The optimization problem is described in the BH_nlp class. This class is interfaced according to the IPOPT library, and supplies the interior-point optimization algorithm with the costs and derivatives calculated in the OptimalControl class.


Executable files are located in the /main folder. Many exec take an inputfile with optimization/MPS parameters.
Example script for creating inputfile:

  (
    echo input
    echo {
    echo tstep = 0.01
    echo T = 2.0
    echo N = 5
    echo Npart = 5
    echo d = 4
    echo M = 10
    echo gamma = 1e-6
    echo cacheProgress = no
    echo maxBondDim = 80
    echo threshold = 1e-8
    echo optTol = 1e-8
    echo }
  ) > InputFile_BHcontrolT2.0
