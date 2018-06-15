#include "BH_nlp.hpp"

// constructor
BH_nlp::BH_nlp(OC_BH& optControlProb, bool cacheProgress)
 : optControlProb(optControlProb), cacheProgress(cacheProgress)
 {
   times = optControlProb.getTimeAxis();
 }

//destructor
BH_nlp::~BH_nlp()
{
}

bool BH_nlp::get_nlp_info(Ipopt::Index& n, Ipopt::Index& m, Ipopt::Index& nnz_jac_g,
                             Ipopt::Index& nnz_h_lag, IndexStyleEnum& index_style)
{
  // The problem described has M variables
  n = optControlProb.getM();

  // N inequality constraints for Umax and Umin each
  m = optControlProb.getN();

  // in this example the Jacobian is dense and contains m*n = N*M nonzeros
  nnz_jac_g = m*n;

  // // the Hessian is approximated using L-BFGS
  // nnz_h_lag = 0;

  // the Hessian is also dense and has n*n total nonzeros, but we
  // only need the lower left corner (since it is symmetric)
  nnz_h_lag = (n*n+n)/2;

  // use the C style indexing (0-based)
  index_style = TNLP::C_STYLE;

  return true;
}

bool BH_nlp::get_bounds_info(Ipopt::Index n, Number* x_l, Number* x_u,
                                Ipopt::Index m, Number* g_l, Number* g_u)
{
  // here, the n and m we gave IPOPT in get_nlp_info are passed back to us.
  // If desired, we could assert to make sure they are what we think they are.

  // lower bounds of the variable
  for (Ipopt::Index i = 0; i < n; i++)
    x_l[i] = -10;

  // upper bounds of the variables
  for (Ipopt::Index i = 0; i < n; i++)
    x_u[i] = 10;

  // constraint functions here are limits on GRAPE control
  double Umin = 2.0;
  double Umax = 100;
  for (Ipopt::Index i = 0; i < m; i++) {
    g_l[i] = Umin;
    g_u[i] = Umax;
  }

  return true;
}

bool BH_nlp::get_starting_point(Ipopt::Index n, bool init_x, Number* x,
                                   bool init_z, Number* z_L, Number* z_U,
                                   Ipopt::Index m, bool init_lambda,
                                   Number* lambda)
{
  // Here, we assume we only have starting values for x, if you code
  // your own NLP, you can provide starting values for the dual variables
  // if you wish to use a warmstart option
  assert(init_x == true);
  assert(init_z == false);
  assert(init_lambda == false);

  // initialize to the given starting point - here 0
  // store initial coefficients for later
  for (Ipopt::Index i = 0; i < n; i++)
  {
    x[i] = 0;
  }
  initialCoeffs = std::vector<double>(x, x + n);

  return true;
}

bool BH_nlp::eval_f(Ipopt::Index n, const Number* x, bool new_x, Number& obj_value)
{
  std::vector<double> control(x, x + n);
  obj_value = optControlProb.getCost(control,new_x);

  return true;
}

bool BH_nlp::eval_grad_f(Ipopt::Index n, const Number* x, bool new_x, Number* grad_f)
{
  std::vector<double> control(x, x + n);
  auto grad = optControlProb.getAnalyticGradient(control,new_x);
  std::copy(grad.begin(), grad.end(), grad_f);

  return true;
}

bool BH_nlp::eval_g(Ipopt::Index n, const Number* x, bool new_x, Ipopt::Index m, Number* g)
{
  std::vector<double> control(x, x + n);

  if (new_x){
    // must calculate psi_t for other eval_* functions if new_x
    optControlProb.propagatePsi(control);
  }

  auto tcontrol = optControlProb.getControl(control);
  std::copy(tcontrol.begin(), tcontrol.end(), g);

  return true;
}

bool BH_nlp::eval_jac_g(Ipopt::Index n, const Number* x, bool new_x,
                           Ipopt::Index m, Ipopt::Index nele_jac, Ipopt::Index* iRow, Ipopt::Index *jCol,
                           Number* values)
{
  if (new_x){
    // must calculate psi_t for other eval_* functions if new_x
    std::vector<double> control(x, x + n);
    optControlProb.propagatePsi(control);
  }

  if (values == NULL) {
    // return the structure of the Jacobian
    // this particular Jacobian is dense
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
        iRow[n*i+j] = i;
        jCol[n*i+j] = j;
      }
    }
  }
  else {
    // return the values of the Jacobian of the constraints
    // data format is vector of vectors (row matrix)
    auto tJac = optControlProb.getControlJacobian();

    // TODO: constraint Jac is constant -> store here and copy reference to values
    size_t ind = 0;
    for(auto& row : tJac)
    {
      for (auto& val : row)
      {
        values[ind++] = val;
      }
    }    

  }

  return true;
}

bool BH_nlp::eval_h(Ipopt::Index n, const Number* x, bool new_x,
                    Number obj_factor, Ipopt::Index m, const Number* lambda,
                    bool new_lambda, Ipopt::Index nele_hess, Ipopt::Index* iRow,
                    Ipopt::Index* jCol, Number* values)
{
  if (values == NULL)
  {
    // return the structure. This is a symmetric matrix, fill the lower left
    // triangle only.
    // the Hessian for this problem is actually dense
    Ipopt::Index idx = 0;
    for (Ipopt::Index row = 0; row < n; row++)
    {
      for (Ipopt::Index col = 0; col <= row; col++)
      {
        iRow[idx] = row;
        jCol[idx] = col;
        idx++;
      }
    }

    assert(idx == nele_hess);
  }
  else
  {
    // return the values. This is a symmetric matrix, fill the lower left
    // triangle only
    // fill the objective portion

    std::vector<double> control(x, x + n);
    auto hess = optControlProb.getHessian(control);

    Ipopt::Index idx = 0;
    for (Ipopt::Index row = 0; row < n; row++)
    {
      for (Ipopt::Index col = 0; col <= row; col++)
      {
        values[idx] = obj_factor * hess[row][col];
        idx++;
      }
    }

    // constraint Hessian is zero
  }
  
  return true;
}

void BH_nlp::finalize_solution(SolverReturn status,
                                  Ipopt::Index n, const Number* x, const Number* z_L,
                                  const Number* z_U, Ipopt::Index m, const Number* g,
                                  const Number* lambda, Number obj_value,
                                  const IpoptData* ip_data, IpoptCalculatedQuantities* ip_cq)
{
  // here is where we would store the solution to variables, or write to a file, etc
  // so we could use the solution.

  // For this example, we write the solution to the console
  printf("\n\nSolution of the primal variables, x\n");
  for (Ipopt::Index i=0; i<n; i++) {
    printf("x[%d] = %e\n", i, x[i]);
  }

  printf("\n\nSolution of the bound multipliers, z_L and z_U\n");
  for (Ipopt::Index i=0; i<n; i++) {
    printf("z_L[%d] = %e\n", i, z_L[i]);
  }
  for (Ipopt::Index i=0; i<n; i++) {
    printf("z_U[%d] = %e\n", i, z_U[i]);
  }

  printf("\n\nObjective value\n");
  printf("f(x*) = %e\n", obj_value);


  // write initial and final control to file
  std::vector<double> finalCoeffs(x, x + n);

  auto initialControl     = optControlProb.getControl(initialCoeffs);
  auto finalControl       = optControlProb.getControl(finalCoeffs);
  auto initialFidelities  = optControlProb.getFidelityForAllT(initialCoeffs);
  auto finalFidelities    = optControlProb.getFidelityForAllT(finalCoeffs);

  std::string filename = "BHrampInitialFinal.txt";
  std::ofstream myfile (filename);
  if (myfile.is_open())
  {
    for (int i = 0; i < m; i++) {
      myfile << times.at(i) << "\t";
      myfile << initialControl.at(i) << "\t";
      myfile << initialFidelities.at(i) << "\t";
      myfile << finalControl.at(i) << "\t";
      myfile << finalFidelities.at(i) << "\n";
    }
    myfile.close();
  }
  else std::cout << "Unable to open file\n";
}

bool BH_nlp::intermediate_callback(AlgorithmMode mode,
                                              Ipopt::Index iter, Number obj_value,
                                              Number inf_pr, Number inf_du,
                                              Number mu, Number d_norm,
                                              Number regularization_size,
                                              Number alpha_du, Number alpha_pr,
                                              Ipopt::Index ls_trials,
                                              const IpoptData* ip_data,
                                              IpoptCalculatedQuantities* ip_cq)
{
  // save current control and cost to file
  if (cacheProgress) {

    std::ofstream outfile;
    std::string filename = "ProgressCache.txt";
    outfile.open(filename, std::ios_base::app);
    if (outfile.is_open())
    {
      outfile << iter << "\t";
      outfile << obj_value << "\t";
      outfile << times.back() << "\t";
      outfile << 2 + ls_trials << "\n";

    }
    else std::cout << "Unable to open file\n";

  }

  return true;
}
