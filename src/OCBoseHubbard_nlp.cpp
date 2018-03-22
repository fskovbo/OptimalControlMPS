#include "OCBoseHubbard_nlp.hpp"

// constructor
OCBoseHubbard_nlp::OCBoseHubbard_nlp(OC_BH& optControlProb, ControlBasis& bControl, bool cacheProgress)
 : optControlProb(optControlProb), bControl(bControl), cacheProgress(cacheProgress) {

   IDstr = random_string(8);

   std::string filename = "BHrampInitial_" + IDstr + ".txt";
   auto u_i = bControl.convControl();
   auto f_i = optControlProb.getFidelityForAllT(bControl);
   std::ofstream myfile (filename);
   if (myfile.is_open())
   {
     for (int i = 0; i < u_i.size(); i++) {
       myfile << u_i.at(i) << "\t";
       myfile << f_i.at(i) << "\n";
     }
     myfile.close();
   }
   else std::cout << "Unable to open file\n";

 }

//destructor
OCBoseHubbard_nlp::~OCBoseHubbard_nlp()
{}

bool OCBoseHubbard_nlp::get_nlp_info(Ipopt::Index& n, Ipopt::Index& m, Ipopt::Index& nnz_jac_g,
                             Ipopt::Index& nnz_h_lag, IndexStyleEnum& index_style)
{
  // The problem described has M variables
  n = bControl.getM();

  // N inequality constraints for Umax and Umin each
  m = bControl.getN();

  // in this example the Jacobian is dense and contains m*n = N*M nonzeros
  nnz_jac_g = m*n;

  // the Hessian is also dense and has 16 total nonzeros, but we
  // only need the lower left corner (since it is symmetric)
  nnz_h_lag = 0;

  // use the C style indexing (0-based)
  index_style = TNLP::C_STYLE;

  return true;
}

bool OCBoseHubbard_nlp::get_bounds_info(Ipopt::Index n, Number* x_l, Number* x_u,
                                Ipopt::Index m, Number* g_l, Number* g_u)
{
  // here, the n and m we gave IPOPT in get_nlp_info are passed back to us.
  // If desired, we could assert to make sure they are what we think they are.

  // the variables have no lower bounds
  for (Ipopt::Index i = 0; i < n; i++)
    x_l[i] = -10;

  // the variables have no upper bounds
  for (Ipopt::Index i = 0; i < n; i++)
    x_u[i] = 10;


  double Umin = 2;
  double Umax = 100;
  for (Ipopt::Index i = 0; i < bControl.getN(); i++) {
    g_l[i] = Umin;
    g_u[i] = Umax;
  }

  return true;
}

bool OCBoseHubbard_nlp::get_starting_point(Ipopt::Index n, bool init_x, Number* x,
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

  // initialize to the given starting point
  auto c = bControl.getCArray();
  std::copy(c.begin(), c.end(), x);

  return true;
}

bool OCBoseHubbard_nlp::eval_f(Ipopt::Index n, const Number* x, bool new_x, Number& obj_value)
{

  bControl.setCArray(x,n);
  obj_value = optControlProb.getCost(bControl);

  return true;
}

bool OCBoseHubbard_nlp::eval_grad_f(Ipopt::Index n, const Number* x, bool new_x, Number* grad_f)
{

  bControl.setCArray(x,n);
  auto grad = optControlProb.getAnalyticGradient(bControl);
  std::copy(grad.second.begin(), grad.second.end(), grad_f);

  return true;
}

bool OCBoseHubbard_nlp::eval_g(Ipopt::Index n, const Number* x, bool new_x, Ipopt::Index m, Number* g)
{

  bControl.convControl(g);

  return true;
}

bool OCBoseHubbard_nlp::eval_jac_g(Ipopt::Index n, const Number* x, bool new_x,
                           Ipopt::Index m, Ipopt::Index nele_jac, Ipopt::Index* iRow, Ipopt::Index *jCol,
                           Number* values)
{
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
    bControl.fmat2array(values);
  }

  return true;
}

void OCBoseHubbard_nlp::finalize_solution(SolverReturn status,
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


  // write U-basis solution to file
  std::vector<double> solution;
  solution.assign(x,x+n);
  bControl.setCArray(solution);
  auto u  = bControl.convControl();
  auto f  = optControlProb.getFidelityForAllT(bControl);

  std::string filename = "BHrampFinal_" + IDstr + ".txt";
  std::ofstream myfile (filename);
  if (myfile.is_open())
  {
    for (int i = 0; i < u.size(); i++) {
      myfile << u.at(i) << "\t";
      myfile << f.at(i) << "\n";
    }
    myfile.close();
  }
  else std::cout << "Unable to open file\n";

}

bool OCBoseHubbard_nlp::intermediate_callback(AlgorithmMode mode,
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
    std::string filename = "ControlCache_" + IDstr + ".txt";
    outfile.open(filename, std::ios_base::app);
    if (outfile.is_open())
    {
      auto control = bControl.convControl();

      for (auto& u : control) {
        outfile << u << "\t";
      }

      outfile << obj_value << "\n";
    }
    else std::cout << "Unable to open file\n";

  }

  return true;
}

std::string OCBoseHubbard_nlp::random_string( size_t length )
{
    auto randchar = []() -> char
    {
        const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
        const size_t max_index = (sizeof(charset) - 1);
        return charset[ rand() % max_index ];
    };
    std::string str(length,0);
    std::generate_n( str.begin(), length, randchar );
    return str;
}