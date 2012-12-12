/** 
* @file completeEvaluationFastMinkernel.cpp
* @brief Demo-Program to show how to call some methods of the FastMinKernel class
* @author Alexander Freytag
* @date 12/08/2011
*/

#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>

#include "core/vector/MatrixT.h"
#include "core/vector/VectorT.h"
#include "gp-hik-core/FastMinKernel.h"
#include "gp-hik-core/tools.h"
#include "gp-hik-core/VectorSorter.h"
#include "gp-hik-core/FeatureMatrixT.h"
#include "gp-hik-core/kernels/IntersectionKernelFunction.h"


using namespace std;
using namespace NICE;

/**
 * @brief Printing main menu.
 * @author Alexander Freytag
 * @date 12/06/2011
 * 
 * @return void
 **/
void print_main_menu()
{
  std::cerr << "===============================================================================================================" << std::endl;
  std::cerr << "|| Welcome to the evaluation programm for our FastMinKernel class                                            ||" << std::endl;
  std::cerr << "||                                                                                                           ||" << std::endl;  
  std::cerr << "|| We will run some tests to evaluate the efficiency of our fast computations compared to the baseline ones. ||" << std::endl;
  std::cerr << "|| Note, that the benefit is larger for higher number of dimensions.                                         ||" << std::endl;
  std::cerr << "|| Note further, that we randomly sample features, wherefore the results might differ from run to run.       ||" << std::endl;
  std::cerr << "|| Finally, note that in practical applications the speed-up is larger due to sparse features.             ||" << std::endl;
  std::cerr << "===============================================================================================================" << std::endl;  
  
  
  std::cout << std::endl << "Input options:" << std::endl;
  std::cout << "   -n <number>  number of examples to generate (optional)"<< std::endl;
  std::cout << "   -d <number>  number of dimensions for each example"<< std::endl;
  std::cout << "   -v 1/0  verbose mode (optional)"<< std::endl;
  return;
}

int main (int argc, char* argv[])
{
  std::cout.precision(15);
  std::cerr.precision(15);
  
  int nEx (5);
  int d (10);
  bool verbose(false);
  bool nGiven (false);
  
  int rc;
  if (argc<2)
  {
    print_main_menu();
    return -1;
  }
  
  while ((rc=getopt(argc,argv,"n:d:v:h"))>=0)
  {
    switch(rc)
    {
      case 'n': 
      {
        nEx = atoi(optarg); 
        nGiven = true;
        break;
      }
      case 'd': d = atoi(optarg); break;
      case 'v': verbose = atoi(optarg); break;
      default: print_main_menu();
    }
  }
  
  srand ( time(NULL) );

  std::vector<int> trainingSizes; trainingSizes.clear();
  std::vector<int> dataDimensions; dataDimensions.clear();
  std::vector<float> timePreparationEfficiently; timePreparationEfficiently.clear();
  std::vector<float> timeMultiplicationEfficiently; timeMultiplicationEfficiently.clear();
  std::vector<float> timePreparationSlowly; timePreparationSlowly.clear();
  std::vector<float> timeMultiplicationSlowly; timeMultiplicationSlowly.clear();
  std::vector<float> timeKSumEfficiently; timeKSumEfficiently.clear();
  std::vector<float> timeKSumSlowly; timeKSumSlowly.clear();
  std::vector<double> errorsMultiplication; errorsMultiplication.clear();
  std::vector<double> errorsKSum; errorsKSum.clear();
  
  int lower (1000);
  int upper(10000);
  int stepSize(1000);
  if (nGiven)
  {
    lower = nEx;
    upper = nEx;
  }
  for (int n = lower; n <= upper; n+=stepSize)
  {
    if (verbose)
      std::cerr << "================================" << std::endl;

    std::cerr << "n: " << n << std::endl;
    trainingSizes.push_back(n);
    dataDimensions.push_back(d);
    
    //generate random data with specified dimensions and number of examples
    std::vector<std::vector<double> > rand_feat;
    generateRandomFeatures(d,n,rand_feat);
    
    //transpose the data structure so that it fits to our fastHIK struct
    std::vector<std::vector<double> > rand_feat_transposed (rand_feat);
    transposeVectorOfVectors(rand_feat_transposed);

    //generate random alpha vectors
    Vector alphas;
    generateRandomFeatures(n, alphas);

    //for these experiments, the noise does not matter
    double noise (0.0);
    
    //---------------- EVALUATE THE RUNTIME needed for initializing both methods (fast-hik vs baseline) ---------------------------
    
    time_t  hik_efficient_preparation_start = clock();
    FastMinKernel fastHIK ( rand_feat, noise );
    
    NICE::VVector A; 
    NICE::VVector B; 
    
    fastHIK.hik_prepare_alpha_multiplications(alphas, A, B);

    float time_hik_efficient_preparation = (float) (clock() - hik_efficient_preparation_start);
    if (verbose)
    {
      std::cerr << "Time for HIK efficient preparation: " << time_hik_efficient_preparation/CLOCKS_PER_SEC << std::endl;
    }
    
    timePreparationEfficiently.push_back(time_hik_efficient_preparation/CLOCKS_PER_SEC);
    
    //---------------- EVALUATE THE ERROR AND RUNTIME FOR MULTIPLY K \alpha (aka kernel_multiply) ---------------------------
    
    Vector beta;
    //tic
    time_t  hik_multiply_start = clock();
    fastHIK.hik_kernel_multiply(A, B, alphas, beta);
    //toc
    float time_hik_multiply = (float) (clock() - hik_multiply_start); 
    if (verbose)
    {
      std::cerr << "Time for HIK multiplication: " << time_hik_multiply/CLOCKS_PER_SEC << std::endl;
    }
    
    timeMultiplicationEfficiently.push_back(time_hik_multiply/CLOCKS_PER_SEC);

    NICE::IntersectionKernelFunction<double> hikSlow;
    //tic
    time_t  hik_slow_prepare_start = clock();
    NICE::Matrix K (hikSlow.computeKernelMatrix(rand_feat_transposed, noise));
    //toc
    float time_hik_slow_prepare = (float) (clock() - hik_slow_prepare_start); 
    if (verbose)
    {
      std::cerr << "Time for HIK slow preparation of Kernel Matrix: " << time_hik_slow_prepare/CLOCKS_PER_SEC << std::endl;
    }
    timePreparationSlowly.push_back(time_hik_slow_prepare/CLOCKS_PER_SEC);
    
    time_t  hik_slow_multiply_start = clock();
    Vector betaSlow = K*alphas;
    
    //toc
    float time_hik_slow_multiply = (float) (clock() - hik_slow_multiply_start); 
    if (verbose)
    {
      std::cerr << "Time for HIK slow multiply: " << time_hik_slow_multiply/CLOCKS_PER_SEC << std::endl;
    }
    timeMultiplicationSlowly.push_back(time_hik_slow_multiply/CLOCKS_PER_SEC);

    Vector diff = beta - betaSlow;
    double error = diff.normL2();
    
    errorsMultiplication.push_back(error);

    if (verbose)
    {
      std::cerr << "error: " << error << std::endl;
    }
    
    //---------------- EVALUATE THE ERROR AND RUNTIME FOR COMPUTING k_* \alpha (aka kernel_sum) ---------------------------
    
    Vector xstar;
    generateRandomFeatures(d, xstar);

    double kSum;    
    //tic
    time_t  hik_ksum_start = clock();
    fastHIK.hik_kernel_sum(A, B, xstar, kSum);
    //toc
    float time_hik_ksum = (float) (clock() - hik_ksum_start); 
    if (verbose)
      std::cerr << "Time for HIK efficient kSum: " << time_hik_ksum/CLOCKS_PER_SEC << std::endl;
    timeKSumEfficiently.push_back(time_hik_ksum/CLOCKS_PER_SEC);
    
    if (verbose)
    {
      std::cerr << "kSum efficiently: " << kSum << std::endl;
    }
    
    //tic
    time_t  hik_ksum_slowly_start = clock();
    std::vector<double> xstar_stl (d, 0.0);
    for (int i = 0; i < d; i++)
    {
      xstar_stl[i] = xstar[i];
    }
    
    Vector kstarSlow ( hikSlow.computeKernelVector(rand_feat_transposed, xstar_stl));
    xstar.resize(xstar_stl.size());
    for ( int i = 0 ; i < xstar.size() ; i++ )
      xstar[i] = xstar_stl[i];
    double kSumSlowly = alphas.scalarProduct(kstarSlow);
    
    //toc
    float time_hik_slowly_ksum = (float) (clock() - hik_ksum_slowly_start); 
    if (verbose)
      std::cerr << "Time for HIK slowly kSum: " << time_hik_slowly_ksum/CLOCKS_PER_SEC << std::endl;
    timeKSumSlowly.push_back(time_hik_slowly_ksum/CLOCKS_PER_SEC);
    
    if (verbose)
    {
      std::cerr << "kSumSlowly: " << kSumSlowly << std::endl;
    }
    
    
    double kSumError( fabs(kSumSlowly - kSum)); 
    errorsKSum.push_back(kSumError);
    
    if (verbose)
      std::cerr << "kSumError: " << kSumError << std::endl;
  }

  //---------------- FINAL OUTPUT ---------------------------
  std::cerr << std::endl <<  "n - d - timePreparationEfficiently - timeMultiplicationEfficiently - timePreparationSlowly - timeMultiplicationSlowly - timeKSumEfficiently - timeKSumSlowly" << std::endl;
  for (int i = 0; i < (int) trainingSizes.size(); i++)
  {
    std::cerr << trainingSizes[i] << " ";
    std::cerr << dataDimensions[i] << " ";
    std::cerr << timePreparationEfficiently[i] << " ";
    std::cerr << timeMultiplicationEfficiently[i] << " ";
    std::cerr << timePreparationSlowly[i] << " ";
    std::cerr << timeMultiplicationSlowly[i] << " ";
    std::cerr << timeKSumEfficiently[i] << " ";
    std::cerr << timeKSumSlowly[i] << " ";
    std::cerr << std::endl;
  }
  
  std::cerr << std::endl << "n - d - errorMultiplication - errorsKSum" << std::endl;
  for (int i = 0; i < (int) trainingSizes.size(); i++)
  {
    std::cerr << trainingSizes[i] << " ";
    std::cerr << dataDimensions[i] << " ";
    std::cerr << errorsMultiplication[i] << " ";
    std::cerr << errorsKSum[i] << " ";
    std::cerr << std::endl;
  }
  
  return 0;
}