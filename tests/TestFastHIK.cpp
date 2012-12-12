#ifdef NICE_USELIB_CPPUNIT

#include <string>
#include <exception>

#include <core/algebra/ILSConjugateGradients.h>
#include <core/algebra/GMStandard.h>
#include <core/basics/Timer.h>

#include <gp-hik-core/tools.h>
#include <gp-hik-core/kernels/IntersectionKernelFunction.h>
#include <gp-hik-core/kernels/GeneralizedIntersectionKernelFunction.h>
#include <gp-hik-core/parameterizedFunctions/ParameterizedFunction.h>
#include <gp-hik-core/parameterizedFunctions/PFAbsExp.h>

#include "TestFastHIK.h"


const bool verbose = false;
const bool verboseStartEnd = true;
const bool solveLinWithoutRand = false;
const uint n = 20;//1500;//1500;//10;
const uint d = 5;//200;//2;
const uint numBins = 11;//1001;//1001;
const uint solveLinMaxIterations = 1000;
const double sparse_prob = 0.6;
const bool smallTest = false;

using namespace NICE;
using namespace std;

CPPUNIT_TEST_SUITE_REGISTRATION( TestFastHIK );

void TestFastHIK::setUp() {
}

void TestFastHIK::tearDown() {
}

bool compareVVector(const NICE::VVector & A, const NICE::VVector & B, const double & tolerance = 10e-8)
{
  bool result(true);
  
//   std::cerr << "A.size(): " << A.size() << " B.size(): " << B.size() << std::endl;
  
  NICE::VVector::const_iterator itA = A.begin();
  NICE::VVector::const_iterator itB = B.begin();
  
  while ( (itA != A.end()) && ( itB != B.end()) )
  {
    if (itA->size() != itB->size())
    {
      result = false;
      break;
    } 
    
//     std::cerr << "itA->size(): " << itA->size() << "itB->size(): " << itB->size() << std::endl;
    for(uint i = 0; (i < itA->size()) && (i < itB->size()); i++)
    {
      if (fabs((*itA)[i] - (*itB)[i]) > tolerance)
      {
        result = false;
        break;        
      }
    }

    if (result == false)
          break;        
    itA++;
    itB++;
//     std::cerr << "foo" << std::endl;
  }
  
  return result;
}

bool compareLUTs(const double* LUT1, const double* LUT2, const int & size, const double & tolerance = 10e-8)
{
  bool result = true;
  
  for (int i = 0; i < size; i++)
  {
    if ( fabs(LUT1[i] - LUT2[i]) > tolerance)
    {
      result = false;
      std::cerr << "problem in : " << i << " / " << size << " LUT1: " << LUT1[i] << " LUT2: " << LUT2[i] << std::endl;
      break;
    }
  }
  
  return result;
}

void TestFastHIK::testKernelMultiplication() 
{
  if (verboseStartEnd)
    std::cerr << "================== TestFastHIK::testKernelMultiplication ===================== " << std::endl;
  vector< vector<double> > dataMatrix;

  generateRandomFeatures ( d, n, dataMatrix );

  int nrZeros(0);
  for ( uint i = 0 ; i < d; i++ )
  {
    for ( uint k = 0; k < n; k++ )
      if ( drand48() < sparse_prob ) 
      {
        dataMatrix[i][k] = 0.0;
        nrZeros++;
      }
  }

  if ( verbose ) {
    cerr << "data matrix: " << endl;
    printMatrix ( dataMatrix );
    cerr << endl;
  }

  double noise = 1.0;
  FastMinKernel fmk ( dataMatrix, noise );
    
  if ( (n*d)>0)
  {
    CPPUNIT_ASSERT_DOUBLES_EQUAL(fmk.getSparsityRatio(), (double)nrZeros/(double)(n*d), 1e-8);
    if (verbose)
      std::cerr << "fmk.getSparsityRatio(): " << fmk.getSparsityRatio() << " (double)nrZeros/(double)(n*d): " << (double)nrZeros/(double)(n*d) << std::endl;
  }
  
  GMHIKernel gmk ( &fmk );
  if (verbose)
    gmk.setVerbose(true); //we want to see the size of size(A)+size(B) for non-sparse vs sparse solution 
  else
    gmk.setVerbose(false); //we don't want to see the size of size(A)+size(B) for non-sparse vs sparse solution 

  Vector y ( n );
  for ( uint i = 0; i < y.size(); i++ )
    y[i] = sin(i);
 
  Vector alpha;
  
  gmk.multiply ( alpha, y );
  
  NICE::IntersectionKernelFunction<double> hikSlow;
  
  // tic
  time_t  slow_start = clock();
  std::vector<std::vector<double> > dataMatrix_transposed (dataMatrix);
  transposeVectorOfVectors(dataMatrix_transposed);
  NICE::Matrix K (hikSlow.computeKernelMatrix(dataMatrix_transposed, noise));
  //toc
  float time_slowComputation = (float) (clock() - slow_start);
  std::cerr << "Time for computing the kernel matrix without using sparsity: " << time_slowComputation/CLOCKS_PER_SEC << " s" << std::endl;  

  // tic
  time_t  slow_sparse_start = clock();

  NICE::Matrix KSparseCalculated (hikSlow.computeKernelMatrix(fmk.featureMatrix(), noise));
  //toc
  float time_slowComputation_usingSparsity = (float) (clock() - slow_sparse_start);
  std::cerr << "Time for computing the kernel matrix using sparsity: " << time_slowComputation_usingSparsity/CLOCKS_PER_SEC << " s" << std::endl;    

  if ( verbose ) 
    cerr << "K = " << K << endl;

  // check the trace calculation
  //CPPUNIT_ASSERT_DOUBLES_EQUAL( K.trace(), fmk.featureMatrix().hikTrace() + noise*n, 1e-12 );
  CPPUNIT_ASSERT_DOUBLES_EQUAL( K.trace(), fmk.featureMatrix().hikTrace() + noise*n, 1e-8 );

  // let us compute the kernel multiplication with the slow version
  Vector alpha_slow = K*y;

  if (verbose)
    std::cerr << "Sparse multiplication [alpha, alpha_slow]: " << std::endl <<  alpha << std::endl << alpha_slow << std::endl << std::endl;
  
  CPPUNIT_ASSERT_DOUBLES_EQUAL((alpha-alpha_slow).normL1(), 0.0, 1e-8);

  // test the case, where we first transform and then use the multiply stuff
  NICE::GeneralizedIntersectionKernelFunction<double> ghikSlow ( 1.2 );

  NICE::Matrix gK ( ghikSlow.computeKernelMatrix(dataMatrix_transposed, noise) );
  ParameterizedFunction *pf = new PFAbsExp( 1.2 );
  fmk.applyFunctionToFeatureMatrix( pf );
//   pf->applyFunctionToFeatureMatrix ( fmk.featureMatrix() );

  Vector galpha;
  gmk.multiply ( galpha, y );

  Vector galpha_slow = gK * y;

  CPPUNIT_ASSERT_DOUBLES_EQUAL((galpha-galpha_slow).normL1(), 0.0, 1e-8);
  if (verboseStartEnd)
    std::cerr << "================== TestFastHIK::testKernelMultiplication done ===================== " << std::endl;
}

void TestFastHIK::testKernelMultiplicationFast() 
{
  if (verboseStartEnd)
    std::cerr << "================== TestFastHIK::testKernelMultiplicationFast ===================== " << std::endl;
  
  Quantization q_gen ( numBins );
  Quantization q ( 2*numBins -1);

  // data is generated, such that there is no approximation error
  vector< vector<double> > dataMatrix;
  for ( uint i = 0; i < d ; i++ )
  {
    vector<double> v;
    v.resize(n);
    for ( uint k = 0; k < n; k++ ) {
      if ( drand48() < sparse_prob ) {
        v[k] = 0;
      } else {
        v[k] = q_gen.getPrototype( (rand() % numBins) );
      }
    }

    dataMatrix.push_back(v);
  }
  
  if ( verbose ) {
    cerr << "data matrix: " << endl;
    printMatrix ( dataMatrix );
    cerr << endl;
  }

  double noise = 1.0;
  FastMinKernel fmk ( dataMatrix, noise );
  
  GMHIKernel gmk ( &fmk );
  if (verbose)
    gmk.setVerbose(true); //we want to see the size of size(A)+size(B) for non-sparse vs sparse solution 
  else
    gmk.setVerbose(false); //we don't want to see the size of size(A)+size(B) for non-sparse vs sparse solution 

  Vector y ( n );
  for ( uint i = 0; i < y.size(); i++ )
    y[i] = sin(i);
   
  ParameterizedFunction *pf = new PFAbsExp ( 1.0 );
  GMHIKernel gmkFast ( &fmk, pf, &q );

//   pf.applyFunctionToFeatureMatrix ( fmk.featureMatrix() );
    
  Vector alpha;
  
  gmk.multiply ( alpha, y );
  
  Vector alphaFast;
  
  gmkFast.multiply ( alphaFast, y );
  
  NICE::IntersectionKernelFunction<double> hikSlow;
  
  std::vector<std::vector<double> > dataMatrix_transposed (dataMatrix);
  transposeVectorOfVectors(dataMatrix_transposed);

  NICE::Matrix K (hikSlow.computeKernelMatrix(dataMatrix_transposed, noise));

  if ( verbose ) 
    cerr << "K = " << K << endl;

  // check the trace calculation
  //CPPUNIT_ASSERT_DOUBLES_EQUAL( K.trace(), fmk.featureMatrix().hikTrace() + noise*n, 1e-12 );
  CPPUNIT_ASSERT_DOUBLES_EQUAL( K.trace(), fmk.featureMatrix().hikTrace() + noise*n, 1e-8 );

  // let us compute the kernel multiplication with the slow version
  Vector alpha_slow = K*y;

  if ( verbose )
    std::cerr << "Sparse multiplication [alpha, alphaFast, alpha_slow]: " << std::endl <<  alpha << std::endl << alphaFast << std::endl << alpha_slow << std::endl << std::endl;
 
  CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, (alphaFast-alpha_slow).normL1(), 1e-8);

  // test the case, where we first transform and then use the multiply stuff
  NICE::GeneralizedIntersectionKernelFunction<double> ghikSlow ( 1.2 );

  NICE::Matrix gK ( ghikSlow.computeKernelMatrix(dataMatrix_transposed, noise) );
  pf->parameters()[0] = 1.2;
  fmk.applyFunctionToFeatureMatrix( pf );
//   pf->applyFunctionToFeatureMatrix ( fmk.featureMatrix() );

  Vector galphaFast;
  gmkFast.multiply ( galphaFast, y );
  
  Vector galpha;
  
  gmk.multiply ( galpha, y );

  Vector galpha_slow = gK * y;
  
  if (verbose)
    std::cerr << "Sparse multiplication [galpha, galphaFast, galpha_slow]: " << std::endl <<  galpha << std::endl << galphaFast << std::endl << galpha_slow << std::endl << std::endl;

  CPPUNIT_ASSERT_DOUBLES_EQUAL((galphaFast-galpha_slow).normL1(), 0.0, 1e-8);
  if (verboseStartEnd)
    std::cerr << "================== TestFastHIK::testKernelMultiplicationFast done ===================== " << std::endl;
}


void TestFastHIK::testKernelSum() 
{
  if (verboseStartEnd)
    std::cerr << "================== TestFastHIK::testKernelSum ===================== " << std::endl;
  
  vector< vector<double> > dataMatrix;
  generateRandomFeatures ( d, n, dataMatrix );

  int nrZeros(0);
  for ( uint i = 0 ; i < d; i++ )
  {
    for ( uint k = 0; k < n; k++ )
      if ( drand48() < sparse_prob ) 
      {
        dataMatrix[i][k] = 0.0;
        nrZeros++;
      }
  }
  
  if ( verbose ) {
    cerr << "data matrix: " << endl;
    printMatrix ( dataMatrix );
    cerr << endl;
  }

  double noise = 1.0;
  FastMinKernel fmk ( dataMatrix, noise );
  
  Vector alpha = Vector::UniformRandom( n, 0.0, 1.0, 0 );

  NICE::VVector ASparse;
  NICE::VVector BSparse;
  fmk.hik_prepare_alpha_multiplications ( alpha, ASparse, BSparse ); 
  
  Vector xstar (d);
  for ( uint i = 0 ; i < d ; i++ )
    if ( drand48() < sparse_prob ) {
      xstar[i] = 0.0;
    } else {
      xstar[i] = rand();
    }
  SparseVector xstarSparse ( xstar );
    
  double betaSparse;
  fmk.hik_kernel_sum ( ASparse, BSparse, xstarSparse, betaSparse );
  
  if (verbose)
    std::cerr << "kernelSumSparse done, now do the thing without exploiting sparsity" << std::endl;

  
  // checking the result
  std::vector<std::vector<double> > dataMatrix_transposed (dataMatrix);
  transposeVectorOfVectors(dataMatrix_transposed);
  NICE::IntersectionKernelFunction<double> hikSlow;

  std::vector<double> xstar_stl;
  xstar_stl.resize(d);
  for ( uint i = 0 ; i < d; i++ )
    xstar_stl[i] = xstar[i];
  std::vector<double> kstar_stl = hikSlow.computeKernelVector ( dataMatrix_transposed, xstar_stl );
  double beta_slow = 0.0;
  for ( uint i = 0 ; i < n; i++ )
    beta_slow += kstar_stl[i] * alpha[i];

  if (verbose)
    std::cerr << "difference of beta_slow and betaSparse: " << fabs(beta_slow - betaSparse) << std::endl;
  CPPUNIT_ASSERT_DOUBLES_EQUAL(beta_slow, betaSparse, 1e-8);
  
  if (verboseStartEnd)
    std::cerr << "================== TestFastHIK::testKernelSum done ===================== " << std::endl;
}


void TestFastHIK::testKernelSumFast() 
{
  if (verboseStartEnd)
    std::cerr << "================== TestFastHIK::testKernelSumFast ===================== " << std::endl;
  
  Quantization q ( numBins );

  // data is generated, such that there is no approximation error
  vector< vector<double> > dataMatrix;
  for ( uint i = 0; i < d ; i++ )
  {
    vector<double> v;
    v.resize(n);
    for ( uint k = 0; k < n; k++ ) {
      if ( drand48() < sparse_prob ) {
        v[k] = 0;
      } else {
        v[k] = q.getPrototype( (rand() % numBins) );
      }
    }

    dataMatrix.push_back(v);
  }
  
  if ( verbose ) {
    cerr << "data matrix: " << endl;
    printMatrix ( dataMatrix );
    cerr << endl;
  }

  double noise = 1.0;
  FastMinKernel fmk ( dataMatrix, noise );
  Vector alpha = Vector::UniformRandom( n, 0.0, 1.0, 0 );
  if ( verbose )
    std::cerr << "alpha = " << alpha << endl;

  // generate xstar
  Vector xstar (d);
  for ( uint i = 0 ; i < d ; i++ )
    if ( drand48() < sparse_prob ) {
      xstar[i] = 0;
    } else {
      xstar[i] = q.getPrototype( (rand() % numBins) );
    }

  // convert to STL vector
  vector<double> xstar_stl;
  xstar_stl.resize(d);
  for ( uint i = 0 ; i < d; i++ )
    xstar_stl[i] = xstar[i];

  if ( verbose ) 
    cerr << "xstar = " << xstar << endl;
 
  for ( double gamma = 1.0 ; gamma < 2.0; gamma += 0.5 ) 
  {
    if (verbose)
      std::cerr << "testing hik_kernel_sum_fast with ghik parameter: " << gamma << endl;

    PFAbsExp pf ( gamma );

//     pf.applyFunctionToFeatureMatrix ( fmk.featureMatrix() );
    fmk.applyFunctionToFeatureMatrix( &pf );

    NICE::VVector A;
    NICE::VVector B;
    if (verbose)
      std::cerr << "fmk.hik_prepare_alpha_multiplications ( alpha, A, B ) " << std::endl;
    fmk.hik_prepare_alpha_multiplications ( alpha, A, B ); 

    if (verbose)
      //std::cerr << "double *Tlookup = fmk.hik_prepare_alpha_multiplications_fast( A, B, q )" << std::endl;
      std::cerr << "double *Tlookup = fmk.hik_prepare_alpha_multiplications_fast_alltogether( alpha, q, &pf )" << std::endl;
    double *TlookupOld = fmk.hik_prepare_alpha_multiplications_fast( A, B, q, &pf ); 
    double *TlookupNew = fmk.hikPrepareLookupTable( alpha, q, &pf ); 
    
    int maxAcces(numBins*d);
    
    if (verbose)
    {
      std::cerr << "TlookupOld:  " << std::endl;
      for (int i = 0; i < maxAcces; i++)
      {
        std::cerr << TlookupOld[i] << " ";
        if ( (i%numBins) == (numBins-1))
          std::cerr << std::endl;
      }
      std::cerr << "TlookupNew:  " << std::endl;
      for (int i = 0; i < maxAcces; i++)
      {
        std::cerr << TlookupNew[i] << " ";
        if ( (i%numBins) == (numBins-1))
          std::cerr << std::endl;
      }    
    }
    
    if (verbose)
      std::cerr << "fmk.hik_kernel_sum_fast ( Tlookup, q, xstar, beta_fast )" << std::endl;
    
    double beta_fast;
    fmk.hik_kernel_sum_fast ( TlookupNew, q, xstar, beta_fast );
    
    NICE::SparseVector xstar_sparse(xstar);
    
    double beta_fast_sparse;
    fmk.hik_kernel_sum_fast ( TlookupNew, q, xstar_sparse, beta_fast_sparse );
    
    double betaSparse;
    fmk.hik_kernel_sum ( A, B, xstar_sparse, betaSparse, &pf );

    // checking the result
    std::vector<std::vector<double> > dataMatrix_transposed (dataMatrix);
    transposeVectorOfVectors(dataMatrix_transposed);
    NICE::GeneralizedIntersectionKernelFunction<double> hikSlow (gamma);

    vector<double> kstar_stl = hikSlow.computeKernelVector ( dataMatrix_transposed, xstar_stl );
    double beta_slow = 0.0;
    for ( uint i = 0 ; i < n; i++ )
      beta_slow += kstar_stl[i] * alpha[i];

    if (verbose)
      std::cerr << "beta_slow: " << beta_slow << std::endl << "beta_fast: " << beta_fast << std::endl << "beta_fast_sparse: " << beta_fast_sparse << std::endl << "betaSparse: " << betaSparse<< std::endl;
    CPPUNIT_ASSERT_DOUBLES_EQUAL(beta_slow, beta_fast_sparse, 1e-8);
  
    delete [] TlookupNew;
    delete [] TlookupOld;
  }
  
  if (verboseStartEnd)
    std::cerr << "================== TestFastHIK::testKernelSumFast done ===================== " << std::endl;

}

void TestFastHIK::testLUTUpdate()
{
  if (verboseStartEnd)
    std::cerr << "================== TestFastHIK::testLUTUpdate ===================== " << std::endl;

  Quantization q ( numBins );

  // data is generated, such that there is no approximation error
  vector< vector<double> > dataMatrix;
  for ( uint i = 0; i < d ; i++ )
  {
    vector<double> v;
    v.resize(n);
    for ( uint k = 0; k < n; k++ ) {
      if ( drand48() < sparse_prob ) {
        v[k] = 0;
      } else {
        v[k] = q.getPrototype( (rand() % numBins) );
      }
    }

    dataMatrix.push_back(v);
  }
  
  if ( verbose ) {
    cerr << "data matrix: " << endl;
    printMatrix ( dataMatrix );
    cerr << endl;
  }

  double noise = 1.0;
  FastMinKernel fmk ( dataMatrix, noise );
  
  ParameterizedFunction *pf = new PFAbsExp ( 1.0 );

  Vector alpha ( n );
  for ( uint i = 0; i < alpha.size(); i++ )
    alpha[i] = sin(i);
  
  if (verbose)
    std::cerr << "prepare LUT" << std::endl;
  double * T = fmk.hikPrepareLookupTable(alpha, q, pf);
  if (verbose)
    std::cerr << "preparation done -- printing T" << std::endl;
  
  int maxAcces(numBins*d);
  if (verbose)
  {
    for (int i = 0; i < maxAcces; i++)
    {
      std::cerr << T[i] << " ";
      if ( (i%numBins) == (numBins-1))
        std::cerr << std::endl;
    }    
  }

  //lets change index 2
  int idx(2);
  double valAlphaOld(alpha[idx]);
  double valAlphaNew(1.2); //this value is definitely different from the previous one
      
  Vector alphaNew(alpha);
  alphaNew[idx] = valAlphaNew;
  
  double * TNew = fmk.hikPrepareLookupTable(alphaNew, q, pf);
  if (verbose)
    std::cerr << "calculated the new LUT, no print it: " << std::endl;
  
  if (verbose)
  {
    for (int i = 0; i < maxAcces; i++)
    {
      std::cerr << TNew[i] << " ";
      if ( (i%numBins) == (numBins-1))
        std::cerr << std::endl;
    } 
  }

  if (verbose)
    std::cerr << "change the old LUT by a new value for alpha_i" << std::endl;
  fmk.hikUpdateLookupTable(T, valAlphaNew, valAlphaOld, idx, q, pf );
  if (verbose)
    std::cerr << "update is done, now print the updated version: " << std::endl;
  
  if (verbose)
  {
    for (int i = 0; i < maxAcces; i++)
    {
      std::cerr << T[i] << " ";
      if ( (i%numBins) == (numBins-1))
        std::cerr << std::endl;
    } 
  }
  
  
  bool equal = compareLUTs(T, TNew, q.size()*d, 10e-8);
  
  if (verbose)
  {
    if (equal)
      std::cerr << "LUTs are equal :) " << std::endl;
    else
    {
      std::cerr << "T are not equal :( " << std::endl;
      for (uint i = 0; i < q.size()*d; i++)
      {
        if ( (i % q.size()) == 0)
          std::cerr << std::endl;
        std::cerr << T[i] << " ";
      }
      std::cerr << "TNew: "<< std::endl;
      for (uint i = 0; i < q.size()*d; i++)
      {
        if ( (i % q.size()) == 0)
          std::cerr << std::endl;
        std::cerr << TNew[i] << " ";
      }     
    
    }    
  }
  
  CPPUNIT_ASSERT(equal == true);
  
  if (verboseStartEnd)
    std::cerr << "================== TestFastHIK::testLUTUpdate done ===================== " << std::endl;
  
    delete [] T;
    delete [] TNew;
}

void TestFastHIK::testLinSolve()
{

  if (verboseStartEnd)
    std::cerr << "================== TestFastHIK::testLinSolve ===================== " << std::endl;

  Quantization q ( numBins );

  // data is generated, such that there is no approximation error
  vector< vector<double> > dataMatrix;
  for ( uint i = 0; i < d ; i++ )
  {
    vector<double> v;
    v.resize(n);
    for ( uint k = 0; k < n; k++ ) {
      if ( drand48() < sparse_prob ) {
        v[k] = 0;
      } else {
        v[k] = q.getPrototype( (rand() % numBins) );
      }
    }

    dataMatrix.push_back(v);
  }
  
  if ( verbose ) {
    cerr << "data matrix: " << endl;
    printMatrix ( dataMatrix );
    cerr << endl;
  }

  double noise = 1.0;
  FastMinKernel fmk ( dataMatrix, noise );
  
  ParameterizedFunction *pf = new PFAbsExp ( 1.0 );
  fmk.applyFunctionToFeatureMatrix( pf );
//   pf->applyFunctionToFeatureMatrix ( fmk.featureMatrix() );

  Vector y ( n );  
  for ( uint i = 0; i < y.size(); i++ )
    y[i] = sin(i);
  
  Vector alpha;
  Vector alphaRandomized;

  std::cerr << "solveLin with randomization" << std::endl;
  // tic
  Timer t;
  t.start();
  //let's try to do 10.000 iterations and sample in each iteration 30 examples randomly
  fmk.solveLin(y,alphaRandomized,q,pf,true,solveLinMaxIterations,30);
  //toc
  t.stop();
  float time_randomizedSolving = t.getLast();
  std::cerr << "Time for solving with random subsets: " << time_randomizedSolving << " s" << std::endl;  
  
  // test the case, where we first transform and then use the multiply stuff
  std::vector<std::vector<double> > dataMatrix_transposed (dataMatrix);
  transposeVectorOfVectors(dataMatrix_transposed);
  
  NICE::GeneralizedIntersectionKernelFunction<double> ghikSlow ( 1.0 );
  NICE::Matrix gK ( ghikSlow.computeKernelMatrix(dataMatrix_transposed, noise) );
  
  Vector K_alphaRandomized;
  K_alphaRandomized.multiply(gK, alphaRandomized);
  
  if (solveLinWithoutRand)
  {
    std::cerr << "solveLin without randomization" << std::endl;
    fmk.solveLin(y,alpha,q,pf,false,1000);
    Vector K_alpha;
    K_alpha.multiply(gK, alpha);
    std::cerr << "now assert that K_alpha == y" << std::endl;
    std::cerr << "(K_alpha-y).normL1(): " << (K_alpha-y).normL1() << std::endl;
  }
   
//   std::cerr << "alpha: " << alpha << std::endl;
//   std::cerr << "K_times_alpha: " << K_alpha << std::endl;
//   std::cerr << "y: " << y << std::endl;
//   
//   Vector test_alpha;
//   ILSConjugateGradients cgm;
//   cgm.solveLin( GMStandard(gK),y,test_alpha);
//   
//   K_alpha.multiply( gK, test_alpha);
//   
//   std::cerr << "test_alpha (CGM): " << test_alpha << std::endl;
//   std::cerr << "K_times_alpha (CGM): " << K_alpha << std::endl;
  
  std::cerr << "now assert that K_alphaRandomized == y" << std::endl;
  std::cerr << "(K_alphaRandomized-y).normL1(): " << (K_alphaRandomized-y).normL1() << std::endl;
  

//   CPPUNIT_ASSERT_DOUBLES_EQUAL((K_alphaRandomized-y).normL1(), 0.0, 1e-6);
  
  if (verboseStartEnd)
    std::cerr << "================== TestFastHIK::testLinSolve done ===================== " << std::endl;
}

void TestFastHIK::testKernelVector()
{
  if (verboseStartEnd)
    std::cerr << "================== TestFastHIK::testKernelVector ===================== " << std::endl;  
  
  std::vector< std::vector<double> > dataMatrix;
  
  std::vector<double> dim1; dim1.push_back(0.2);dim1.push_back(0.1);dim1.push_back(0.0);dim1.push_back(0.0);dim1.push_back(0.4); dataMatrix.push_back(dim1);
  std::vector<double> dim2; dim2.push_back(0.3);dim2.push_back(0.6);dim2.push_back(1.0);dim2.push_back(0.4);dim2.push_back(0.3); dataMatrix.push_back(dim2);
  std::vector<double> dim3; dim3.push_back(0.5);dim3.push_back(0.3);dim3.push_back(0.0);dim3.push_back(0.6);dim3.push_back(0.3); dataMatrix.push_back(dim3);
  
  if ( verbose ) {
    std::cerr << "data matrix: " << std::endl;
    printMatrix ( dataMatrix );
    std::cerr << endl;
  }

  double noise = 1.0;
  FastMinKernel fmk ( dataMatrix, noise );

  std::vector<double> xStar; xStar.push_back(0.2);xStar.push_back(0.7);xStar.push_back(0.1);
  NICE::Vector xStarVec (xStar);
  std::vector<double> x2; x2.push_back(0.7);x2.push_back(0.3);xStar.push_back(0.0);
  NICE::Vector x2Vec (x2);
  
  NICE::SparseVector xStarsparse( xStarVec );
  NICE::SparseVector x2sparse( x2Vec );
  
  NICE::Vector k1;
  fmk.hikComputeKernelVector( xStarsparse, k1 );
  
  NICE::Vector k2;
  fmk.hikComputeKernelVector( x2sparse, k2 );
   
  NICE::Vector k1GT(5); k1GT[0] = 0.6; k1GT[1] = 0.8; k1GT[2] = 0.7; k1GT[3] = 0.5; k1GT[4] = 0.6;
  NICE::Vector k2GT(5); k2GT[0] = 0.5; k2GT[1] = 0.4; k2GT[2] = 0.3; k2GT[3] = 0.3; k2GT[4] = 0.7;
  
  if (verbose)
  {
    std::cerr << "k1: " << k1 << std::endl;
    std::cerr << "GT: " << k1GT << std::endl;
    std::cerr << "k2: " << k2 << std::endl;
    std::cerr << "GT: " << k2GT << std::endl;
  }
    
  for (int i = 0; i < 5; i++)
  {
    CPPUNIT_ASSERT_DOUBLES_EQUAL(k1[i]-k1GT[i], 0.0, 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(k2[i]-k2GT[i], 0.0, 1e-6);
  }

  
  if (verboseStartEnd)
    std::cerr << "================== TestFastHIK::testKernelVector done ===================== " << std::endl;
  
}

void TestFastHIK::testAddExample()
{
  if (verboseStartEnd)
    std::cerr << "================== TestFastHIK::testAddExample ===================== " << std::endl;  
  
  std::vector< std::vector<double> > dataMatrix;
  int dim = 3;
  int number = 5;
  
  if (!smallTest)
  {
    dim = d;
    number = n;
  }
  
  if (smallTest)
  {
    dataMatrix.resize(3);
    //we explicitely give some values which can easily be verified
    dataMatrix[0].push_back(0.2);dataMatrix[0].push_back(0.1);dataMatrix[0].push_back(0.0);dataMatrix[0].push_back(0.0);dataMatrix[0].push_back(0.4); 
    dataMatrix[1].push_back(0.3);dataMatrix[1].push_back(0.6);dataMatrix[1].push_back(1.0);dataMatrix[1].push_back(0.4);dataMatrix[1].push_back(0.3);
    dataMatrix[2].push_back(0.5);dataMatrix[2].push_back(0.3);dataMatrix[2].push_back(0.0);dataMatrix[2].push_back(0.6);dataMatrix[2].push_back(0.3);
  }
  else
  {
    // randomly generate features
    generateRandomFeatures ( dim, number, dataMatrix );

    // and make them sparse
    int nrZeros(0);
    for ( int i = 0 ; i < dim; i++ )
    {
      for ( int k = 0; k < number; k++ )
        if ( drand48() < sparse_prob ) 
        {
          dataMatrix[i][k] = 0.0;
          nrZeros++;
        }
    }    
  }
  
  if ( verbose ) {
    std::cerr << "data matrix: " << std::endl;
    printMatrix ( dataMatrix );
    std::cerr << endl;
  }
  
  double noise = 1.0;
  //check the features stored in the fmk
  FastMinKernel fmk ( dataMatrix, noise );  
  NICE::Vector alpha;
  
  ParameterizedFunction *pf = new PFAbsExp( 1.2 ); //1.0 is okay
  fmk.applyFunctionToFeatureMatrix( pf );
//   pf->applyFunctionToFeatureMatrix ( fmk.featureMatrix() );  
  
  std::cerr << "generate alpha" << std::endl;
  
  if (smallTest)
  {
    //we explicitely give some values which can easily be verified
    alpha = Vector(5,1.0);
    alpha[0] = 0.1;alpha[1] = 0.2;alpha[2] = 0.4;alpha[3] = 0.8;alpha[4] = 1.6;
  }
  else
  {  // randomly generate features
     alpha = Vector::UniformRandom( number, 0.0, 1.0, 0 );
  }
  
  
  std::cerr << "generate xStar" << std::endl;
  std::vector<double> xStar;
  if (smallTest)
  {
    // we check the following cases: largest elem in dim, smallest elem in dim, zero element
    // remember to adapt the feature in some lines apart as well    
    xStar.push_back(0.9);xStar.push_back(0.0);xStar.push_back(0.1);
  }
  else
  {
    // again: random sampling
    for ( int i = 0 ; i < dim; i++ )
    {
      if ( drand48() < sparse_prob ) 
        xStar.push_back(0.0);
      else
        xStar.push_back(drand48());
    }
  }
  NICE::Vector xStarVec (xStar);
  NICE::SparseVector xStarSV (xStarVec);
  
  // check the alpha-preparations
  NICE::VVector A;
  NICE::VVector B;
  fmk.hik_prepare_alpha_multiplications( alpha, A, B );
  
  //check the quantization and LUT construction
  Quantization q ( numBins );  
  //direct
//   double * LUT = fmk.hikPrepareLookupTable(alpha, q);
  //indirect
  double * LUT = fmk.hik_prepare_alpha_multiplications_fast( A, B, q, pf );
  
  //check for kernel vector norm approximation
  NICE::VVector AForKVN;
  fmk.hikPrepareKVNApproximation(AForKVN);
  
  //check the LUTs for fast kernel vector norm approximation
  //direct
  double* LUT_kernelVectorNormDirect = fmk.hikPrepareLookupTableForKVNApproximation(q, pf );
  //indirect
  double* LUT_kernelVectorNorm = fmk.hikPrepareKVNApproximationFast( AForKVN, q, pf );
  
  bool LUTKVN_equal( compareLUTs( LUT_kernelVectorNorm, LUT_kernelVectorNormDirect, q.size()*dim ) );
  
  if (verbose)
  {
    if (LUTKVN_equal == false)
    {
      std::cerr << "LUTKVN is not equal :( " << std::endl;
        std::cerr << "LUT_kernelVectorNorm: " << std::endl;
        for ( uint i = 0; i < q.size()*dim; i++ )
        {
          if ( (i % q.size()) == 0)
            std::cerr << std::endl;
          std::cerr << LUT_kernelVectorNorm[i] << " ";
        }
        std::cerr << "LUT_kernelVectorNormDirect: "<< std::endl;
        for ( uint i = 0; i < q.size()*dim; i++ )
        {
          if ( (i % q.size()) == 0)
            std::cerr << std::endl;
          std::cerr << LUT_kernelVectorNormDirect[i] << " ";
        }      
    }
  }
  CPPUNIT_ASSERT( LUTKVN_equal == true );
  
  if (verbose)
    std::cerr << "start the incremental learning part" << std::endl;

  // ------  Incremental Learning -----
  
  double newAlpha;
  if (smallTest) 
    newAlpha = 3.2;
  else
    newAlpha = drand48();
  alpha.append(newAlpha);
   
  // add an example
  if (verbose)
    std::cerr << "addExample" << std::endl;  
  fmk.addExample( xStarSV, pf );  
  
  // update the alpha preparation
  if (verbose)  
    std::cerr << "update Alpha Preparation" << std::endl;
  fmk.updatePreparationForAlphaMultiplications( xStarSV, newAlpha, A, B, pf );
  
  // update the LUT for fast multiplications
  if (verbose)  
    std::cerr << "update LUT" << std::endl;
  fmk.updateLookupTableForAlphaMultiplications( xStarSV, newAlpha, LUT, q, pf );
  
  //update VVector for Kernel vector norm
  if (verbose)  
    std::cerr << "update VVector for Kernel vector norm" << std::endl;
  fmk.updatePreparationForKVNApproximation( xStarSV, AForKVN, pf );
  
  // update LUT for kernel vector norm
  if (verbose)  
    std::cerr << "update LUT for kernel vector norm" << std::endl;
  fmk.updateLookupTableForKVNApproximation( xStarSV, LUT_kernelVectorNorm, q, pf );
  
  //and batch retraining  
  if (verbose)  
    std::cerr << "perform batch retraining " << std::endl;  
  for ( int i = 0 ; i < dim; i++ )
    dataMatrix[i].push_back(xStar[i]);
  
  FastMinKernel fmk2 ( dataMatrix, noise );
  fmk2.applyFunctionToFeatureMatrix( pf );
  
  NICE::VVector A2;
  NICE::VVector B2;
  if (verbose)  
    std::cerr << "prepare alpha multiplications" << std::endl;
  fmk2.hik_prepare_alpha_multiplications( alpha, A2, B2 );
 
  // compare the content of the data matrix
  if (verbose)  
    std::cerr << "do the comparison of the resulting feature matrices" << std::endl;
  if (verbose)
  {
    std::cerr << "fmk.featureMatrix().print()" << std::endl;
    fmk.featureMatrix().print(std::cerr);
  
    std::cerr << "fmk2.featureMatrix().print()" << std::endl;
    fmk2.featureMatrix().print(std::cerr);
  }  
  
  CPPUNIT_ASSERT(fmk.featureMatrix() == fmk2.featureMatrix());

  //compare the preparation for alpha multiplications
  if (verbose)  
    std::cerr << "do the comparison of the resulting matrices A and B" << std::endl;
  CPPUNIT_ASSERT(compareVVector(A, A2));  
  CPPUNIT_ASSERT(compareVVector(B, B2));
  
  if (verbose)
  {
    std::cerr << "compare the preparation for alpha multiplications" << std::endl;
    std::cerr << "A: " << std::endl;
    A.store(std::cerr);
    std::cerr << "A2: " << std::endl;
    A2.store(std::cerr);
    std::cerr << "B: " << std::endl;
    B.store(std::cerr);
    std::cerr << "B2: " << std::endl;
    B2.store(std::cerr);
  }  
  
  // compare the resulting LUTs
  if (verbose)
    std::cerr << "prepare LUT" << std::endl;
  double * LUT2 = fmk2.hikPrepareLookupTable( alpha, q, pf );    
  if (verbose)
    std::cerr << "do the comparison of the resulting LUTs" << std::endl;  
  bool LUTequal( compareLUTs( LUT, LUT2, q.size()*dim) );
  
  if (verbose)
  {
    if ( LUTequal )
      std::cerr << "LUTs are equal :) " << std::endl;
    else
    {
      std::cerr << "LUTs are not equal :( " << std::endl;
      std::cerr << "new feature vector: " << xStarVec << std::endl;
      
      std::cerr << "newAlpha: " << newAlpha <<  " alpha " << alpha << std::endl;
      std::cerr << "LUT: " << std::endl;
      for ( uint i = 0; i < q.size()*dim; i++ )
      {
        if ( (i % q.size()) == 0)
          std::cerr << std::endl;
        std::cerr << LUT[i] << " ";
      }
      std::cerr << "LUT2: "<< std::endl;
      for ( uint i = 0; i < q.size()*dim; i++ )
      {
        if ( (i % q.size()) == 0)
          std::cerr << std::endl;
        std::cerr << LUT2[i] << " ";
      }     
    }
  }
  CPPUNIT_ASSERT( LUTequal );
  
  //check for kernel vector norm approximation
  NICE::VVector A2ForKVN;
  fmk2.hikPrepareKVNApproximation( A2ForKVN );
  bool KVN_equal ( compareVVector(AForKVN, A2ForKVN) );
 
  if (verbose)
  {
    if ( KVN_equal )
      std::cerr << "VVectors for kernel vector norm are equal :) " << std::endl;
    else
    {
      std::cerr << "VVectors for vector norm are not equal :( " << std::endl;
      std::cerr << "new feature vector: " << xStarVec << std::endl;
      
      std::cerr << "AForKVN: " << std::endl;
      AForKVN.store(std::cerr);
      
      std::cerr << "A2ForKVN: "<< std::endl;
      A2ForKVN.store(std::cerr);
    }
  }  
  
  CPPUNIT_ASSERT( KVN_equal );  
  
  //check for kernel vector norm approximation with LUTs
  if (verbose)
    std::cerr << "prepare LUT for kernel vector norm" << std::endl;
  double* LUT2_kernelVectorNorm = fmk2.hikPrepareLookupTableForKVNApproximation( q, pf );  
  if (verbose)
    std::cerr << "do the comparison of the resulting LUTs for kernel vector norm computation" << std::endl;
  bool LUT_KVN_equal( compareLUTs ( LUT_kernelVectorNorm, LUT2_kernelVectorNorm, q.size()*dim ) );
  
  if (verbose)
  {
    if ( LUT_KVN_equal )
      std::cerr << "LUTs for kernel vector norm are equal :) " << std::endl;
    else
    {
      std::cerr << "LUTs kernel vector norm are not equal :( " << std::endl;
      std::cerr << "new feature vector: " << xStarVec << std::endl;
      
      std::cerr << "LUT_kernelVectorNorm: " << std::endl;
      for ( int i = 0; i < q.size()*dim; i++ )
      {
        if ( (i % q.size()) == 0)
          std::cerr << std::endl;
        std::cerr << LUT_kernelVectorNorm[i] << " ";
      }
      std::cerr << std::endl << "LUT2_kernelVectorNorm: "<< std::endl;
      for ( uint i = 0; i < q.size()*dim; i++ )
      {
        if ( (i % q.size()) == 0)
          std::cerr << std::endl;
        std::cerr << LUT2_kernelVectorNorm[i] << " ";
      }     
    }
  }  
  
  CPPUNIT_ASSERT( LUT_KVN_equal );
  
  delete [] LUT;
  delete [] LUT2;
  
  delete [] LUT_kernelVectorNorm;
  delete [] LUT2_kernelVectorNorm;
  
  if (verboseStartEnd)
    std::cerr << "================== TestFastHIK::testAddExample done ===================== " << std::endl;  
}

void TestFastHIK::testAddMultipleExamples()
{
  if (verboseStartEnd)
    std::cerr << "================== TestFastHIK::testAddMultipleExamples ===================== " << std::endl;  
  
  std::vector< std::vector<double> > dataMatrix;
  int dim = d;
  int number = n;

  // randomly generate features
  generateRandomFeatures ( dim, number, dataMatrix );

  // and make them sparse
  int nrZeros(0);
  for ( int i = 0 ; i < dim; i++ )
  {
    for ( int k = 0; k < number; k++ )
      if ( drand48() < sparse_prob ) 
      {
        dataMatrix[i][k] = 0.0;
        nrZeros++;
      }
  }    
  
  if ( verbose ) {
    std::cerr << "data matrix: " << std::endl;
    printMatrix ( dataMatrix );
    std::cerr << endl;
  }
  
  double noise = 1.0;
  //check the features stored in the fmk
  FastMinKernel fmk ( dataMatrix, noise );  
  NICE::Vector alpha;
  
  ParameterizedFunction *pf = new PFAbsExp( 1.0 ); //1.0 is okay
  fmk.applyFunctionToFeatureMatrix( pf );
  
  std::cerr << "generate alpha" << std::endl;  
  
  // randomly generate features
  alpha = Vector::UniformRandom( number, 0.0, 1.0, 0 );
   
/*  // check the alpha-preparations
  NICE::VVector A;
  NICE::VVector B;
  fmk.hik_prepare_alpha_multiplications( alpha, A, B );*/  
  
  if (verbose)
    std::cerr << "start the incremental learning part" << std::endl;

  // ------  Incremental Learning -----
    
  std::cerr << "generate xStar" << std::endl;
  std::vector<NICE::SparseVector > newExamples;
  int nrOfNewExamples(5);
  // again: random sampling
  for (int i = 0; i < nrOfNewExamples; i++)
  {
    NICE::Vector xStar(dim);
    for ( int j = 0 ; j < dim; j++ )
    {
      if ( drand48() < sparse_prob ) 
      {
        xStar[j] = 0.0;
        dataMatrix[j].push_back(0.0);
      }
      else
      {
        double tmp(drand48());
        xStar[j] = tmp;
        dataMatrix[j].push_back(tmp);
      }
    }
    
    NICE::SparseVector xStarSV (xStar);
    newExamples.push_back(xStarSV);
  }    

  // add an example
  if (verbose)
    std::cerr << "addExample" << std::endl;  
  for (int i = 0; i < nrOfNewExamples; i++)
  {
    fmk.addExample( newExamples[i], pf );  
  }
  
  int oldSize(alpha.size());
  alpha.resize( oldSize + nrOfNewExamples);
  for (int i = 0; i < nrOfNewExamples; i++)
  {
    alpha[oldSize + i] = drand48();
  }
   
  
  // update the alpha preparation
  if (verbose)  
    std::cerr << "update Alpha Preparation" << std::endl;
  // check the alpha-preparations
  NICE::VVector A;
  NICE::VVector B;
  fmk.hik_prepare_alpha_multiplications( alpha, A, B );   
  
  FastMinKernel fmk2 ( dataMatrix, noise );
  fmk2.applyFunctionToFeatureMatrix( pf );  
  
  NICE::VVector A2;
  NICE::VVector B2;
  fmk2.hik_prepare_alpha_multiplications( alpha, A2, B2 );
  
  bool equalA = compareVVector( A, A2 );
  bool equalB = compareVVector( B, B2 );
  
  CPPUNIT_ASSERT(equalA == true);
  CPPUNIT_ASSERT(equalB == true);  
  
  if (verboseStartEnd)
    std::cerr << "================== TestFastHIK::testAddMultipleExamples done ===================== " << std::endl;  
}

#endif