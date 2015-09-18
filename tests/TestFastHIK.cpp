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
#include <gp-hik-core/GMHIKernelRaw.h>
//
//
#include "gp-hik-core/quantization/Quantization.h"
#include "gp-hik-core/quantization/Quantization1DAequiDist0To1.h"

#include "TestFastHIK.h"

const bool b_debug = false;
const bool verbose = true;
const bool verboseStartEnd = true;
const bool solveLinWithoutRand = false;
const uint n = 1500;//1500;//1500;//10;
const uint d = 100;//200;//2;
const uint numBins = 11;//1001;//1001;
const uint solveLinMaxIterations = 1000;
const double sparse_prob = 0.6;
const bool smallTest = false;

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



using namespace NICE;
using namespace std;

CPPUNIT_TEST_SUITE_REGISTRATION( TestFastHIK );

void TestFastHIK::setUp() {
}

void TestFastHIK::tearDown() {
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

  if ( b_debug ) {
    cerr << "data matrix: " << endl;
    printMatrix ( dataMatrix );
    cerr << endl;
  }

  double noise = 1.0;
  NICE::Timer t;
  t.start();
  FastMinKernel fmk ( dataMatrix, noise );
  t.stop();
  if (verbose)
    std::cerr << "Time for FastMinKernel setup: " << t.getLast() << endl;

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


  // Test the GMHIKernel interface
  Vector alpha;

  t.start();
  gmk.multiply ( alpha, y );
  t.stop();
  if (verbose)
      std::cerr << "Time for kernel multiplication with GMHIKernel: " << t.getLast() << std::endl;


  // convert data structures to test the GMHIKernelRaw interface
  std::vector<std::vector<double> > dataMatrix_transposed (dataMatrix);
  transposeVectorOfVectors(dataMatrix_transposed);
  std::vector< const NICE::SparseVector * > dataMatrix_sparse;
  for ( std::vector< std::vector<double> >::const_iterator i = dataMatrix_transposed.begin(); i != dataMatrix_transposed.end(); i++ )
  {
    Vector w ( *i );
    SparseVector *v = new SparseVector ( w );
    dataMatrix_sparse.push_back(v);
  }

  t.start();
  GMHIKernelRaw gmk_raw ( dataMatrix_sparse, noise );
  t.stop();
  if (verbose)
    std::cerr << "Time for GMHIKernelRaw setup: " << t.getLast() << std::endl;

  Vector alpha_raw;
  t.start();
  gmk_raw.multiply ( alpha_raw, y );
  t.stop();
  if (verbose)
      std::cerr << "Time for kernel multiplication with GMHIKernelRaw: " << t.getLast() << std::endl;




  // compute the kernel matrix multiplication exactly
  NICE::IntersectionKernelFunction<double> hikSlow;

  // tic
  time_t  slow_start = clock();
  NICE::Matrix K (hikSlow.computeKernelMatrix(dataMatrix_transposed, noise));
  //toc
  float time_slowComputation = (float) (clock() - slow_start);
  if (verbose)
    std::cerr << "Time for computing the kernel matrix without using sparsity: " << time_slowComputation/CLOCKS_PER_SEC << " s" << std::endl;

  // tic
  time_t  slow_sparse_start = clock();

  NICE::Matrix KSparseCalculated (hikSlow.computeKernelMatrix(fmk.featureMatrix(), noise));
  //toc
  float time_slowComputation_usingSparsity = (float) (clock() - slow_sparse_start);
  if (verbose)
    std::cerr << "Time for computing the kernel matrix using sparsity: " << time_slowComputation_usingSparsity/CLOCKS_PER_SEC << " s" << std::endl;

  // check the trace calculation
  //CPPUNIT_ASSERT_DOUBLES_EQUAL( K.trace(), fmk.featureMatrix().hikTrace() + noise*n, 1e-12 );
  CPPUNIT_ASSERT_DOUBLES_EQUAL( K.trace(), fmk.featureMatrix().hikTrace() + noise*n, 1e-8 );

  // let us compute the kernel multiplication with the slow version
  Vector alpha_slow = K*y;

  if (b_debug)
    std::cerr << "Sparse multiplication [alpha, alpha_slow, alpha_raw]: " << std::endl <<  alpha << std::endl << alpha_slow << std::endl << alpha_raw << std::endl << std::endl;

  CPPUNIT_ASSERT_DOUBLES_EQUAL((alpha-alpha_slow).normL1(), 0.0, 1e-8);
  CPPUNIT_ASSERT_DOUBLES_EQUAL((alpha_raw-alpha_slow).normL1(), 0.0, 1e-8);

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

  delete pf;
}

void TestFastHIK::testKernelMultiplicationFast()
{
  if (verboseStartEnd)
    std::cerr << "================== TestFastHIK::testKernelMultiplicationFast ===================== " << std::endl;

  NICE::Quantization * q_gen;
  q_gen = new Quantization1DAequiDist0To1 ( numBins );

  NICE::Quantization * q;
  q = new Quantization1DAequiDist0To1 ( 2*numBins -1 );

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
        v[k] = q_gen->getPrototype( (rand() % numBins) );
      }
    }

    dataMatrix.push_back(v);
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
  GMHIKernel gmkFast ( &fmk, pf, q );


//   pf.applyFunctionToFeatureMatrix ( fmk.featureMatrix() );

  Vector alpha;

  gmk.multiply ( alpha, y );

  Vector alphaFast;

  gmkFast.multiply ( alphaFast, y );

  NICE::IntersectionKernelFunction<double> hikSlow;

  std::vector<std::vector<double> > dataMatrix_transposed (dataMatrix);
  transposeVectorOfVectors(dataMatrix_transposed);

  NICE::Matrix K (hikSlow.computeKernelMatrix(dataMatrix_transposed, noise));

  // check the trace calculation
  //CPPUNIT_ASSERT_DOUBLES_EQUAL( K.trace(), fmk.featureMatrix().hikTrace() + noise*n, 1e-12 );
  CPPUNIT_ASSERT_DOUBLES_EQUAL( K.trace(), fmk.featureMatrix().hikTrace() + noise*n, 1e-8 );

  // let us compute the kernel multiplication with the slow version
  Vector alpha_slow = K*y;

  if ( b_debug )
    std::cerr << "Sparse multiplication [alpha, alphaFast, alpha_slow]: " << std::endl <<  alpha << std::endl << alphaFast << std::endl << alpha_slow << std::endl << std::endl;

  CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, (alphaFast-alpha_slow).normL1(), 1e-8);

  // test the case, where we first transform and then use the multiply stuff
  NICE::GeneralizedIntersectionKernelFunction<double> ghikSlow ( 1.2 );

  NICE::Matrix gK ( ghikSlow.computeKernelMatrix(dataMatrix_transposed, noise) );
  pf->parameters()[0] = 1.2;
  fmk.applyFunctionToFeatureMatrix( pf );

  Vector galphaFast;
  gmkFast.multiply ( galphaFast, y );

  Vector galpha;

  gmk.multiply ( galpha, y );

  Vector galpha_slow = gK * y;

  if ( b_debug )
    std::cerr << "Sparse multiplication [galpha, galphaFast, galpha_slow]: " << std::endl <<  galpha << std::endl << galphaFast << std::endl << galpha_slow << std::endl << std::endl;

  // clean-up
  delete q_gen;
  delete q;

  // final assertion
  CPPUNIT_ASSERT_DOUBLES_EQUAL((galphaFast-galpha_slow).normL1(), 0.0, 1e-8);
  if (verboseStartEnd)
    std::cerr << "================== TestFastHIK::testKernelMultiplicationFast done ===================== " << std::endl;

  delete pf;
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

  if ( b_debug ) {
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

  NICE::Quantization * q;
  q = new Quantization1DAequiDist0To1 ( numBins );

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
        v[k] = q->getPrototype( (rand() % numBins) );
      }
    }

    dataMatrix.push_back(v);
  }

  if ( b_debug ) {
    cerr << "data matrix: " << endl;
    printMatrix ( dataMatrix );
    cerr << endl;
  }

  double noise = 1.0;
  FastMinKernel fmk ( dataMatrix, noise );
  Vector alpha = Vector::UniformRandom( n, 0.0, 1.0, 0 );
  if ( b_debug )
    std::cerr << "alpha = " << alpha << endl;

  // generate xstar
  Vector xstar (d);
  for ( uint i = 0 ; i < d ; i++ )
    if ( drand48() < sparse_prob ) {
      xstar[i] = 0;
    } else {
      xstar[i] = q->getPrototype( (rand() % numBins) );
    }

  // convert to STL vector
  vector<double> xstar_stl;
  xstar_stl.resize(d);
  for ( uint i = 0 ; i < d; i++ )
    xstar_stl[i] = xstar[i];

  if ( b_debug )
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

    if (b_debug)
      //std::cerr << "double *Tlookup = fmk.hik_prepare_alpha_multiplications_fast( A, B, q )" << std::endl;
      std::cerr << "double *Tlookup = fmk.hik_prepare_alpha_multiplications_fast_alltogether( alpha, q, &pf )" << std::endl;

    double *TlookupOld = fmk.hik_prepare_alpha_multiplications_fast( A, B, q, &pf );
    double *TlookupNew = fmk.hikPrepareLookupTable( alpha, q, &pf );

    int maxAcces(numBins*d);

    if (b_debug)
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

    if (b_debug)
      std::cerr << "beta_slow: " << beta_slow << std::endl << "beta_fast: " << beta_fast << std::endl << "beta_fast_sparse: " << beta_fast_sparse << std::endl << "betaSparse: " << betaSparse<< std::endl;

    // clean-up
    delete [] TlookupNew;
    delete [] TlookupOld;

    // final assertion
    CPPUNIT_ASSERT_DOUBLES_EQUAL(beta_slow, beta_fast_sparse, 1e-8);


  } // for-loop

  // clean-up
  delete q;

  if (verboseStartEnd)
    std::cerr << "================== TestFastHIK::testKernelSumFast done ===================== " << std::endl;

}

void TestFastHIK::testLUTUpdate()
{
  if (verboseStartEnd)
    std::cerr << "================== TestFastHIK::testLUTUpdate ===================== " << std::endl;

  NICE::Quantization * q;
  q = new Quantization1DAequiDist0To1 ( numBins );

  // data is generated, such that there is no approximation error
  std::vector< std::vector<double> > dataMatrix;
  for ( uint i = 0; i < d ; i++ )
  {
    std::vector<double> v;
    v.resize(n);
    for ( uint k = 0; k < n; k++ ) {
      if ( drand48() < sparse_prob ) {
        v[k] = 0;
      } else {
        v[k] = q->getPrototype( (rand() % numBins) );
      }
    }

    dataMatrix.push_back(v);
  }

  if ( b_debug ) {
    cerr << "data matrix: " << endl;
    printMatrix ( dataMatrix );
    cerr << endl;
  }

  double noise = 1.0;
  NICE::FastMinKernel fmk ( dataMatrix, noise );

  NICE::ParameterizedFunction *pf = new PFAbsExp ( 1.0 );

  NICE::Vector alpha ( n );
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


  bool equal = compareLUTs(T, TNew, q->getNumberOfBins()*d, 10e-8);

  if (verbose)
  {
    if (equal)
      std::cerr << "LUTs are equal :) " << std::endl;
    else
    {
      std::cerr << "T are not equal :( " << std::endl;
      for (uint i = 0; i < q->getNumberOfBins()*d; i++)
      {
        if ( (i % q->getNumberOfBins()) == 0)
          std::cerr << std::endl;
        std::cerr << T[i] << " ";
      }
      std::cerr << "TNew: "<< std::endl;
      for (uint i = 0; i < q->getNumberOfBins()*d; i++)
      {
        if ( (i % q->getNumberOfBins()) == 0)
          std::cerr << std::endl;
        std::cerr << TNew[i] << " ";
      }

    }
  }



  // clean-up
  delete q;
  delete pf;
  delete [] T;
  delete [] TNew;

  // final assertion
  CPPUNIT_ASSERT(equal == true);

  if (verboseStartEnd)
    std::cerr << "================== TestFastHIK::testLUTUpdate done ===================== " << std::endl;

}

void TestFastHIK::testLinSolve()
{

  if (verboseStartEnd)
    std::cerr << "================== TestFastHIK::testLinSolve ===================== " << std::endl;

  NICE::Quantization * q;
  q = new Quantization1DAequiDist0To1 ( numBins );

  // data is generated, such that there is no approximation error
  std::vector< std::vector<double> > dataMatrix;
  for ( uint i = 0; i < d ; i++ )
  {
    std::vector<double> v;
    v.resize(n);
    for ( uint k = 0; k < n; k++ ) {
      if ( drand48() < sparse_prob ) {
        v[k] = 0;
      } else {
        v[k] = q->getPrototype( (rand() % numBins) );
      }
    }

    dataMatrix.push_back(v);
  }

  if ( b_debug ) {
    std::cerr << "data matrix: " << std::endl;
    printMatrix ( dataMatrix );
    std::cerr << std::endl;
  }

  double noise = 1.0;
  NICE::FastMinKernel fmk ( dataMatrix, noise );

  NICE::ParameterizedFunction *pf = new NICE::PFAbsExp ( 1.0 );
  fmk.applyFunctionToFeatureMatrix( pf );

  NICE::Vector y ( n );
  for ( uint i = 0; i < y.size(); i++ )
    y[i] = sin(i);

  NICE::Vector alpha;
  NICE::Vector alphaRandomized;

  if ( verbose )
    std::cerr << "solveLin with randomization" << std::endl;
  // tic
  NICE::Timer t;
  t.start();
  //let's try to do 10.000 iterations and sample in each iteration 30 examples randomly
  fmk.solveLin(y,alphaRandomized,q,pf,true,solveLinMaxIterations,30);
  //toc
  t.stop();
  float time_randomizedSolving = t.getLast();
  if ( verbose )
    std::cerr << "Time for solving with random subsets: " << time_randomizedSolving << " s" << std::endl;

  // test the case, where we first transform and then use the multiply stuff
  std::vector<std::vector<double> > dataMatrix_transposed (dataMatrix);
  transposeVectorOfVectors(dataMatrix_transposed);

  NICE::GeneralizedIntersectionKernelFunction<double> ghikSlow ( 1.0 );
  NICE::Matrix gK ( ghikSlow.computeKernelMatrix(dataMatrix_transposed, noise) );

  NICE::Vector K_alphaRandomized;
  K_alphaRandomized.multiply(gK, alphaRandomized);

  if (solveLinWithoutRand)
  {
    if ( verbose )
      std::cerr << "solveLin without randomization" << std::endl;
    fmk.solveLin(y,alpha,q,pf,false,1000);
    Vector K_alpha;
    K_alpha.multiply(gK, alpha);

    if ( verbose )
    {
      std::cerr << "now assert that K_alpha == y" << std::endl;
      std::cerr << "(K_alpha-y).normL1(): " << (K_alpha-y).normL1() << std::endl;
    }
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

  if ( verbose )
  {
    std::cerr << "now assert that K_alphaRandomized == y" << std::endl;
    std::cerr << "(K_alphaRandomized-y).normL1(): " << (K_alphaRandomized-y).normL1() << std::endl;
  }

  // clean-up
  delete q;
  delete pf;

  // final assertion
  CPPUNIT_ASSERT_DOUBLES_EQUAL((K_alphaRandomized-y).normL1(), 0.0, 1e-6);

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

  if ( b_debug ) {
    std::cerr << "data matrix: " << std::endl;
    printMatrix ( dataMatrix );
    std::cerr << endl;
  }

  double noise = 1.0;
  FastMinKernel fmk ( dataMatrix, noise, b_debug );


  std::vector<double> xStar; xStar.push_back(0.2);xStar.push_back(0.7);xStar.push_back(0.1);
  NICE::Vector xStarVec (xStar);
  std::vector<double> x2; x2.push_back(0.7);x2.push_back(0.3);xStar.push_back(0.0);
  NICE::Vector x2Vec (x2);

  NICE::SparseVector xStarsparse( xStarVec );
  NICE::SparseVector x2sparse( x2Vec );


  if ( b_debug )
  {
    fmk.store ( std::cerr );
    xStarsparse.store ( std::cerr );
  }

  NICE::Vector k1;
  fmk.hikComputeKernelVector( xStarsparse, k1 );


  NICE::Vector k2;
  fmk.hikComputeKernelVector( x2sparse, k2 );

  NICE::Vector k1GT(5); k1GT[0] = 0.6; k1GT[1] = 0.8; k1GT[2] = 0.7; k1GT[3] = 0.5; k1GT[4] = 0.6;
  NICE::Vector k2GT(5); k2GT[0] = 0.5; k2GT[1] = 0.4; k2GT[2] = 0.3; k2GT[3] = 0.3; k2GT[4] = 0.7;

  if (b_debug)
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

#endif
