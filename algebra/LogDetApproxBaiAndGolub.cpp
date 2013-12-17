/** 
 * @file LogDetApproxBaiAndGolub.cpp
* @brief LogDet Approximation as stated by Bai and Golub ("Bounds for the Trace of the Inverse and the Determinant of Symmetric Positive Definite Matrices" in Annals of Numerical Mathematics) (Implementation)
* @author Alexander Freytag
* @date 05-01-2012 (dd-mm-yyyy)
*/

#include <limits>
#include <cmath> 

#include "gp-hik-core/algebra/LogDetApproxBaiAndGolub.h"
#include "core/vector/VectorT.h"


using namespace NICE;
using namespace std;

LogDetApproxBaiAndGolub::LogDetApproxBaiAndGolub()
{
  verbose = false;
}

LogDetApproxBaiAndGolub::~LogDetApproxBaiAndGolub()
{
}

void LogDetApproxBaiAndGolub::setVerbose(const bool & _verbose)
{
  verbose = _verbose;
}

double LogDetApproxBaiAndGolub::getLogDetApproximation(const NICE::Matrix & A)
{
  //todo compute lowest and largest eigenvalue if suitable methods here!

  double lambdaLowerBound(0.0);
  NICE::Vector ones(A.rows(), 1.0);
  NICE::Vector rightMultiplication;
  rightMultiplication.multiply(A, ones);

  //there is no nice way for multiplying two NICE::Vectors and returning a scalar :(
  rightMultiplication *= ones;
  //TODO For some reason I get an compilation error here: /home/luetz/code/fast-hik/nice/./core/vector/VectorT.tcc:539: undefined reference to `ippGetStatusString(IppStatus)'
  //Therefor the nasty workaround :(
  // 	double lambdaUpperBound(rightMultiplication.Sum());
  double lambdaUpperBound(0);
  for ( uint i = 0; i < rightMultiplication.size(); i++)
  {
    lambdaUpperBound += rightMultiplication[i];
  }

  //we could also call return getLogDetApproximation(A,lambdaUpperBound,lambdaLowerBound); - but this would need a second function call and we only have to write 3 extra lines of code
  double logDetLowerBound(getLogDetApproximationLowerBound(A.trace(), A.squaredFrobeniusNorm(), lambdaLowerBound, A.rows()) );
  double logDetUpperBound(getLogDetApproximationUpperBound(A.trace(), A.squaredFrobeniusNorm(), lambdaUpperBound, A.rows()) );

  return (fabs(logDetLowerBound) + fabs(logDetUpperBound) ) / 2.0;
}


double LogDetApproxBaiAndGolub::getLogDetApproximation(const NICE::Matrix A, const double & lambdaUpperBound, const double & lambdaLowerBound)
{
  double logDetLowerBound(getLogDetApproximationLowerBound(A.trace(), A.squaredFrobeniusNorm(), lambdaLowerBound, A.rows()) );
  double logDetUpperBound(getLogDetApproximationUpperBound(A.trace(), A.squaredFrobeniusNorm(), lambdaUpperBound, A.rows()) );

  return (fabs(logDetLowerBound) + fabs(logDetUpperBound) ) / 2.0;
}

double LogDetApproxBaiAndGolub::getLogDetApproximation(const double & mu1, const double & mu2, const double & lambdaUpperBound, const double & lambdaLowerBound, const int & n )
{
  double logDetLowerBound(getLogDetApproximationLowerBound(mu1, mu2, lambdaLowerBound, n) );
  double logDetUpperBound(getLogDetApproximationUpperBound(mu1, mu2, lambdaUpperBound, n) );

  return (logDetLowerBound + logDetUpperBound ) / 2.0;
}


double LogDetApproxBaiAndGolub::getLogDetApproximationUpperBound(const double & mu1, const double & mu2, const double & lambdaUpperBound, const int & n )
{
  double tUpper(numeric_limits<double>::max());
  if (  (lambdaUpperBound*n-mu1) != 0)
    tUpper = (lambdaUpperBound*mu1 - mu2) / (lambdaUpperBound*n-mu1);

//  if ( tUpper < 1e-10 ) {
//    fthrow(Exception, "LogDetApproxBaiAndGolub::getLogDetApproximationLowerBound: tUpper < 0.0 !! " << mu1 << " " << mu2 << " " << n << " " << lambdaUpperBound );
//  }

  // boundUpper = [log(beta) , log(tUpper)] * ([beta , tUpper; power(beta,2), power(tUpper,2)]^-1 * [mu1;mu2]);
  //inversion of a 2x2-matrix can be done explicitely: A^{-1} = \frac{1}{ad-bc} \bmatrix{ d & -b \\ -c & a}
  NICE::Matrix InnerMatrix(2,2);
  InnerMatrix(0,0) = pow(tUpper,2);
  InnerMatrix(0,1) = -tUpper;
  InnerMatrix(1,0) = -pow(lambdaUpperBound,2);
  InnerMatrix(1,1) = lambdaUpperBound;
  InnerMatrix *= 1.0/(lambdaUpperBound*pow(tUpper,2) - tUpper*pow(lambdaUpperBound,2));

  NICE::Vector leftSide(2);
  leftSide(0) = log(lambdaUpperBound);
  leftSide(1) = log(tUpper);

  if (verbose)
  {
    cerr << "Left side: " << leftSide << endl;
    cerr << InnerMatrix << endl;
  }

  NICE::Vector rightSide(2);
  rightSide(0) = mu1;
  rightSide(1) = mu2;

  NICE::Vector rightMultiplication;
  rightMultiplication.multiply(InnerMatrix,rightSide);

  //there is no nice way for multiplying two NICE::Vectors and returning a scalar :(
  leftSide *= rightMultiplication;

  // 	return leftSide.Sum();
  //TODO For some reason I get an compilation error here: /home/luetz/code/fast-hik/nice/./core/vector/VectorT.tcc:539: undefined reference to `ippGetStatusString(IppStatus)'
  //Therefor the nasty workaround :(
  double result(0.0);

  for ( uint i = 0; i < leftSide.size(); i++)
  {
    result += leftSide[i];
  }
  return result;
}


double LogDetApproxBaiAndGolub::getLogDetApproximationLowerBound(const double & mu1, const double & mu2, const double & lambdaLowerBound, const int & n )
{
  double tLower(numeric_limits<double>::max());
  if (  (lambdaLowerBound*n-mu1) != 0)
    tLower = (lambdaLowerBound*mu1 - mu2) / (lambdaLowerBound*n-mu1);

  // boundLower = [log(alpha) , log(tLower)] * ([alpha , tLower; power(alpha,2), power(tLower,2)]\[mu1;mu2]);
  //inversion of a 2x2-matrix can be done explicitely: A^{-1} = \frac{1}{ad-bc} \bmatrix{ d & -b \\ -c & a}
  NICE::Matrix InnerMatrix(2,2);
  InnerMatrix(0,0) = pow(tLower,2);
  InnerMatrix(0,1) = -tLower;
  InnerMatrix(1,0) = -pow(lambdaLowerBound,2);
  InnerMatrix(1,1) = lambdaLowerBound;
  InnerMatrix *= 1.0/(lambdaLowerBound*pow(tLower,2) - tLower*pow(lambdaLowerBound,2));

  NICE::Vector leftSide(2);
  leftSide(0) = log(lambdaLowerBound);
  leftSide(1) = log(tLower);

  NICE::Vector rightSide(2);
  rightSide(0) = mu1;
  rightSide(1) = mu2;

  NICE::Vector rightMultiplication;
  rightMultiplication.multiply(InnerMatrix,rightSide);

  return leftSide.scalarProduct( rightMultiplication );
  
//   //there is no nice way for multiplying two NICE::Vectors and returning a scalar so far
//   leftSide *= rightMultiplication;
// 
//   //nasty workaround for leftSide.Sum(), which does not compile properly on all machines
//   double result(0.0);
// 
//   for ( uint i = 0; i < leftSide.size(); i++)
//   {
//     result += leftSide[i];
//   }
//   return result;
}
