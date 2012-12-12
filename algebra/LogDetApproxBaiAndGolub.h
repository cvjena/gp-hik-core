/** 
* @file LogDetApproxBaiAndGolub.h
* @brief LogDet Approximation as stated by Bai and Golub ("Bounds for the Trace of the Inverse and the Determinant of Symmetric Positive Definite Matrices" in Annals of Numerical Mathematics") (Interface)
* @author Alexander Freytag
* @date 05-01-2012 (dd-mm-yyyy)
*/
#ifndef LOGDETAPPROXBAIANDGOLUBINCLUDE
#define LOGDETAPPROXBAIANDGOLUBINCLUDE

#include "gp-hik-core/algebra/LogDetApprox.h"

namespace NICE {

 /** 
 * @class LogDetApproxBaiAndGolub
 * @brief LogDet Approximation as stated by Bai and Golub ("Bounds for the Trace of the Inverse and the Determinant of Symmetric Positive Definite Matrices" in Annals of Numerical Mathematics")
 * @author Alexander Freytag
 */
 
  class LogDetApproxBaiAndGolub : public LogDetApprox
  {

    protected:
      bool verbose;
      
    public:
      
      //------------------------------------------------------
      // several constructors and destructors
      //------------------------------------------------------
      
      /** 
      * @brief Default constructor
      * @author Alexander Freytag
      * @date 05-01-2012 (dd-mm-yyyy)
      */
      LogDetApproxBaiAndGolub();
      
      /** 
      * @brief Default destructor
      * @author Alexander Freytag
      * @date 05-01-2012 (dd-mm-yyyy)
      */
      ~LogDetApproxBaiAndGolub();
      
      //------------------------------------------------------
      // get and set methods
      //------------------------------------------------------      
      void setVerbose(const bool & _verbose);
      
      //------------------------------------------------------
      // high level methods
      //------------------------------------------------------
      
      /** 
      * @brief  Compute an approximation for the logDet using Bai and Golubs paper
      * @author Alexander Freytag
      * @date 05-01-2012 (dd-mm-yyyy)
      * @pre A has to be symmetric and positive definite
      * @param A symmetric positive definite matrix
      * @return approximated logDet of A, computed by taking (LogDetUpperBound+LogDetLowerBound)/2
      */
      virtual double getLogDetApproximation(const NICE::Matrix & A);
      
      /** 
      * @brief  Compute an approximation for the logDet using Bai and Golubs paper
      * @author Alexander Freytag
      * @date 05-01-2012 (dd-mm-yyyy)
      * @pre A has to be symmetric and positive definite
      * @param A symmetric positive definite matrix
      * @param lambdaUpperBound guaranteed upper bound on the eigenvalues of A
      * @param lambdaLowerBound guaranteed lower bound on the eignvalues of A
      * @return approximated logDet of A, computed by taking (LogDetUpperBound+LogDetLowerBound)/2
      */
      double getLogDetApproximation(const NICE::Matrix A, const double & lambdaUpperBound, const double & lambdaLowerBound);
      
      /** 
      * @brief  Compute an approximation for the logDet using Bai and Golubs paper
      * @author Alexander Freytag
      * @date 05-01-2012 (dd-mm-yyyy)
      * @param mu1 ideally the trace of matrix A
      * @param mu2 ideally the frobenius norm of matrix A
      * @param lambdaUpperBound guaranteed upper bound on the eigenvalues of A
      * @param lambdaLowerBound guaranteed lower bound on the eignvalues of A
      * @param n number of rows in A (equals number of training examples used to compute the matrix, if A is a kernel matrix)
      * @return approximated logDet of A, computed by taking (LogDetUpperBound+LogDetLowerBound)/2
      */
      double getLogDetApproximation(const double & mu1, const double & mu2, const double & lambdaUpperBound, const double & lambdaLowerBound, const int & n );
      
      /** 
      * @brief  Compute an upper bound on the logDet using Bai and Golubs paper
      * @author Alexander Freytag
      * @date 05-01-2012 (dd-mm-yyyy)
      * @param mu1 ideally the trace of matrix A
      * @param mu2 ideally the frobenius norm of matrix A
      * @param lambdaUpperBound the guaranteed upper bound on the eigenvalues of A
      * @param n number of rows in A (equals number of training examples used to compute the matrix, if A is a kernel matrix)
      * @return guaranteed upper bound on the log det of A, if the inputs are correctly computed
      */
      double getLogDetApproximationUpperBound(const double & mu1, const double & mu2, const double & lambdaUpperBound, const int & n );
      
      /** 
      * @brief  Compute a lower bound on the logDet using Bai and Golubs paper
      * @author Alexander Freytag
      * @date 05-01-2012 (dd-mm-yyyy)
      * @param mu1 ideally the trace of matrix A
      * @param mu2 ideally the frobenius norm of matrix A
      * @param lambdaLowerBound the guaranteed lower bound on the eigenvalues of A
      * @param n number of rows in A (equals number of training examples used to compute the matrix, if A is a kernel matrix)
      * @return guaranteed lower bound on the log det of A, if the inputs are correctly computed
      */
      double getLogDetApproximationLowerBound(const double & mu1, const double & mu2, const double & lambdaLowerBound, const int & n );
  };
} //namespace

#endif