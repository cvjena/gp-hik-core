/** 
* @file GeneralizedIntersectionKernelFunction.cpp
* @brief The generalized intersection kernel function as distance measure between two histograms interpreted as vectors (Implementation)
* @author Alexander Freytag
* @date 08-12-2011 (dd-mm-yyyy)
*/

#include <gp-hik-core/SortedVectorSparse.h>

#include "GeneralizedIntersectionKernelFunction.h"
#include <math.h>

using namespace NICE;

template <typename T>
GeneralizedIntersectionKernelFunction<T>::GeneralizedIntersectionKernelFunction()
{
  exponent = 1.0;
}

template <typename T>
GeneralizedIntersectionKernelFunction<T>::GeneralizedIntersectionKernelFunction(const double & _exponent)
{
  exponent = _exponent;
}

template <typename T>
GeneralizedIntersectionKernelFunction<T>::~GeneralizedIntersectionKernelFunction()
{
}

template <typename T>
void GeneralizedIntersectionKernelFunction<T>::set_exponent(const double & _exponent)
{
  exponent = _exponent;
}

template <typename T>
double GeneralizedIntersectionKernelFunction<T>::get_exponent()
{
  return exponent;
}

template <typename T>
double GeneralizedIntersectionKernelFunction<T>::measureDistance ( const std::vector<T> & a, const std::vector<T> & b  )
{
  int size( (int) a.size());
  if ((int) b.size() < size)
    size = (int) b.size();

  double distance(0.0);

  for (int i = 0; i < size; i++)
  {
    if ( a[i] < b[i])
      distance += pow((double) a[i],exponent);
    else
      distance += pow((double) b[i],exponent);
  }
  return distance;
}

template <typename T>
NICE::Matrix GeneralizedIntersectionKernelFunction<T>::computeKernelMatrix ( const std::vector<std::vector<T> > & X  )
{
  NICE::Matrix K;

  K.resize((int) X.size(), (int) X.size());

  for (int i = 0; i < (int) X.size(); i++)
  {
    for (int j = i; j < (int) X.size(); j++)
    {
      //Kernel matrix has to be symmetric
      K(i,j) = measureDistance(X[i],X[j]);
      K(j,i) = measureDistance(X[i],X[j]);
    }
  }

  return K;
}

template <typename T>
NICE::Matrix GeneralizedIntersectionKernelFunction<T>::computeKernelMatrix ( const std::vector<std::vector<T> > & X , const double & noise)
{
  NICE::Matrix K(computeKernelMatrix(X));
  for (int i = 0; i < (int) X.size(); i++)
    K(i,i) += noise;
  return K;
}

template <typename T>
NICE::Matrix GeneralizedIntersectionKernelFunction<T>::computeKernelMatrix ( const NICE::FeatureMatrixT<T>  & X , const double & noise)
{
  NICE::Matrix K;  
  K.resize(X.get_n(), X.get_n());
  
  //run over every dimension and add the corresponding min-values to the entries in the kernel matrix
  for (int dim = 0; dim < X.get_d(); dim++)
  {
   const std::multimap< double, typename SortedVectorSparse<double>::dataelement> & nonzeroElements = X.getFeatureValues(dim).nonzeroElements();
    
    //compute the min-values (similarities) between every pair in this dimension, zero elements do not influence this
    SortedVectorSparse<double>::const_elementpointer it1 = nonzeroElements.begin();  
    for (; it1 != nonzeroElements.end(); it1++)
    {
      int i(it1->second.first);
      SortedVectorSparse<double>::const_elementpointer it2 = it1;  
      for (; it2 != nonzeroElements.end(); it2++)
      {  
        int j(it2->second.first);
        double val(pow(std::min(it1->second.second, it2->second.second),exponent));
        K(i,j) += val;
        //kernel-matrix has to be symmetric, but avoid adding twice the value to the main-diagonal
        if ( i != j)
          K(j,i) += val;
      } // for-j-loop
    } // for-i-loop
    
  }//dim-loop  
  
  //add noise on the main diagonal
  for (int i = 0; i < (int) X.get_n(); i++)
    K(i,i) += noise;
  return K;
}

template <typename T>
std::vector<double> GeneralizedIntersectionKernelFunction<T>::computeKernelVector ( const std::vector<std::vector<T> > & X , const std::vector<T> & xstar)
{
  std::vector<double> kstar;

  kstar.resize((int) X.size());

  for (int i = 0; i < (int) X.size(); i++)
  {
    kstar[i] = measureDistance(X[i], xstar);
  }

  return kstar;
}

template <typename T>
void GeneralizedIntersectionKernelFunction<T>::sayYourName() 
{
  std::cerr << "I'm the Generalized Intersection Kernel." << std::endl;
}