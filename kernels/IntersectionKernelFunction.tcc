/** 
* @file IntersectionKernelFunction.cpp
* @brief The intersection kernel function as distance measure between two histograms interpreted as vectors (Implementation)
* @author Alexander Freytag
* @date 08-12-2011 (dd-mm-yyyy)
*/

#include "IntersectionKernelFunction.h"

#include <gp-hik-core/SortedVectorSparse.h>

using namespace NICE;

template <typename T>
double IntersectionKernelFunction<T>::measureDistance ( const std::vector<T> & a, const std::vector<T> & b  )
{
  int size( (int) a.size());
  if ((int) b.size() < size)
    size = (int) b.size();

  double distance(0.0);

  for (int i = 0; i < size; i++)
  {
    if ( a[i] < b[i])
      distance += (double) a[i];
    else
      distance += (double) b[i];
  }
  return distance;
}

template <typename T>
NICE::Matrix IntersectionKernelFunction<T>::computeKernelMatrix ( const std::vector<std::vector<T> > & X  )
{
  NICE::Matrix K;
  
  K.resize((int) X.size(), (int) X.size());
  
  double valTmp;
  for (int i = 0; i < (int) X.size(); i++)
  {
    for (int j = i; j < (int) X.size(); j++)
    {
      valTmp = measureDistance(X[i],X[j]);
      //Kernel matrix has to be symmetric
      K(i,j) = valTmp;
      //kernel-matrix has to be symmetric, but avoid adding twice the value to the main-diagonal
      if ( i != j)
        K(j,i) = valTmp;
    }
  }

  return K;
}

template <typename T>
NICE::Matrix IntersectionKernelFunction<T>::computeKernelMatrix ( const std::vector<std::vector<T> > & X , const double & noise)
{
  NICE::Matrix K(computeKernelMatrix(X));
  for (int i = 0; i < (int) X.size(); i++)
    K(i,i) += noise;
  return K;
}

template <typename T>
NICE::Matrix IntersectionKernelFunction<T>::computeKernelMatrix ( const NICE::FeatureMatrixT<T>  & X , const double & noise)
{
  NICE::Matrix K;  
  K.resize(X.get_n(), X.get_n());
  K.set(0.0);
  
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
        double val(std::min(it1->second.second, it2->second.second));
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
std::vector<double> IntersectionKernelFunction<T>::computeKernelVector ( const std::vector<std::vector<T> > & X , const std::vector<T> & xstar)
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
IntersectionKernelFunction<T>::IntersectionKernelFunction()
{
}

template <typename T>
IntersectionKernelFunction<T>::~IntersectionKernelFunction()
{
}

template <typename T>
double IntersectionKernelFunction<T>::measureDistance ( const NICE::SparseVector & a, const NICE::SparseVector & b  )
{
  double distance(0.0);
  
  for (NICE::SparseVector::const_iterator itA = a.begin(); itA != a.end(); itA++)
  {
    NICE::SparseVector::const_iterator itB = b.find(itA->first);
    if (itB != b.end())
      distance += std::min(itA->second , itB->second);
  }
  
  return distance;  
}

template <typename T>
NICE::Matrix IntersectionKernelFunction<T>::computeKernelMatrix ( const std::vector<NICE::SparseVector > & X , const double & noise)
{
  
  std::cerr << "compute Kernel Matrix with vector of sparse vectors called " << std::endl;
  NICE::Matrix K;
  std::cerr << "NICE::Matrix initialized" << std::endl;
  std::cerr << "now perform the resize to: "<< X.size() << std::endl;
  K.resize(X.size(), X.size());
  std::cerr << "NICE::matrix set to size : " << X.size() << std::endl;
  K.set(0.0);
  std::cerr << "set entries to zero" << std::endl; 
  
  std::cerr << "compute Kernel Matrix" << std::endl;
  for (int i = 0; i < X.size(); i++)
  {
    std::cerr << i << " / " << X.size() << std::endl;
    for (int j = i; j < X.size(); j++)
    {
      K(i,j) = measureDistance(X[i],X[j]);
      if (i!=j)
       K(j,i) = K(i,j);
    }
  }
  std::cerr << "compute kernel matrix done" << std::endl;
  
  //add noise on the main diagonal
  for (int i = 0; i < (int) X.size(); i++)
    K(i,i) += noise;
  return K;
}

template <typename T>
NICE::Vector IntersectionKernelFunction<T>::computeKernelVector ( const std::vector<NICE::SparseVector> & X , const NICE::SparseVector & xstar)
{
  NICE::Vector kstar;

  kstar.resize((int) X.size());  
  
  if (X.size() > 0)
  {
    for (int i = 0; i < (int) X.size(); i++)
    {
      kstar[i] = measureDistance(X[i], xstar);
    }  
  }
  
  return kstar;
}

template <typename T>
void IntersectionKernelFunction<T>::sayYourName() 
{
  std::cerr << "I'm the Intersection Kernel." << std::endl;
}
