// /** 
// * @file IntersectionKernelFunction.cpp
// * @brief Implementation for the intersection kernel function as distance measure between two histograms interpreted as vectors 
// * @author Alexander Freytag
// * @date 08-12-2011 (dd-mm-yyyy)
// */
// 
// #include "IntersectionKernelFunction.h"
// 
// using namespace NICE;
// 
// template <typename T>
// IntersectionKernelFunction<T>::IntersectionKernelFunction()
// {
// }
// 
// template <typename T>
// IntersectionKernelFunction<T>::~IntersectionKernelFunction()
// {
// }
// 
// template <typename T>
// double IntersectionKernelFunction<T>::measureDistance ( const NICE::SparseVector & a, const NICE::SparseVector & b  )
// {
//   double distance(0.0);
//   
//   for (NICE::SparseVector::const_iterator itA = a.begin(); itA != a.end(); itA++)
//   {
//     NICE::SparseVector::const_iterator itB = b.find(itA->first);
//     if (itB != b.end())
//       distance += std::min(itA->second , itB->second);
//   }
//   
//   return distance;  
// }
// 
// template <typename T>
// NICE::Matrix IntersectionKernelFunction<T>::computeKernelMatrix ( const std::vector<NICE::SparseVector > & X , const double & noise)
// {
//   NICE::Matrix K;
//   K.resize(X.size(), X.size());
//   K.set(0.0);
//   
//   for (int i = 0; i < X.size(); i++)
//   {
//     for (int j = i; j < X.size(); j++)
//     {
//       K(i,j) = measureDistance(X[i],X[j]);
//       if (i!=j)
//        K(j,i) = K(i,j);
//     }
//   }
//   
//   //add noise on the main diagonal
//   for (int i = 0; i < (int) X.size(); i++)
//     K(i,i) += noise;
//   return K;
// }
// 
// template <typename T>
// std::vector<double> IntersectionKernelFunction<T>::computeKernelVector ( const std::vector<NICE::SparseVector> & X , const NICE::SparseVector & xstar)
// {
//   std::vector<double> kstar;
// 
//   kstar.resize((int) X.size());  
//   
//   for (int i = 0; i < (int) X.size(); i++)
//   {
//     kstar[i] = measureDistance(X[i], xstar);
//   }
// 
//   return kstar;
// }
// 
// template <typename T>
// void IntersectionKernelFunction<T>::sayYourName() 
// {
//   std::cerr << "I'm the Intersection Kernel." << std::endl;
// }