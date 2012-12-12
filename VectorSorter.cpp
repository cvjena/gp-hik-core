// /** 
// * @file VectorSorter.cpp
// * @brief Interface for a std::vector coming up with several methods for ordering the elements
// * @author Alexander Freytag
// * @date 12/07/2011
// */
// 
// #include "VectorSorter.h"
// 
// using namespace NICE;
// 
// 		/** default constructor*/
// // 		template<class T>
// // 		VectorSorter<T>::VectorSorter() : std::vector<T>() {}
// 		/** standard constructor*/
// // 		template<class T>
// // 		VectorSorter<T>::VectorSorter(const VectorSorter<T> &v) : std::vector<T>(v) {}
// 		/** standard constructor*/
// // 		template<class T>
// // 		VectorSorter<T>::VectorSorter(const std::vector<T> &v) : std::vector<T>(v) {}
// 		/** comparison of elements*/
// 		template<class T>
// 		bool VectorSorter<T>::operator()(int a, int b) { return (*this)[a] < (*this)[b];}
// 		
// 		template<class T>
// 		std::vector<int> VectorSorter<T>::getOrderPermutation()
// 		{
// 			std::vector<int> rv((*this).size());
// 			int idx = 0;
// 			for (std::vector<int>::iterator i = rv.begin(); i != rv.end(); i++)
// 			{
// 				*i = idx++;
// 			}
// 			std::sort(rv.begin(), rv.end(), *this);
// 			return rv;
// 		}
// 		
// 		template<class T>
// 		std::vector<T> VectorSorter<T>::getOrderInSeparateVector()
// 		{
// 			std::vector<T> rv((*this));
// 			std::sort(rv.begin(), rv.end());
// 			return rv;
// 		}
// 		
// 		template<class T>
// 		void VectorSorter<T>::getOrder()
// 		{
// 			std::sort((*this).begin(), (*this).end());
// 		}
