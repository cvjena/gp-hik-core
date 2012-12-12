/** 
* @file VectorSorter.h
* @brief Obsolete: A std::vector coming up with several methods for ordering the elements (Interface and Implementation)
* @author Alexander Freytag
* @date 12/07/2011
*/
#ifndef VECTORSORTERINCLUDE
#define VECTORSORTERINCLUDE

#include <vector>
#include <algorithm>

namespace NICE {
  
 /** 
 * @class VectorSorter
 * @brief Obsolete: A std::vector coming up with several methods for ordering the elements
 * @author Alexander Freytag
 */    
  
  template<class T> class VectorSorter : public std::vector<T>{
    public:
      /** 
      * @brief default constructor
      * @author Alexander Freytag
      * @date 12/07/2011
      */
      VectorSorter() : std::vector<T>() {}

      /** 
      * @brief standard constructor
      * @author Alexander Freytag
      * @date 12/07/2011
      */
      VectorSorter(const VectorSorter<T> &v) : std::vector<T>(v) {}
      
      /** 
      * @brief standard constructor
      * @author Alexander Freytag
      * @date 12/07/2011
      */
      VectorSorter(const std::vector<T> &v) : std::vector<T>(v) {}

      /** 
      * @brief standard destructor
      * @author Alexander Freytag
      * @date 12/07/2011
      */
      ~VectorSorter() {}
      
      /** 
      * @brief comparison of elements
      * @author Alexander Freytag
      * @date 12/07/2011
      */
      bool operator()(int a, int b) { return (*this)[a] < (*this)[b];};
      
      /** 
      * @brief Computes the permutation of the elements for a proper (ascending) ordering
      * @author Alexander Freytag
      * @date 12/07/2011
      */
      std::vector<int> getOrderPermutation()
      {
        std::vector<int> rv((*this).size());
        int idx = 0;
        for (std::vector<int>::iterator i = rv.begin(); i != rv.end(); i++)
        {
          *i = idx++;
        }
        std::sort(rv.begin(), rv.end(), *this);
        return rv;
      };
      
      /** 
      * @brief Orders the elements of the vector in ascending order and stores them in a seperate vector
      * @author Alexander Freytag
      * @date 12/07/2011
      */
      std::vector<T> getOrderInSeparateVector()
      {
        std::vector<T> rv((*this));
        std::sort(rv.begin(), rv.end());
        return rv;
      };
      
      /** 
      * @brief Orders the elements of the vector in ascending order
      * @author Alexander Freytag
      * @date 12/07/2011
      */
      void getOrder()
      {
        std::sort((*this).begin(), (*this).end());
      };

  };
  
} // namespace

#endif
