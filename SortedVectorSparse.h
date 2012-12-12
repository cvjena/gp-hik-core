/**
* @file SortedVectorSparse.h
* @brief A sparse vector that is always sorted and keeps index mapping! (Interface and Implementation)
* @author Alexander Freytag
* @date 10-01-2012 (dd-mm-yyyy)
*/
#ifndef SORTEDVECTORSPARSEINCLUDE
#define SORTEDVECTORSPARSEINCLUDE

#include <vector>
#include <cmath>
#include <map>
#include <algorithm>
#include <iostream>
#include <limits>

#include <core/basics/Exception.h>
#include <core/vector/VectorT.h>
#include <core/vector/SparseVectorT.h>
#include "core/basics/Persistent.h"

namespace NICE {

 /** 
 * @class SortedVectorSparse
 * @brief A sparse vector that is always sorted and keeps index mapping!
 * @author Alexander Freytag
 */  
  
template<class T> class SortedVectorSparse : NICE::Persistent{

  public:
    //! original index, transformed feature value
    typedef typename std::pair< int, T > dataelement;
    typedef typename std::multimap< T, dataelement >::iterator elementpointer;
    typedef typename std::multimap< T, dataelement >::const_iterator const_elementpointer;
    typedef typename std::multimap< T, dataelement >::const_reverse_iterator const_reverse_elementpointer;

  protected:
    T tolerance;
    int n;
    
    //! verbose flag for output after calling the restore-function
    bool verbose;

    //! mapping of the original feature value to the index and the transformed feature value
    std::multimap< T, dataelement > nzData;

    //! non zero index mapping, original index -> pointer to the element
    std::map<int, elementpointer > nonzero_indices;

  public:
    /**
    * @brief default constructor
    * @author Alexander Freytag
    * @date 10-01-2012 (dd-mm-yyyy)
    */
    SortedVectorSparse() {
      n = 0;
      tolerance = ( T ) 10e-10;
      verbose = false;
    }

    /**
    * @brief standard constructor
    * @author Alexander Freytag
    * @date 10-01-2012 (dd-mm-yyyy)
    */
    SortedVectorSparse ( const SortedVectorSparse<T> &v ) : nzData ( v.nzData )
    {
      this->tolerance = v.getTolerance();
      this->n = v.getN();
      this->nonzero_indices = v.nonzero_indices;
      this->verbose = v.getVerbose();      
    }

    SortedVectorSparse ( const std::vector<T> &v, const T & _tolerance )
    {
      tolerance = _tolerance;
      n = 0;
      insert ( v );
      verbose = false;
    }

    /**
    * @brief standard destructor
    * @author Alexander Freytag
    * @date 10-01-2012 (dd-mm-yyyy)
    */
    ~SortedVectorSparse() {}

    T getTolerance() const {
      return tolerance;
    };
    int getN() const {
      return n;
    };
    void setTolerance ( const T & _tolerance ) {
      if ( _tolerance < 0 )
        this->tolerance = -_tolerance;
      else
        this->tolerance = _tolerance;
    };


    void setN ( const int & _n ) {
      n = _n;
    };
    int getZeros() const {
      //std::cerr << "n in getZeros: " << n << std::endl;
      return n - nzData.size();
    };
    int getNonZeros() const {
      return nzData.size();
    };

    /**
    * @brief add an element to the vector. If feature number is set, we do not check, wether this feature was already available or not!
    *
    * @param newElement element which will be added
    * @param featureNumber the index of the new element (optional)
    */
    void insert ( const T & newElement, const int & featureNumber = -1 )
    {
      int newIndex ( featureNumber);
      if ( featureNumber < 0)
        newIndex = n;      
      
      if ( !checkSparsity ( newElement ) )
      {
        // element is not sparse
        std::pair<T, dataelement > p ( newElement, dataelement ( newIndex, newElement ) );
        elementpointer it = nzData.insert ( p );
        nonzero_indices.insert ( std::pair<int, elementpointer> ( newIndex, it ) );
      }
      n++;
    }
  
    /**
    * @brief add an element to the vector. If feature number is set, we do not check, wether this feature was already available or not!
    *
    * @param newElement element which will be added
    * @param newElementTransformed transformed feature value
    * @param featureNumber the index of the new element (optional)
    */
    void insert ( const T & newElement, const T & newElementTransformed, const int & featureNumber = -1 )
    {
      int newIndex ( featureNumber);
      if ( featureNumber < 0)
        newIndex = n;
      
      if ( !checkSparsity ( newElement ) )
      {
        // element is not sparse
        
        std::pair<T, dataelement > p ( newElement, dataelement ( newIndex, newElementTransformed ) );
        elementpointer it = nzData.insert ( p );
        nonzero_indices.insert ( std::pair<int, elementpointer> ( newIndex, it ) );
      }
      n++;
    }

    /**
    * @brief add a vector of new elements to the vector 
    *
    * @param v new element which will be added
    */
    void insert ( const std::vector<T> &v )
    {
      for ( uint i = 0; i < v.size(); i++ )
        insert ( v[i] );
    }
    /**
    * @brief add a vector of new elements to the vector. It doesn't make much sense to have such a function, but who knows...
    *
    * @param v Vector of new Elements
    */
    void insert ( const NICE::SparseVector* v )
    {
      for (NICE::SparseVector::const_iterator vIt = v->begin(); vIt != v->end(); vIt++)
      {
        insert((T)vIt->second);
      }
    }
    
    /**
    * @brief non-efficient access to a specific non-zero element
    *
    * @param an index of a non-zero element (not the original index!)
    *
    * @return value of the element (not the original value)
    */
    T accessNonZero ( int a ) const
    {
      const_elementpointer it = nzData.begin();
      advance ( it, a );
      dataelement de = it->second;

      return de.second;
    };

    /**
    * @brief access the transformed value
    *
    * @param a original index of the element
    *
    * @return value of the element
    */
    inline T access ( int a ) const
    {
      typename std::map<int, elementpointer>::const_iterator i = nonzero_indices.find ( a );
      if ( i != nonzero_indices.end() ) {
        // accessing a nonzero element
        const elementpointer & it = i->second;
        const dataelement & de = it->second;
        // we access the transformed value here and not the
        // original one
        return de.second;
      } else {
        // the element is zero
        return ( T ) 0;
      }
    }

    /**
    * @brief access the original value
    *
    * @param a original index of the element
    *
    * @return value of the element
    */
    inline T accessOriginal ( int a ) const
    {
      typename std::map<int, elementpointer>::const_iterator i = nonzero_indices.find ( a );
      if ( i != nonzero_indices.end() ) {
        // accessing a nonzero element
        elementpointer it = i->second;
        return it->first;
      } else {
        // the element is zero
        return ( T ) 0;
      }
    }

    std::multimap< T, dataelement > & nonzeroElements()
    {
      return nzData;
    }

    const std::multimap< T, dataelement > & nonzeroElements() const
    {
      return nzData;
    }

    const std::map< int, elementpointer> & nonzeroIndices() const
    {
      return nonzero_indices;
    }

    /**
    * @brief check whether the elment is sparse with the given tolerance
    *
    * @param element
    *
    * @return
    */
    bool checkSparsity ( T element )
    {
      if ( element > tolerance )
        return false;
      if ( element < -tolerance )
        return false;

      return true;
    }

    /**
    * @brief set a specific element. A boolean flag controls
    * whether we set the transformed value or the original value. Setting the original
    * value (default case) is highly inefficient. Setting the transformed value is appropiate
    * when applying an order preserving transformation.
    *
    * @param a proper index
    * @param newElement element value
    */
    void set ( int a, T newElement, bool setTransformedValue = false )
    {
      if ( a >= n || a < 0 )
        fthrow ( Exception, "SortedVectorSparse::set(): out of bounds" );

      typename std::map<int, elementpointer>::iterator i = nonzero_indices.find ( a );

      // check whether the element was previously non-sparse
      if ( i != nonzero_indices.end() ) {
        elementpointer it = i->second;

        if ( checkSparsity ( newElement ) ) {
          // old: non-sparse, new:sparse
          // delete the element
          nzData.erase ( it );
          nonzero_indices.erase ( i );
        } else {
          // old: non-sparse, new: non-sparse
          // The following statement would be nice, but it is not allowed.
          // This is also the reason why we implemented the transformed feature value ability.
          // it->first = newElement;
          if ( setTransformedValue ) {
            // set the transformed value
            it->second.second = newElement;
          } else {
            // the following is a weird tricky and expensive
            set ( a, 0.0 );
            //std::cerr << "Element after step 1: " << access(a) << std::endl;
            set ( a, newElement );
          }
          //std::cerr << "Element after step 2: " << access(a) << std::endl;
        }
      } else {
        // the element was previously sparse
        if ( !checkSparsity ( newElement ) )
        {
          //std::cerr << "changing a zero value to a non-zero value " << newElement << std::endl;
          // old element is not sparse
          dataelement de ( a, newElement );
          std::pair<T, dataelement> p ( newElement, de );
          elementpointer it = nzData.insert ( p );
          nonzero_indices.insert ( std::pair<int, elementpointer> ( a, it ) );
        }
      }
    }

    SortedVectorSparse<T> operator= ( const SortedVectorSparse<T> & F )
    {
      this->tolerance = F.getTolerance();
      this->n = F.getN();
      this->nonzero_indices = F.nonzero_indices;
      this->nzData = F.nzData;

      return *this;
    }

    /**
    * @brief Computes the permutation of the non-zero elements for a proper (ascending) ordering
    * @author Alexander Freytag
    * @date 10-01-2012 (dd-mm-yyyy)
    */
    std::vector<int> getPermutationNonZero() const
    {
      std::vector<int> rv ( nzData.size() );
      int idx = 0;
      for ( typename std::multimap<T, dataelement>::const_iterator it = nzData.begin(); it != nzData.end(); it++, idx++ )
      {
        rv[idx] = it->second.first;
      }
      return rv;
    };

    /**
    * @brief Computes the permutation of the non-zero elements for a proper (ascending) ordering
    * @author Alexander Freytag
    * @date 23-01-2012 (dd-mm-yyyy)
    * @return  std::map<int, int>, with the absolute feature numbers as key element and their permutation as second
    */
    std::map<int, int> getPermutationNonZeroReal() const
    {
      std::map<int, int> rv;
//         int idx = 0;
//         for (typename std::multimap<T, dataelement>::const_iterator it = nzData.begin(); it != nzData.end(); it++, idx++)
//         {
//           //inserts the real feature number as key
//           //TODO DO not insert the feature, but its original index, which is stored somewhere else!
//           rv.insert(std::pair<int,int>(it->second.first,it->second.second));
//           std::cerr << "inserting: " << it->second.first << " - " << it->second.second << std::endl;
//           //if we want to use the relative feature number (realtive to non-zero elements), use the following
//           //rv.insert(std::pair<int,int>(idx,it->second.first));
//         }

      int nrZeros ( this->getZeros() );

      int idx = 0;
      for ( typename std::multimap<T, dataelement>::const_iterator it = nzData.begin(); it != nzData.end(); it++, idx++ )
      {
        //inserts the real feature number as key
        rv.insert ( std::pair<int, int> ( nrZeros + idx, it->second.first ) );
      }
      return rv;
    };

    /**
    * @brief Computes the permutation of the non-zero elements for a proper (ascending) ordering
    * @author Alexander Freytag
    * @date 23-01-2012 (dd-mm-yyyy)
    * @return  std::map<int, int>, with the relative feature numbers as key element  (realtive to non-zero elements) and their permutation as second
    */
    std::map<int, int> getPermutationNonZeroRelative() const
    {
      std::map<int, int> rv;
      int idx = 0;
      for ( typename std::multimap<T, dataelement>::const_iterator it = nzData.begin(); it != nzData.end(); it++, idx++ )
      {
        //inserts the real feature number as key
        //rv.insert(std::pair<int,int>(it->second.first,it->second.first));
        //if we want to use the relative feature number (realtive to non-zero elements), use the following
        rv.insert ( std::pair<int, int> ( idx, it->second.first ) );
      }
      return rv;
    };



    /**
    * @brief Computes the permutation of the elements for a proper (ascending) ordering
    */
    std::vector<int> getPermutation() const
    {
      std::vector<int> rv ( n );

      int idx = n - 1;
      typename std::multimap<T, dataelement>::const_reverse_iterator it ;
      for ( it = nzData.rbegin(); it != nzData.rend() && ( it->first > tolerance ); it++, idx-- )
      {
        rv[ idx ] = it->second.first;
      }

      for ( int i = n - 1 ; i >= 0 ; i-- )
        if ( nonzero_indices.find ( i ) == nonzero_indices.end() )
        {
          rv[ idx ] = i;
          idx--;
        }

      for ( ; it != nzData.rend(); it++, idx-- )
      {
        rv[ idx ] = it->second.first;
      }

      return rv;
    };

    /**
    * @brief Orders the elements of the vector in ascending order and stores them in a seperate vector
    * @author Alexander Freytag
    * @date 10-01-2012 (dd-mm-yyyy)
    */
    std::vector<std::pair<int, T> > getOrderInSeparateVector() const
    {
      std::vector<std::pair<int, T> > rv;
      rv.resize ( nzData.size() );
      uint idx = 0;
      for ( typename std::multimap<T, dataelement>::const_iterator it = nzData.begin(); it != nzData.end(); it++, idx++ )
      {
        rv[idx].first = it->second.first;
        rv[idx].second = it->second.second;
      }
      return rv;
    };

    /**
    * @brief get the median of the vector including zero elements
    *
    * @return return the median value
    */
    T getMedian() const
    {
      if ( n % 2 == 1 ) 
      {
        // even number of training examples
        int medianPosition = nzData.size() - (int)(n/2);
        if ( medianPosition < 0 ) 
          return 0.0;
        else
          return accessNonZero(medianPosition); 
      } else {
        // odd number of training examples
        int medianA = nzData.size() - (int)(n/2);
        int medianB = nzData.size() - (int)((n+1)/2);
        T a = 0.0;
        T b = 0.0;
        if ( medianA >= 0)
          a = accessNonZero( medianA );
        if ( medianB >= 0)
          b = accessNonZero( medianB );
        return (a + b)/2.0;
      }
    }
    
    /**
    * @brief get the maximum of the vector including zero elements
    *
    * @return return the median value
    */
    T getMax() const
    {
      if (nzData.size() > 0)
        return accessNonZero(nzData.size()-1);
      return (T) 0.0;
    }
    
    /**
    * @brief get the minimum of the vector including zero elements
    *
    * @return return the median value
    */
    T getMin() const
    {
      if (nzData.size() < (uint) n)
        return (T) 0.0;
      return accessNonZero(0);
    }
    
    

    /**
    * @brief get median feature values for each class seperately, we do not apply averaging when the number
    * of examples is even
    *
    * @param classMedians resulting sparse vector, i.e. classMedians[classno] is the median value 
    * of every example of class classno
    * @param labels vector of labels with the same size n as the current vector
    * @param elementCounts this vector contains the number of examples for each class, compute this using the labels
    * for efficiency reasons
    */
    void getClassMedians ( SparseVector & classMedians, const Vector & labels, const Vector & elementCounts ) const
    {
      if ( labels.size() != n )
        fthrow(Exception, "Label vector has to have the same size as the SortedVectorSparse structure");
      Vector c ( elementCounts );
      for ( uint i = 0 ; i < c.size(); i++ )
        c[i] /= 2;
      // now we have in c the position of the current median
      typename std::multimap<T, dataelement>::const_reverse_iterator it;

      for ( it = nzData.rbegin(); it != nzData.rend(); it++ )
      {
        const dataelement & de = it->second;
        int origIndex = de.first;
        double value = de.second;
        int classno = labels[origIndex];
        c[ classno ]--;
        if ( c[classno] == 0 )
          classMedians[classno] = value;
      }

      // remaining medians are zero!
      for ( uint classno = 0 ; classno < c.size(); classno++ )
        if ( c[classno] > 0 )
          classMedians[classno] = 0.0;
    }

    /**
    * @brief Print the content of the sparse vector
    * @author Alexander Freytag
    * @date 12-01-2012 (dd-mm-yyyy)
    */
    void print(std::ostream & os) const
    {
      typename std::multimap<T, dataelement>::const_iterator it = nzData.begin();

      if (os.good())
      {
        for ( ; it != nzData.end() ; it++ )
        {
          if ( it->first < ( T ) 0.0 )
            os << it->first << " ";
          else
            break;
        }

        for ( int i = 0; i < getZeros(); i++ )
        {
          os << ( T ) 0.0 << " " ;
        }

        for ( ; ( it != nzData.end() ); it++ )
        {
          os << it->second.second << " ";
        }
        os << std::endl;
      }
    }
    
    /** set verbose flag used for restore-functionality*/
    void setVerbose( const bool & _verbose) { verbose = _verbose;};
    bool getVerbose( ) const { return verbose;};
    
    
    /** Persistent interface */
    virtual void restore ( std::istream & is, int format = 0 )
    {
      if (is.good())
      {
        is.precision (std::numeric_limits<double>::digits10 + 1);
        
        std::string tmp;
        is >> tmp; //class name
        
        is >> tmp;
        is >> tolerance;
               
        is >> tmp;
        is >> n;
               
        is >> tmp;
        int size;
        is >> size;
        
        is >> tmp;
        
        T origValue;
        int origIndex;
        T transformedValue;
        
        nzData.clear();
        for (int i = 0; i < size; i++)
        {
         
          is >> origValue;
          is >> origIndex;
          is >> transformedValue;
        
          std::pair<T, dataelement > p ( origValue, dataelement ( origIndex, transformedValue ) );
          elementpointer it = nzData.insert ( p);
          nonzero_indices.insert ( std::pair<int, elementpointer> ( origIndex, it ) );
        }
        
        if (verbose)
        {
          std::cerr << "SortedVectorSparse::restore" << std::endl;      
          std::cerr << "tolerance: " << tolerance << std::endl;          
          std::cerr << "n: " << n << std::endl;          
          std::cerr << "size: " << size << std::endl;          
        }
      }
      else
      {
        std::cerr << "SortedVectorSparse::restore -- InStream not initialized - restoring not possible!" << std::endl;
      }      
    };
    virtual void store ( std::ostream & os, int format = 0 ) const
    {
      if (os.good())
      {
        os.precision (std::numeric_limits<double>::digits10 + 1);
        os << "SortedVectorSparse" << std::endl;
        os << "tolerance: " << tolerance << std::endl;
        os << "n: " << n << std::endl;
        os << "nonZeros: " << nzData.size() << std::endl;
        os << "underlying_data_(sorted)" << std::endl;
        for (const_elementpointer elP = nzData.begin();  elP != nzData.end(); elP++)
        {
          os << elP->first << " " << elP->second.first << " " << elP->second.second << " ";
        }
        os << std::endl;
      }
      else
      {
        std::cerr << "SortedVectorSparse::store -- OutStream not initialized - storing not possible!" << std::endl;
      }      
    };    
    
    virtual void clear (){};
};

} // namespace

#endif
