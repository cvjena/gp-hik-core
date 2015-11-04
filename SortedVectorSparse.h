/**
* @file SortedVectorSparse.h
* @brief A sparse vector that is always sorted and keeps index mapping! (Interface and Implementation)
* @author Alexander Freytag
* @date 10-01-2012 (dd-mm-yyyy)
*/
#ifndef SORTEDVECTORSPARSEINCLUDE
#define SORTEDVECTORSPARSEINCLUDE

// STL includes
#include <vector>
#include <cmath>
#include <map>
#include <algorithm>
#include <iostream>
#include <limits>

// NICE-core includes
#include <core/basics/Exception.h>
#include <core/basics/Persistent.h>
//
#include <core/vector/VectorT.h>
#include <core/vector/SparseVectorT.h>


namespace NICE {

 /**
 * @class SortedVectorSparse
 * @brief A sparse vector that is always sorted and keeps index mapping!
 * @author Alexander Freytag
 */

template<class T> class SortedVectorSparse : NICE::Persistent{

  public:
    //! original index, transformed feature value
    typedef typename std::pair< uint, T > dataelement;
    typedef typename std::multimap< T, dataelement >::iterator elementpointer;
    typedef typename std::multimap< T, dataelement >::const_iterator const_elementpointer;
    typedef typename std::multimap< T, dataelement >::const_reverse_iterator const_reverse_elementpointer;

  protected:
    T tolerance;
    uint ui_n;

    //! b_verbose flag for output after calling the restore-function
    bool b_verbose;

    //! mapping of the original feature value to the index and the transformed feature value
    std::multimap< T, dataelement > nzData;

    //! non zero index mapping, original index -> pointer to the element
    std::map<uint, elementpointer > nonzero_indices;

  public:
    /**
    * @brief default constructor
    * @author Alexander Freytag
    * @date 10-01-2012 (dd-mm-yyyy)
    */
    SortedVectorSparse() {
      this->ui_n = 0;
      this->tolerance = ( T ) 10e-10;
      this->b_verbose = false;
    }

    /**
    * @brief standard constructor
    * @author Alexander Freytag
    * @date 10-01-2012 (dd-mm-yyyy)
    */
    SortedVectorSparse ( const SortedVectorSparse<T> &_v ) : nzData ( _v.nzData )
    {
      this->tolerance = _v.getTolerance();
      this->ui_n = _v.getN();
      this->nonzero_indices = _v.nonzero_indices;
      this->b_verbose = _v.getVerbose();
    }

    SortedVectorSparse ( const std::vector<T> &_v, const T & _tolerance )
    {
      this->tolerance = _tolerance;
      this->ui_n = 0;
      this->insert ( _v );
      this->b_verbose = false;
    }

    /**
    * @brief standard destructor
    * @author Alexander Freytag
    * @date 10-01-2012 (dd-mm-yyyy)
    */
    ~SortedVectorSparse() {}

    T getTolerance() const {
      return this->tolerance;
    };
    uint getN() const {
      return this->ui_n;
    };
    void setTolerance ( const T & _tolerance ) {
      if ( _tolerance < 0 )
        this->tolerance = -_tolerance;
      else
        this->tolerance = _tolerance;
    };


    void setN ( const uint & _n ) {
      this->ui_n = _n;
    };
    uint getZeros() const {
      //std::cerr << "n in getZeros: " << n << std::endl;
      return this->ui_n - this->nzData.size();
    };
    uint getNonZeros() const {
      return this->nzData.size();
    };

    /**
    * @brief add an element to the vector. If feature number is set, we do not check, wether this feature was already available or not!
    *
    * @param _newElement element which will be added
    * @param _specifyFeatureNumber specify whether to use the optinally given index
    * @param _featureNumber the index of the new element (optional)
    */
    void insert ( const T & _newElement,
                  const bool _specifyFeatureNumber = false,
                  const uint & _featureNumber = 0
                )
    {

      uint newIndex ( this->ui_n );
      if ( _specifyFeatureNumber )
        newIndex = _featureNumber;

      if ( !checkSparsity ( _newElement ) )
      {
        // element is not sparse
        std::pair<T, dataelement > p ( _newElement, dataelement ( newIndex, _newElement ) );
        elementpointer it = this->nzData.insert ( p );
        this->nonzero_indices.insert ( std::pair<uint, elementpointer> ( newIndex, it ) );
      }
      this->ui_n++;
    }

    /**
    * @brief add an element to the vector. If feature number is set, we do not check, wether this feature was already available or not!
    *
    * @param newElement element which will be added
    * @param newElementTransformed transformed feature value
    * @param _specifyFeatureNumber specify whether to use the optinally given index*
    * @param featureNumber the index of the new element (optional)
    */
    void insert ( const T & _newElement,
                  const T & _newElementTransformed,
                  const bool _specifyFeatureNumber = false,
                  const uint & _featureNumber = 0
                )
    {
      uint newIndex ( this->ui_n );
      if ( _specifyFeatureNumber )
        newIndex = _featureNumber;

      if ( !checkSparsity ( _newElement ) )
      {
        // element is not sparse

        std::pair<T, dataelement > p ( _newElement, dataelement ( newIndex,_newElementTransformed ) );
        elementpointer it = this->nzData.insert ( p );
        this->nonzero_indices.insert ( std::pair<uint, elementpointer> ( newIndex, it ) );
      }
      this->ui_n++;
    }

    /**
    * @brief add a vector of new elements to the vector
    *
    * @param v new element which will be added
    */
    void insert ( const std::vector<T> &_v )
    {
      for ( uint i = 0; i < _v.size(); i++ )
        this->insert ( _v[i] );
    }
    /**
    * @brief add a vector of new elements to the vector. It doesn't make much sense to have such a function, but who knows...
    *
    * @param v Vector of new Elements
    */
    void insert ( const NICE::SparseVector* _v )
    {
      for (NICE::SparseVector::const_iterator vIt = _v->begin(); vIt != _v->end(); vIt++)
      {
        this->insert((T)vIt->second);
      }
    }

    /**
    * @brief non-efficient access to a specific non-zero element
    *
    * @param an index of a non-zero element (not the original index!)
    *
    * @return value of the element (not the original value)
    */
    T accessNonZero ( uint _a ) const
    {
      const_elementpointer it = this->nzData.begin();
      advance ( it, _a );
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
    inline T access ( uint _a ) const
    {
      typename std::map<uint, elementpointer>::const_iterator i = this->nonzero_indices.find ( _a );
      if ( i != this->nonzero_indices.end() ) {
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
    inline T accessOriginal ( uint _a ) const
    {
      typename std::map<uint, elementpointer>::const_iterator i = this->nonzero_indices.find ( _a );
      if ( i != this->nonzero_indices.end() ) {
        // accessing a nonzero element
        elementpointer it = i->second;
        return it->first;
      } else {
        // the element is zero
        return ( T ) 0;
      }
    }

    inline T getLargestValueUnsafe ( const double & _quantile = 1.0,
                                     const bool & _getTransformedValue = false
                                   ) const
    {
      if ( this->getNonZeros() == 0)
      {
        return  0.0;
      }

        uint idxDest ( round ( (this->getNonZeros() - 1) * _quantile)  );

        if ( _quantile > 0.5 )
        {
          typename std::multimap< T, dataelement >::const_reverse_iterator it = this->nzData.rend();

          //
          // take as many backward steps as indicated by _quantile
          for ( uint idx = this->getNonZeros(); idx > idxDest; idx-- )
          {
            it++;
          }
          // alternative usage for random access iterators:
          // it = it + (uint) this->getNonZeros() * ( 1.0 -  _quantile );

            if ( _getTransformedValue )
              return it->second.second;
            else
              return it->first;
        }
        else
        {
          typename std::multimap< T, dataelement >::const_iterator it = this->nzData.begin();

          // take as many steps forward as indicated by _quantile
          for ( uint idx = 0; idx < idxDest; idx++ )
          {
            it++;
          }
          // alternative usage for random access iterators:
          // it = it + (uint) this->getNonZeros() * _quantile;

          if ( _getTransformedValue )
            return it->second.second;
          else
            return it->first;
        }
    }

    inline T getLargestTransformedValueUnsafe ( const double & _quantile = 1.0 ) const
    {
        uint idxDest ( round ( (this->getNonZeros() - 1) * _quantile )  );

        if ( _quantile > 0.5 )
        {
          typename std::multimap< T, dataelement >::const_reverse_iterator it = this->nzData.rend();

          //
          // take as many backward steps as indicated by _quantile
          for ( uint idx = this->getNonZeros(); idx > idxDest; idx-- )
          {
            it++;
          }
          // alternative usage for random access iterators:
          // it = it + (uint) this->getNonZeros() * ( 1.0 -  _quantile );

          return it->second.second;
        }
        else
        {
          typename std::multimap< T, dataelement >::const_iterator it = this->nzData.begin();

          // take as many steps forward as indicated by _quantile
          for ( uint idx = 0; idx < idxDest; idx++ )
          {
            it++;
          }
          // alternative usage for random access iterators:
          // it = it + (uint) this->getNonZeros() * _quantile;

          return it->second.second;
        }
    }

    std::multimap< T, dataelement > & nonzeroElements()
    {
      return this->nzData;
    }

    const std::multimap< T, dataelement > & nonzeroElements() const
    {
      return this->nzData;
    }

    const std::map< uint, elementpointer> & nonzeroIndices() const
    {
      return this->nonzero_indices;
    }

    /**
    * @brief check whether the elment is sparse with the given tolerance
    *
    * @param element
    *
    * @return
    */
    bool checkSparsity ( T _element )
    {
      if ( _element > this->tolerance )
        return false;
      if ( _element < -this->tolerance )
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
    void set ( uint _a,
               T _newElement,
               bool _setTransformedValue = false )
    {
      if ( _a >= this->ui_n || _a < 0 )
        fthrow ( Exception, "SortedVectorSparse::set(): out of bounds" );

      typename std::map<uint, elementpointer>::iterator i = this->nonzero_indices.find ( _a );

      // check whether the element was previously non-sparse
      if ( i != this->nonzero_indices.end() ) {
        elementpointer it = i->second;

        if ( checkSparsity ( _newElement ) ) {
          // old: non-sparse, new:sparse
          // delete the element
          this->nzData.erase ( it );
          this->nonzero_indices.erase ( i );
        } else {
          // old: non-sparse, new: non-sparse
          // The following statement would be nice, but it is not allowed.
          // This is also the reason why we implemented the transformed feature value ability.
          // it->first = newElement;
          if ( _setTransformedValue ) {
            // set the transformed value
            it->second.second = _newElement;
          } else {
            // the following is a weird tricky and expensive
            this->set ( _a, 0.0 );
            //std::cerr << "Element after step 1: " << access(a) << std::endl;
            this->set ( _a, _newElement );
          }
          //std::cerr << "Element after step 2: " << access(a) << std::endl;
        }
      } else {
        // the element was previously sparse
        if ( !checkSparsity ( _newElement ) )
        {
          //std::cerr << "changing a zero value to a non-zero value " << newElement << std::endl;
          // old element is not sparse
          dataelement de ( _a, _newElement );
          std::pair<T, dataelement> p ( _newElement, de );
          elementpointer it = this->nzData.insert ( p );
          this->nonzero_indices.insert ( std::pair<uint, elementpointer> ( _a, it ) );
        }
      }
    }

    SortedVectorSparse<T> operator= ( const SortedVectorSparse<T> & _F )
    {
      this->tolerance = _F.getTolerance();
      this->ui_n = _F.getN();
      this->nonzero_indices = _F.nonzero_indices;
      this->nzData = _F.nzData;

      return *this;
    }

    /**
    * @brief Computes the permutation of the non-zero elements for a proper (ascending) ordering
    * @author Alexander Freytag
    * @date 10-01-2012 (dd-mm-yyyy)
    */
    std::vector<uint> getPermutationNonZero() const
    {
      std::vector<uint> rv ( this->nzData.size() );
      uint idx = 0;
      for ( typename std::multimap<T, dataelement>::const_iterator it = this->nzData.begin(); it != this->nzData.end(); it++, idx++ )
      {
        rv[idx] = it->second.first;
      }
      return rv;
    };

    /**
    * @brief Computes the permutation of the non-zero elements for a proper (ascending) ordering
    * @author Alexander Freytag
    * @date 23-01-2012 (dd-mm-yyyy)
    * @return  std::map<uint, uint>, with the absolute feature numbers as key element and their permutation as second
    */
    std::map<uint, uint> getPermutationNonZeroReal() const
    {
      std::map<uint, uint> rv;
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

      uint nrZeros ( this->getZeros() );

      uint idx = 0;
      for ( typename std::multimap<T, dataelement>::const_iterator it = this->nzData.begin(); it != this->nzData.end(); it++, idx++ )
      {
        //inserts the real feature number as key
        rv.insert ( std::pair<uint, uint> ( nrZeros + idx, it->second.first ) );
      }
      return rv;
    };

    /**
    * @brief Computes the permutation of the non-zero elements for a proper (ascending) ordering
    * @author Alexander Freytag
    * @date 23-01-2012 (dd-mm-yyyy)
    * @return  std::map<uint, uint>, with the relative feature numbers as key element  (realtive to non-zero elements) and their permutation as second
    */
    std::map<uint, uint> getPermutationNonZeroRelative() const
    {
      std::map<uint, uint> rv;
      uint idx = 0;
      for ( typename std::multimap<T, dataelement>::const_iterator it = this->nzData.begin(); it != this->nzData.end(); it++, idx++ )
      {
        //inserts the real feature number as key
        //rv.insert(std::pair<int,int>(it->second.first,it->second.first));
        //if we want to use the relative feature number (realtive to non-zero elements), use the following
        rv.insert ( std::pair<uint, uint> ( idx, it->second.first ) );
      }
      return rv;
    };



    /**
    * @brief Computes the permutation of the elements for a proper (ascending) ordering
    */
    std::vector<uint> getPermutation() const
    {
      std::vector<uint> rv ( this->ui_n );

      uint idx = std::max( this->ui_n - 1, (uint) 0 );
      typename std::multimap<T, dataelement>::const_reverse_iterator it ;
      for ( it = this->nzData.rbegin(); it != this->nzData.rend() && ( it->first > tolerance ); it++, idx-- )
      {
        rv[ idx ] = it->second.first;
      }

      uint i = std::max( this->ui_n - 1, (uint) 0 );
      for ( int iCnt = this->ui_n - 1 ; iCnt >= 0 ; i--, iCnt-- )
        if ( nonzero_indices.find ( i ) == nonzero_indices.end() )
        {
          rv[ idx ] = i;
          idx--;
        }

      for ( ; it != this->nzData.rend(); it++, idx-- )
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
    std::vector<std::pair<uint, T> > getOrderInSeparateVector() const
    {
      std::vector<std::pair<uint, T> > rv;
      rv.resize ( this->nzData.size() );
      uint idx = 0;
      for ( typename std::multimap<T, dataelement>::const_iterator it = this->nzData.begin(); it != this->nzData.end(); it++, idx++ )
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
      if ( this->ui_n % 2 == 1 )
      {
        // even number of training examples
        uint medianPosition = this->nzData.size() - this->ui_n/2;
        if ( medianPosition < 0 ) //FIXME not possible with uint anymore
          return 0.0;
        else
          return this->accessNonZero(medianPosition);
      } else {
        // odd number of training examples
        uint medianA = this->nzData.size() - this->ui_n/2;
        uint medianB = this->nzData.size() - (this->ui_n+1)/2;
        T a = 0.0;
        T b = 0.0;
        if ( medianA >= 0)
          a = this->accessNonZero( medianA );
        if ( medianB >= 0)
          b = this->accessNonZero( medianB );
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
      if ( this->nzData.size() > 0 )
        return this->accessNonZero( this->nzData.size()-1 );
      return (T) 0.0;
    }

    /**
    * @brief get the minimum of the vector including zero elements
    *
    * @return return the median value
    */
    T getMin() const
    {
      if ( this->nzData.size() < this->ui_n )
        return (T) 0.0;
      return this->accessNonZero(0);
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
    void getClassMedians ( SparseVector & _classMedians,
                           const Vector & _labels,
                           const Vector & _elementCounts
                         ) const
    {
      if ( _labels.size() != this->ui_n )
        fthrow(Exception, "Label vector has to have the same size as the SortedVectorSparse structure");
      Vector c ( _elementCounts );
      for ( uint i = 0 ; i < c.size(); i++ )
        c[i] /= 2;
      // now we have in c the position of the current median
      typename std::multimap<T, dataelement>::const_reverse_iterator it;

      for ( it = this->nzData.rbegin(); it != this->nzData.rend(); it++ )
      {
        const dataelement & de = it->second;
        uint origIndex = de.first;
        double value = de.second;
        int classno = _labels[origIndex];
        c[ classno ]--;
        if ( c[classno] == 0 )
          _classMedians[classno] = value;
      }

      // remaining medians are zero!
      for ( uint classno = 0 ; classno < c.size(); classno++ )
        if ( c[classno] > 0 )
          _classMedians[classno] = 0.0;
    }

    /**
    * @brief Print the content of the sparse vector
    * @author Alexander Freytag
    * @date 12-01-2012 (dd-mm-yyyy)
    */
    void print(std::ostream & _os) const
    {
      typename std::multimap<T, dataelement>::const_iterator it = nzData.begin();

      if (_os.good())
      {
        for ( ; it != nzData.end() ; it++ )
        {
          if ( it->first < ( T ) 0.0 )
            _os << it->first << " ";
          else
            break;
        }

        for ( int i = 0; i < this->getZeros(); i++ )
        {
          _os << ( T ) 0.0 << " " ;
        }

        for ( ; ( it != this->nzData.end() ); it++ )
        {
          _os << it->second.second << " ";
        }
        _os << std::endl;
      }
    }

    /** set b_verbose flag used for restore-functionality*/
    void setVerbose( const bool & _verbose) { this->b_verbose = _verbose;};
    bool getVerbose( ) const { return this->b_verbose;};


    /** Persistent interface */
    virtual void restore ( std::istream & _is,
                           int _format = 0
                         )
    {
      bool b_restoreVerbose ( false );
      if ( _is.good() )
      {
        if ( b_restoreVerbose )
          std::cerr << " restore SortedVectorSparse" << std::endl;

        std::string tmp;
        _is >> tmp; //class name

        if ( ! this->isStartTag( tmp, "SortedVectorSparse" ) )
        {
            std::cerr << " WARNING - attempt to restore SortedVectorSparse, but start flag " << tmp << " does not match! Aborting... " << std::endl;
            throw;
        }

        _is.precision ( std::numeric_limits<double>::digits10 + 1);

        bool b_endOfBlock ( false ) ;

        while ( !b_endOfBlock )
        {
          _is >> tmp; // start of block

          if ( this->isEndTag( tmp, "SortedVectorSparse" ) )
          {
            b_endOfBlock = true;
            continue;
          }

          tmp = this->removeStartTag ( tmp );

          if ( b_restoreVerbose )
            std::cerr << " currently restore section " << tmp << " in SortedVectorSparse" << std::endl;

          if ( tmp.compare("tolerance") == 0 )
          {
            _is >> this->tolerance;
            _is >> tmp; // end of block
            tmp = this->removeEndTag ( tmp );
          }
          else if ( tmp.compare("ui_n") == 0 )
          {
            _is >> this->ui_n;
            _is >> tmp; // end of block
            tmp = this->removeEndTag ( tmp );
          }
          else if ( tmp.compare("underlying_data_(sorted)") == 0 )
          {
            _is >> tmp; // start of block

            uint nonZeros;
            if ( ! this->isStartTag( tmp, "nonZeros" ) )
            {
              std::cerr << "Attempt to restore SortedVectorSparse, but found no information about nonZeros elements. Aborting..." << std::endl;
              throw;
            }
            else
            {
              _is >> nonZeros;
              _is >> tmp; // end of block
              tmp = this->removeEndTag ( tmp );
            }

            _is >> tmp; // start of block

            if ( ! this->isStartTag( tmp, "data" ) )
            {
              std::cerr << "Attempt to restore SortedVectorSparse, but found no data. Aborting..." << std::endl;
              throw;
            }
            else
            {
              T origValue;
              uint origIndex;
              T transformedValue;

              this->nzData.clear();
              for ( uint i = 0; i < nonZeros; i++)
              {
                _is >> origValue;
                _is >> origIndex;
                _is >> transformedValue;

                std::pair<T, dataelement > p ( origValue, dataelement ( origIndex, transformedValue ) );
                elementpointer it = this->nzData.insert ( p);
                this->nonzero_indices.insert ( std::pair<uint, elementpointer> ( origIndex, it ) );
              }

              _is >> tmp; // end of block
              tmp = this->removeEndTag ( tmp );
            }


            _is >> tmp; // end of block
            tmp = this->removeEndTag ( tmp );
          }
          else
          {
            std::cerr << "WARNING -- unexpected SortedVectorSparse object -- " << tmp << " -- for restoration... aborting" << std::endl;
            throw;
          }
        }

      }
      else
      {
        std::cerr << "SortedVectorSparse::restore -- InStream not initialized - restoring not possible!" << std::endl;
        throw;
      }
    };

    virtual void store ( std::ostream & _os,
                         int _format = 0
                       ) const
    {
      if ( _os.good() )
      {
        // show starting point
        _os << this->createStartTag( "SortedVectorSparse" ) << std::endl;

        _os.precision (std::numeric_limits<double>::digits10 + 1);

        _os << this->createStartTag( "tolerance" ) << std::endl;
        _os << tolerance << std::endl;
        _os << this->createEndTag( "tolerance" ) << std::endl;

        _os << this->createStartTag( "ui_n" ) << std::endl;
        _os << this->ui_n << std::endl;
        _os << this->createEndTag( "ui_n" ) << std::endl;


        _os << this->createStartTag( "underlying_data_(sorted)" ) << std::endl;
        _os << this->createStartTag( "nonZeros" ) << std::endl;
        _os << this->getNonZeros() << std::endl;
        _os << this->createEndTag( "nonZeros" ) << std::endl;

        _os << this->createStartTag( "data" ) << std::endl;
        for (const_elementpointer elP = this->nzData.begin();  elP != this->nzData.end(); elP++)
        {
          _os << elP->first << " " << elP->second.first << " " << elP->second.second << " ";
        }
        _os << std::endl;
        _os << this->createEndTag( "data" ) << std::endl;
        _os << this->createEndTag( "underlying_data_(sorted)" ) << std::endl;

        // done
        _os << this->createEndTag( "SortedVectorSparse" ) << std::endl;
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
