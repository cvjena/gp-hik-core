/**
* @file FeatureMatrixT.tcc
* @brief A feature matrix, storing (sparse) features sorted per dimension (Implementation)
* @author Alexander Freytag
* @date 07-12-2011 (dd-mm-yyyy)
*/
// #ifndef FEATUREMATRIX_TCC
// #define FEATUREMATRIX_TCC

// gp-hik-core includes
#include "FeatureMatrixT.h"

namespace NICE {



    //------------------------------------------------------
    // several constructors and destructors
    //------------------------------------------------------

    // Default constructor
    template <typename T>
    FeatureMatrixT<T>::FeatureMatrixT()
    {
      this->ui_n = 0;
      this->ui_d = 0;
      this->features.clear();
      this->b_verbose = false;
      this->b_debug = false;
    }


    // Recommended constructor
    template <typename T>
    FeatureMatrixT<T>::FeatureMatrixT(const std::vector<std::vector<T> > & _features,
                                      const uint & _dim
                                     )
    {
      this->ui_n = 0;
      if (_dim < 0)
        this->ui_d = (*_features.begin()).size();
      else
        this->ui_d = _dim;

      for (typename std::vector<std::vector<T> >::const_iterator it = _features.begin(); it != _features.end(); it++)
      {
        add_feature(*it);
      }
      this->b_verbose = false;
      this->b_debug = false;
    }

    //Constructor reading data from a vector of sparse vector pointers
    template <typename T>
    FeatureMatrixT<T>::
    FeatureMatrixT(const std::vector< const NICE::SparseVector * > & _X,
                   const bool _dimensionsOverExamples,
                   const uint & _dim
                  )
    {
      this->features.clear();

      // resize our data structure
      set_d( _dim );

      // set number of examples n
      if ( this->ui_d > 0 )
      {
        if (_dimensionsOverExamples) //do we have dim x examples ?
          this->ui_n = _X[0]->getDim(); //NOTE Pay attention: we assume, that this number is set!
        else //we have examples x dimes (as usually done)
          this->ui_n = _X.size();
      }


      // insert all values
      if (_dimensionsOverExamples) //do we have dim x examples ?
      {
        for (uint dim = 0; dim < this->ui_d; dim++)
        {
          this->features[dim].insert( _X[dim] );
        }
      }
      else //we have examples x dimes (as usually done)
      {
        //loop over every example to add its content
        for (uint nr = 0; nr < this->ui_n; nr++)
        {
          //loop over every dimension to add the specific value to the corresponding SortedVectorSparse
          for (NICE::SparseVector::const_iterator elemIt = _X[nr]->begin(); elemIt != _X[nr]->end(); elemIt++)
          {
            //elemIt->first: dim, elemIt->second: value
            this->features[elemIt->first].insert( (T) elemIt->second, nr);
          }//for non-zero-values of the feature
        }//for every new feature
      }//if dimOverEx

      //set n for the internal data structure SortedVectorSparse
      for (typename std::vector<NICE::SortedVectorSparse<T> >::iterator it = this->features.begin(); it != this->features.end(); it++)
        (*it).setN( this->ui_n );
    }

#ifdef NICE_USELIB_MATIO
    //Constructor reading data from matlab-files
    template <typename T>
    FeatureMatrixT<T>::
    FeatureMatrixT(const sparse_t & _features,
                   const uint & _dim
                  )
    {
      if (_dim < 0)
        set_d( _features.njc -1 );
      else
        set_d( _dim );

      uint nMax(0);

      for ( uint i = 0; i < _features.njc-1; i++ ) //walk over dimensions
      {
        for ( uint j = _features.jc[i]; j < _features.jc[i+1] && j < _features.ndata; j++ ) //walk over single features, which are sparsely represented
        {
          this->features[i].insert(((T*)_features.data)[j], _features.ir[ j]);
          if ((_features.ir[ j])>nMax)
            nMax = _features.ir[ j];
        }
      }
      for (typename std::vector<NICE::SortedVectorSparse<T> >::iterator it = this->features.begin(); it != this->features.end(); it++)
      {
        (*it).setN(nMax+1);
      }
      this->ui_n = nMax+1;
      this->b_verbose = false;
    }

    //Constructor reading data from matlab-files
    template <typename T>
    FeatureMatrixT<T>::
    FeatureMatrixT(const sparse_t & _features,
                   const std::map<uint, uint> & _examples,
                   const uint & _dim)
    {
      if (_dim < 0)
        set_d(_features.njc -1);
      else
        set_d(_dim);

      uint nMax(0);

      for ( uint i = 0; i < _features.njc-1; i++ ) //walk over dimensions
      {
        for ( uint j = _features.jc[i]; j < _features.jc[i+1] && j < _features.ndata; j++ ) //walk over single features, which are sparsely represented
        {
          uint example_index = _features.ir[ j];
          std::map<uint, uint>::const_iterator it = examples.find(example_index);
          if ( it != examples.end() ) {
            this->features[i].insert(((T*)_features.data)[j], it->second /* new index */);
            if (it->second > nMax)
              nMax = it->second;
          }
        }
      }
      for (typename std::vector<NICE::SortedVectorSparse<T> >::iterator it = this->features.begin(); it != this->features.end(); it++)
        (*it).setN(nMax+1);

      this->ui_n = nMax+1;
      this->b_verbose = false;
    }
#endif

    // Default destructor
    template <typename T>
    FeatureMatrixT<T>::~FeatureMatrixT()
    {
    }

    //------------------------------------------------------
    // several get and set methods including access operators
    //------------------------------------------------------

    // Get number of examples
    template <typename T>
    uint FeatureMatrixT<T>::get_n() const
    {
      return this->ui_n;
    }

    //  Get number of dimensions
    template <typename T>
    uint FeatureMatrixT<T>::get_d() const
    {
      return this->ui_d;
    }

    //  Sets the given dimension and re-sizes internal data structure. WARNING: this will completely remove your current data!
    template <typename T>
    void FeatureMatrixT<T>::set_d(const uint & _d)
    {
      this->ui_d = _d;
      this->features.resize( this->ui_d );
    }

    template <typename T>
    void FeatureMatrixT<T>::setVerbose( const bool & _verbose)
    {
      this->b_verbose = _verbose;
    }

    template <typename T>
    bool FeatureMatrixT<T>::getVerbose( )   const
    {
      return this->b_verbose;
    }

    template <typename T>
    void FeatureMatrixT<T>::setDebug( const bool & _debug)
    {
      this->b_debug = _debug;
    }

    template <typename T>
    bool FeatureMatrixT<T>::getDebug( )   const
    {
      return this->b_debug;
    }

    //  Matrix-like operator for element access, performs validity check
    template <typename T>
    inline T FeatureMatrixT<T>::operator()(const uint _row,
                                           const uint _col
                                          ) const
    {
      if ( (_row < 0) || (_col < 0) || (_row > this->ui_d) || (_col > this->ui_n) )
      {
        fthrow(Exception, "FeatureMatrixT: out of bounds");
      }
      else
        return ( this->features[_row]).access(_col);
    }

    template<class T>
    inline bool
    FeatureMatrixT<T>::operator==(const FeatureMatrixT<T> & _F) const
    {
      if ( ( this->get_n() != _F.get_n()) || (this->get_d() != _F.get_d()) )
      {
        fthrow(Exception, "FeatureMatrixT<T>::operator== : (n != F.get_n()) || (d != F.get_d()) -- number of dimensions does not fit");
      }
      else if ((this->ui_n == 0) || (this->ui_d == 0))
      {
        return true;
      }

      for (uint i = 0; i < this->ui_d; i++)
      {
        for (uint j = 0; j < this->ui_n; j++)
        {
          // FIXME: it would be more efficient if we compare SortedVectorSparse objects here
          if(!((*this)(i,j) == _F(i,j)))
            return false;
        }
      }
      return true;
    }

    template<class T>
    inline bool
    FeatureMatrixT<T>::operator!=(const FeatureMatrixT<T> & _F) const
    {
      if ( ( (*this).get_n() != _F.get_n()) ||
           ( (*this).get_d() != _F.get_d())
         )
      {
        fthrow(Exception, "FeatureMatrixT::operator!=(): (n != F.get_n()) || (d != F.get_d()) -- number of dimensions does not fit");
      }
      else if ((this->ui_n == 0) || (this->ui_d == 0))
      {
        return false;
      }

      for (uint i = 0; i < this->ui_d; i++)
      {
        for (uint j = 0; j < this->ui_n; j++)
        {
          if(!((*this)(i,j) == _F(i,j)))
            return true;
        }
      }
      return false;
    }

    template<typename T>
    inline FeatureMatrixT<T>&
    FeatureMatrixT<T>::operator=(const FeatureMatrixT<T> & _F)
    {
      this->set_d(_F.get_d());

      this->ui_n = _F.get_n();

      for (uint i = 0; i < (*this).get_d(); i++)
      {
        // use the operator= of SortedVectorSparse
        features[i] = _F[i];
      }

      return *this;
    }

    //  Element access without validity check
    template <typename T>
    inline T FeatureMatrixT<T>::getUnsafe(const uint _row,
                                          const uint _col
                                         ) const
    {
      return (this->features[_row]).access(_col);
    }

    //! Element access of original values without validity check
    template <typename T>
    inline T FeatureMatrixT<T>::getOriginal(const uint _row,
                                            const uint _col
                                           ) const
    {
      return (this->features[_row]).accessOriginal(_col);
    }

    //  Sets a specified element to the given value, performs validity check
    template <typename T>
    inline void FeatureMatrixT<T>::set (const uint _row,
                                        const uint _col,
                                        const T & _newElement,
                                        bool _setTransformedValue
                                       )
    {
      if ( (_row < 0) || (_col < 0) || (_row > this->ui_d) || (_col > this->ui_n) )
      {
        return;
      }
      else
        (this->features[_row]).set ( _col, _newElement, _setTransformedValue );
    }

    //  Sets a specified element to the given value, without validity check
    template <typename T>
    inline void FeatureMatrixT<T>::setUnsafe (const uint _row,
                                              const uint _col,
                                              const T & _newElement,
                                              bool _setTransformedValue
                                             )
    {
      (this->features[_row]).set ( _col, _newElement, _setTransformedValue );
    }

    //  Acceess to all element entries of a specified dimension, including validity check
    template <typename T>
    void FeatureMatrixT<T>::getDimension(const uint & _dim,
                                         NICE::SortedVectorSparse<T> & _dimension
                                        ) const
    {
      if ( (_dim < 0) || (_dim > this->ui_d) )
      {
        return;
      }
      else
        _dimension = this->features[_dim];
    }

    //  Acceess to all element entries of a specified dimension, without validity check
    template <typename T>
    void FeatureMatrixT<T>::getDimensionUnsafe(const uint & _dim,
                                               NICE::SortedVectorSparse<T> & _dimension
                                              ) const
    {
      _dimension = this->features[_dim];
    }

    // Finds the first element in a given dimension, which equals elem
    template <typename T>
    void FeatureMatrixT<T>::findFirstInDimension(const uint & _dim,
                                                 const T & _elem,
                                                 uint & _position
                                                ) const
    {
      _position = 0;
      if ( _dim > this->ui_d )
        return;

      std::pair< typename SortedVectorSparse<T>::elementpointer, typename SortedVectorSparse<T>::elementpointer > eit;
      eit =  this->features[_dim].nonzeroElements().equal_range ( _elem );

      _position = distance( this->features[_dim].nonzeroElements().begin(), eit.first );

      if ( _elem > this->features[_dim].getTolerance() )
        _position += this->features[_dim].getZeros();

    }

    //  Finds the last element in a given dimension, which equals elem
    template <typename T>
    void FeatureMatrixT<T>::findLastInDimension(const uint & _dim,
                                                const T & _elem,
                                                uint & _position
                                               ) const
    {
      _position = 0;
      if ( _dim > this->ui_d )
        return;

      std::pair< typename SortedVectorSparse<T>::const_elementpointer, typename SortedVectorSparse<T>::const_elementpointer > eit =  this->features[_dim].nonzeroElements().equal_range ( _elem );

      _position = distance( this->features[_dim].nonzeroElements().begin(), eit.second );

      if ( _elem > this->features[_dim].getTolerance() )
        _position += this->features[_dim].getZeros();
    }

    //  Finds the first element in a given dimension, which is larger as elem
    template <typename T>
    void FeatureMatrixT<T>::findFirstLargerInDimension(const uint & _dim,
                                                       const T & _elem,
                                                       uint & _position
                                                      ) const
    {
      _position = 0;
      if ( _dim > this->ui_d )
        return;

      //no non-zero elements?
      if (this->features[_dim].getNonZeros() <= 0)
      {
        // if element is greater than zero, than is should be added at the last position
        if (_elem > this->features[_dim].getTolerance() )
          _position = this->ui_n;

        //if not, position is 0
        return;
      }

      if (this->features[_dim].getNonZeros() == 1)
      {
        // if element is greater than the only nonzero element, then it is larger than everything else
        if (this->features[_dim].nonzeroElements().begin()->first <= _elem)
          _position = this->ui_n;

        //if not, but the element is still greater than zero, than
        else if (_elem > this->features[_dim].getTolerance() )
          _position = this->ui_n -1;

        return;
      }


      // standard case - not everything is zero and not only a single element is zero

      // find pointer to last non-zero element
      // FIXME no idea why this should be necessary...
      typename SortedVectorSparse<T>::const_elementpointer it =  this->features[_dim].nonzeroElements().end(); //this is needed !!!
      // find pointer to first element largern than the given value
      it = this->features[_dim].nonzeroElements().upper_bound ( _elem ); //if all values are smaller, this does not do anything at all

      _position = distance( this->features[_dim].nonzeroElements().begin(), it );


      if ( _elem > this->features[_dim].getTolerance() )
      {
        //position += features[dim].getZeros();
        _position += this->ui_n - this->features[_dim].getNonZeros();
      }
    }

    //  Finds the last element in a given dimension, which is smaller as elem
    template <typename T>
    void FeatureMatrixT<T>::findLastSmallerInDimension(const uint & _dim,
                                                       const T & _elem,
                                                       uint & _position
                                                      ) const
    {
      _position = 0;
      if ( (_dim < 0) || (_dim > this->ui_d))
        return;

      typename SortedVectorSparse<T>::const_elementpointer it =  this->features[_dim].nonzeroElements().lower_bound ( _elem );
      _position = distance( this->features[_dim].nonzeroElements().begin(), it );

      if ( it->first > this->features[_dim].getTolerance() )
        _position += this->features[_dim].getZeros();
    }
    
    template <typename T>
    T FeatureMatrixT<T>::getLargestValue ( const bool & _getTransformedValue ) const
    {
      T vmax = (T) 0; 
      T vtmp = (T) 0;
      
      uint tmp ( 0 );
      for ( typename std::vector<NICE::SortedVectorSparse<T> >::const_iterator it = this->features.begin();
            it != this->features.end();
            it++, tmp++
      )
      {
        vtmp = it->getLargestValueUnsafe( 1.0 /*quantile, we are interested in the largest value*/, _getTransformedValue );
        if ( vtmp > vmax )
        {
          vmax = vtmp;
        }
      }
      return vmax;
    }
    
    template <typename T>
    NICE::VectorT<T> FeatureMatrixT<T>::getLargestValuePerDimension ( const double & _quantile, 
                                                                      const bool & _getTransformedValue
                                                                    ) const     
    {
      NICE::VectorT<T> vmax ( this->get_d() );
      
      uint tmp ( 0 );      
      typename NICE::VectorT<T>::iterator vmaxIt = vmax.begin();
      for ( typename std::vector<NICE::SortedVectorSparse<T> >::const_iterator it = this->features.begin();
            it != this->features.end();
            it++, vmaxIt++, tmp++
      )
      {       
        *vmaxIt = it->getLargestValueUnsafe( _quantile, _getTransformedValue );
      }    
      return vmax;
    }
    
    //------------------------------------------------------
    // high level methods
    //------------------------------------------------------

    template <typename T>
    void FeatureMatrixT<T>::applyFunctionToFeatureMatrix ( const NICE::ParameterizedFunction *_pf )
    {
      if (_pf != NULL)
      {
        // REMARK: might be inefficient due to virtual calls
        if ( !_pf->isOrderPreserving() )
          fthrow(Exception, "ParameterizedFunction::applyFunctionToFeatureMatrix: this function is optimized for order preserving transformations");

        uint d = this->get_d();
        for (uint dim = 0; dim < d; dim++)
        {
          std::multimap< double, typename SortedVectorSparse<double>::dataelement> & nonzeroElements = this->getFeatureValues(dim).nonzeroElements();
          for ( SortedVectorSparse<double>::elementpointer i = nonzeroElements.begin(); i != nonzeroElements.end(); i++ )
          {
            SortedVectorSparse<double>::dataelement & de = i->second;

            //TODO check, wether the element is "sparse" afterwards
            de.second = _pf->f( dim, i->first );
          }
        }

        /*for ( int i = 0 ; i < featureMatrix.get_n(); i++ )
          for ( int index = 0 ; index < featureMatrix.get_d(); index++ )
            featureMatrix.set(index, i, f( (uint)index, featureMatrix.getOriginal(index,i) ), isOrderPreserving() );*/
      }
      else
      {
        //no pf given -> nothing to do
      }
    }


    //Computes the ratio of sparsity across the matrix
    template <typename T>
    double FeatureMatrixT<T>:: computeSparsityRatio() const
    {
      double ratio(0.0);
      for (typename std::vector<NICE::SortedVectorSparse<T> >::const_iterator it = this->features.begin(); it != this->features.end(); it++)
      {
        ratio += (*it).getZeros() / (double) (*it).getN();
      }
      if (this->features.size() != 0)
        ratio /= double(this->features.size());
      return ratio;
    }

    //  add a new feature and insert its elements at the end of each dimension vector
    template <typename T>
    void FeatureMatrixT<T>::add_feature( const std::vector<T> & _feature,
                                         const NICE::ParameterizedFunction *_pf
                                       )
    {
      if (this->ui_n == 0)
      {
        this->set_d( _feature.size() );
      }

      if ( _feature.size() != this->ui_d)
      {
        fthrow(Exception, "FeatureMatrixT<T>::add_feature - number of dimensions does not fit");
        return;
      }

      for (uint dimension = 0; dimension <  this->features.size(); dimension++)
      {
        if (_pf != NULL)
          this->features[dimension].insert( _feature[dimension], _pf->f( dimension, _feature[dimension]) );
        else
          this->features[dimension].insert( _feature[dimension] );
      }
      this->ui_n++;
    }
    //  add a new feature and insert its elements at the end of each dimension vector
    template <typename T>
    void FeatureMatrixT<T>::add_feature(const NICE::SparseVector & _feature,
                                        const ParameterizedFunction *_pf
                                       )
    {
      if (this->ui_n == 0)
      {
        this->set_d( _feature.size() );
      }

      if ( _feature.getDim() > this->ui_d)
      {
        fthrow(Exception, "FeatureMatrixT<T>::add_feature - number of dimensions does not fit");
        return;
      }

      for (NICE::SparseVector::const_iterator it = _feature.begin(); it != _feature.end(); it++)
      {
        if (_pf != NULL)
          this->features[it->first].insert( (T) it->second, _pf->f( it->first, (T) it->second), true /* _specifyFeatureNumber */, this->ui_n );
        else
          this->features[it->first].insert( (T) it->second, true /* _specifyFeatureNumber */, this->ui_n );
      }
      this->ui_n++;
    }

    //  add several new features and insert their elements in the already ordered structure
    template <typename T>
    void FeatureMatrixT<T>::add_features(const std::vector<std::vector<T> > & _features )
    {
      //TODO do we need the parameterized function here as well? usually, we add several features and run applyFunctionToFeatureMatrix afterwards.
      // check this please :)

      //TODO assure that every feature has the same dimension
      if (this->ui_n == 0)
      {
        this->set_d(_features.size());
      }

      //pay attention: we assume now, that we have a vector (over dimensions) containing vectors over features (examples per dimension) - to be more efficient
      for (uint dim = 0; dim < this->ui_d; dim++)
      {
          this->features[dim].insert( _features[dim] );
      }

      //update the number of our features
      this->ui_n += _features[0].size();
    }

    template <typename T>
    void FeatureMatrixT<T>::set_features(const std::vector<std::vector<T> > & _features,
                                         std::vector<std::vector<uint> > & _permutations,
                                         const uint & _dim
                                        )
    {
      this->features.clear();
      this->set_d( std::max ( _dim, (const uint) _features.size() ) );

      if ( this->ui_d > 0 )
        this->ui_n = _features[0].size();

      //pay attention: we assume now, that we have a vector (over dimensions) containing vectors over features (examples per dimension) - to be more efficient
      for (uint dim = 0; dim < this->ui_d; dim++)
      {
        this->features[dim].insert( _features[dim] );
      }

      this->getPermutations( _permutations );
    }

    template <typename T>
    void FeatureMatrixT<T>::set_features(const std::vector<std::vector<T> > & _features,
                                         std::vector<std::map<uint,uint> > & _permutations,
                                         const uint & _dim
                                        )
    {
      this->features.clear();
      this->set_d( std::max ( _dim, _features.size() ) );

      if ( this->ui_d > 0 )
        this->ui_n = _features[0].size();

      //pay attention: we assume now, that we have a vector (over dimensions) containing vectors over features (examples per dimension) - to be more efficient
      for (uint dim = 0; dim < this->ui_d; dim++)
      {
        this->features[dim].insert( _features[dim] );
      }

      this->getPermutations( _permutations );
    }

    template <typename T>
    void FeatureMatrixT<T>::set_features(const std::vector<std::vector<T> > & _features,
                                         const uint & _dim
                                        )
    {
      this->features.clear();
      this->set_d( std::max ( _dim, (const uint) _features.size() ) );

      if ( this->ui_d > 0 )
        this->ui_n = _features[0].size();

      //pay attention: we assume now, that we have a vector (over dimensions) containing vectors over features (examples per dimension) - to be more efficient
      for (uint dim = 0; dim < this->ui_d; dim++)
      {
        if ( this->b_debug )
        {
          std::cerr << " dim: " << dim << " add " << _features[dim].size() << " examples " << std::endl;
        }

        this->features[dim].insert( _features[dim] );
      }
    }

    template <typename T>
    void FeatureMatrixT<T>::set_features(const std::vector< const NICE::SparseVector * > & _features,
                                         const bool _dimensionsOverExamples,
                                         const uint & _dim
                                        )
    {
      this->features.clear();
      if (_features.size() == 0)
      {
        std::cerr << "set_features without features" << std::endl;
      }


      // resize our data structure
      //therefore, let's first of all figure out if the user specified a dimension or not.
      uint dimTmp ( _dim );

      if (_dimensionsOverExamples) //do we have dim x examples ?
      {
        if ( _features.size() > dimTmp )
        {
          dimTmp = _features.size();
        }
      }
      else //we have examples x dimes (as usually done)
      {
        if (_features.size() > 0) //and have at least one example
        {
          try{
            if ( _features[0]->getDim() > dimTmp )
            {
              dimTmp = _features[0]->getDim();
            }
          }
          catch(...)
          {
            std::cerr << "FeatureMatrixT<T>::set_features -- something went wrong using getDim() of SparseVectors" << std::endl;
          }
        }
      }
      this->set_d( dimTmp );

      // set number of examples n
      if ( this->ui_d > 0 )
      {
        if ( _dimensionsOverExamples ) //do we have dim x examples ?
          this->ui_n = _features[0]->getDim(); //NOTE Pay attention: we assume, that this number is set!
        else //we have examples x dimes (as usually done)
          this->ui_n = _features.size();
      }

      // insert all values
      if ( _dimensionsOverExamples ) //do we have dim x examples ?
      {
        for (uint dim = 0; dim < this->ui_d; dim++)
        {
          this->features[dim].insert( _features[dim] );
        }
      }
      else //we have examples x dimes (as usually done)
      {
        if ( this->b_debug )
          std::cerr << "FeatureMatrixT<T>::set_features " << this->ui_n << " new examples" << std::endl;
        //loop over every example to add its content
        for (uint nr = 0; nr < this->ui_n; nr++)
        {
          if ( this->b_debug )
            std::cerr << "add feature nr. " << nr << " / " << _features.size() << " ";
          //loop over every dimension to add the specific value to the corresponding SortedVectorSparse
          for (NICE::SparseVector::const_iterator elemIt = _features[nr]->begin(); elemIt != _features[nr]->end(); elemIt++)
          {
            if ( this->b_debug )
              std::cerr << elemIt->first << "-" << elemIt->second << " ";
            //elemIt->first: dim, elemIt->second: value
            this->features[elemIt->first].insert( (T) elemIt->second, true /* _specifyFeatureNumber */, nr);
          }//for non-zero-values of the feature
          if ( this->b_debug )
            std::cerr << std::endl;
        }//for every new feature
        if ( this->b_debug )
          std::cerr << "FeatureMatrixT<T>::set_features done" << std::endl;
      }//if dimOverEx

      //set n for the internal data structure SortedVectorSparse
      for (typename std::vector<NICE::SortedVectorSparse<T> >::iterator it = this->features.begin(); it != this->features.end(); it++)
        (*it).setN( this->ui_n );
    }

    template <typename T>
    void FeatureMatrixT<T>::getPermutations( std::vector<std::vector<uint> > & _permutations) const
    {
      for (uint dim = 0; dim < this->ui_d; dim++)
      {
        std::vector<uint> perm (  (this->features[dim]).getPermutation() );
        _permutations.push_back(perm);
      }
    }

    template <typename T>
    void FeatureMatrixT<T>::getPermutations( std::vector<std::map<uint,uint> > & _permutations) const
    {
      for (uint dim = 0; dim < this->ui_d; dim++)
      {
        std::map<uint,uint> perm (  (this->features[dim]).getPermutationNonZeroReal() );
        _permutations.push_back(perm);
      }
    }


    //  Prints the whole Matrix (outer loop over dimension, inner loop over features)
    template <typename T>
    void FeatureMatrixT<T>::print(std::ostream & _os) const
    {
      if (_os.good())
      {
        for (uint dim = 0; dim < this->ui_d; dim++)
        {
          this->features[dim].print(_os);
        }
      }
    }

    // Computes the whole non-sparse matrix. WARNING: this may result in a really memory-consuming data-structure!
    template <typename T>
    void FeatureMatrixT<T>::computeNonSparseMatrix(NICE::MatrixT<T> & _matrix,
                                                   bool _transpose
                                                  ) const
    {
      if ( _transpose )
        _matrix.resize(this->get_n(),this->get_d());
      else
        _matrix.resize(this->get_d(),this->get_n());

      _matrix.set((T)0.0);
      uint dimIdx(0);
      for (typename std::vector<NICE::SortedVectorSparse<T> >::const_iterator it = this->features.begin(); it != this->features.end(); it++, dimIdx++)
      {
        std::map< uint, typename NICE::SortedVectorSparse<T>::elementpointer>  nonzeroIndices= (*it).nonzeroIndices();
        for (typename std::map< uint, typename NICE::SortedVectorSparse<T>::elementpointer>::const_iterator inIt = nonzeroIndices.begin(); inIt != nonzeroIndices.end(); inIt++)
        {
          uint featIndex = ((*inIt).second)->second.first;
          if ( _transpose )
            _matrix(featIndex,dimIdx) =((*inIt).second)->second.second;
          else
            _matrix(dimIdx,featIndex) =((*inIt).second)->second.second;
        }
      }
    }

    // Computes the whole non-sparse matrix. WARNING: this may result in a really memory-consuming data-structure!
    template <typename T>
    void FeatureMatrixT<T>::computeNonSparseMatrix(std::vector<std::vector<T> > & _matrix,
                                                   bool _transpose
                                                  ) const
    {
      if ( _transpose )
        _matrix.resize(this->get_n());
      else
        _matrix.resize(this->get_d());

      // resizing the matrix
      for ( uint i = 0 ; i < _matrix.size(); i++ )
        if ( _transpose )
          _matrix[i] = std::vector<T>(this->get_d(), 0.0);
        else
          _matrix[i] = std::vector<T>(this->get_n(), 0.0);

      uint dimIdx(0);
      for (typename std::vector<NICE::SortedVectorSparse<T> >::const_iterator it = this->features.begin(); it != this->features.end(); it++, dimIdx++)
      {
        std::map< uint, typename NICE::SortedVectorSparse<T>::elementpointer>  nonzeroIndices= (*it).nonzeroIndices();
        for (typename std::map< uint, typename NICE::SortedVectorSparse<T>::elementpointer>::const_iterator inIt = nonzeroIndices.begin(); inIt != nonzeroIndices.end(); inIt++)
        {
          uint featIndex = ((*inIt).second)->second.first;
          if ( _transpose )
            _matrix[featIndex][dimIdx] =((*inIt).second)->second.second;
          else
            _matrix[dimIdx][featIndex] =((*inIt).second)->second.second;
        }
      }
    }

    // Swaps to specified elements, performing a validity check
    template <typename T>
    void FeatureMatrixT<T>::swap(const uint & _row1,
                                 const uint & _col1,
                                 const uint & _row2,
                                 const uint & _col2
                                )
    {
      if ( (_row1 < 0) || (_col1 < 0) || (_row1 > this->ui_d) || (_col1 > this->ui_n) ||
           (_row2 < 0) || (_col2 < 0) || (_row2 > this->ui_d) || (_col2 > this->ui_n)
         )
      {
        return;
      }
      else
      {
        //swap
        T tmp = (*this)(_row1, _col1);
        (*this).set(_row1, _col1, (*this)(_row2,_col2));
        (*this).set(_row2, _col2, tmp);
      }
    }

    //  Swaps to specified elements, without performing a validity check
    template <typename T>
    void FeatureMatrixT<T>::swapUnsafe(const uint & _row1,
                                       const uint & _col1,
                                       const uint & _row2,
                                       const uint & _col2
                                      )
    {
      //swap
      T tmp = (*this)(_row1, _col1);
      (*this).set(_row1, _col1, (*this)(_row2,_col2));
      (*this).set(_row2, _col2, tmp);
    }

    template <typename T>
    void FeatureMatrixT<T>::hikDiagonalElements( Vector & _diagonalElements ) const
    {
      uint dimIdx = 0;
      // the function calculates the diagonal elements of a HIK kernel matrix
      _diagonalElements.resize(this->ui_n);
      _diagonalElements.set(0.0);
      // loop through all dimensions
      for (typename std::vector<NICE::SortedVectorSparse<T> >::const_iterator it = this->features.begin(); it != this->features.end(); it++, dimIdx++)
      {
        std::map< uint, typename NICE::SortedVectorSparse<T>::elementpointer>  nonzeroIndices= (*it).nonzeroIndices();
        // loop through all features
        for (typename std::map< uint, typename NICE::SortedVectorSparse<T>::elementpointer>::const_iterator inIt = nonzeroIndices.begin(); inIt != nonzeroIndices.end(); inIt++)
        {
          uint index = inIt->first;
          typename NICE::SortedVectorSparse<T>::elementpointer p = inIt->second;
          typename NICE::SortedVectorSparse<T>::dataelement de = p->second;

          _diagonalElements[index] += de.second;
        }
      }
    }

    template <typename T>
    double FeatureMatrixT<T>::hikTrace() const
    {
      Vector diagonalElements;
      this->hikDiagonalElements( diagonalElements );
      return diagonalElements.Sum();

    }

    template <typename T>
    uint FeatureMatrixT<T>::getNumberOfNonZeroElementsPerDimension(const uint & dim) const
    {
      //FIXME we could return a boolean indicating success and return the actual number via call-by-reference
      if ( (dim < 0) || (dim > this->ui_d))
        return 0;
      return this->features[dim].getNonZeros();
    }

    template <typename T>
    uint FeatureMatrixT<T>::getNumberOfZeroElementsPerDimension(const uint & dim) const
    {
      if ( (dim < 0) || (dim > this->ui_d))
        return 0;
      return this->ui_n - this-> features[dim].getNonZeros();
    }


    template <typename T>
    void FeatureMatrixT<T>::restore ( std::istream & _is,
                                      int _format
                                    )
    {
      bool b_restoreVerbose ( false );
      if ( _is.good() )
      {
        if ( b_restoreVerbose )
          std::cerr << " restore FeatureMatrixT" << std::endl;

        std::string tmp;
        _is >> tmp; //class name

        if ( ! this->isStartTag( tmp, "FeatureMatrixT" ) )
        {
            std::cerr << " WARNING - attempt to restore FeatureMatrixT, but start flag " << tmp << " does not match! Aborting... " << std::endl;
            throw;
        }

        _is.precision ( std::numeric_limits<double>::digits10 + 1);

        bool b_endOfBlock ( false ) ;

        while ( !b_endOfBlock )
        {
          _is >> tmp; // start of block

          if ( this->isEndTag( tmp, "FeatureMatrixT" ) )
          {
            b_endOfBlock = true;
            continue;
          }

          tmp = this->removeStartTag ( tmp );

          if ( b_restoreVerbose )
            std::cerr << " currently restore section " << tmp << " in FeatureMatrixT" << std::endl;

          if ( tmp.compare("ui_n") == 0 )
          {
            _is >> this->ui_n;
            _is >> tmp; // end of block
            tmp = this->removeEndTag ( tmp );
          }
          else if ( tmp.compare("ui_d") == 0 )
          {
            _is >> this->ui_d;
            _is >> tmp; // end of block
            tmp = this->removeEndTag ( tmp );
          }
          else if ( tmp.compare("features") == 0 )
          {
            //NOTE assumes d to be read first!
            this->features.resize( this->ui_d);
            //now read features for every dimension
            for (uint dim = 0; dim < this->ui_d; dim++)
            {
              NICE::SortedVectorSparse<T> svs;
              this->features[dim] = svs;
              this->features[dim].restore(_is, _format);
            }

            _is >> tmp; // end of block
            tmp = this->removeEndTag ( tmp );
          }
          else
          {
            std::cerr << "WARNING -- unexpected FeatureMatrixT object -- " << tmp << " -- for restoration... aborting" << std::endl;
            throw;
          }
        }

      }
      else
      {
        std::cerr << "FeatureMatrixT<T>::restore -- InStream not initialized - restoring not possible!" << std::endl;
        throw;
      }
    }

    template <typename T>
    void FeatureMatrixT<T>::store ( std::ostream & _os,
                                    int _format
                                  ) const
    {
      if (_os.good())
      {
        // show starting point
        _os << this->createStartTag( "FeatureMatrixT" ) << std::endl;

        _os.precision (std::numeric_limits<double>::digits10 + 1);

        _os << this->createStartTag( "ui_n" ) << std::endl;
        _os << this->ui_n << std::endl;
        _os << this->createEndTag( "ui_n" ) << std::endl;


        _os << this->createStartTag( "ui_d" ) << std::endl;
        _os << this->ui_d << std::endl;
        _os << this->createEndTag( "ui_d" ) << std::endl;

              //now write features for every dimension
        _os << this->createStartTag( "features" ) << std::endl;
        for (uint dim = 0; dim < this->ui_d; dim++)
        {
          this->features[dim].store(_os,_format);
        }
        _os << this->createEndTag( "features" ) << std::endl;

        // done
        _os << this->createEndTag( "FeatureMatrixT" ) << std::endl;
      }
      else
      {
        std::cerr << "FeatureMatrixT<T>::store -- OutStream not initialized - storing not possible!" << std::endl;
      }
    }

    template <typename T>
    void FeatureMatrixT<T>::clear ()
    {}

} // namespace

// #endif
