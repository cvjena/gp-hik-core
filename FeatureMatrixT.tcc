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
      n = 0;
      d = 0;
      features.clear();
      verbose = false;
      debug = false;
    }
    

    // Recommended constructor
    template <typename T>
    FeatureMatrixT<T>::FeatureMatrixT(const std::vector<std::vector<T> > & _features, const int & _dim)
    {
      n = 0;
      if (_dim < 0)
        d = (*_features.begin()).size();
      else
        d = _dim;
      
      for (typename std::vector<std::vector<T> >::const_iterator it = _features.begin(); it != _features.end(); it++)
      {
        add_feature(*it);
      }
      verbose = false;
      debug = false;
    }

    //Constructor reading data from a vector of sparse vector pointers
    template <typename T>
    FeatureMatrixT<T>::
    FeatureMatrixT(const std::vector< const NICE::SparseVector * > & X, const bool dimensionsOverExamples, const int & _dim)
    {
      features.clear();
      
      // resize our data structure
      if (_dim >= 0) //did the user specified the number of dimensions?
        set_d(_dim);
      else //dimensions not specified by users
      {
        if (dimensionsOverExamples) //do we have dim x examples ?
        {
          set_d(X.size());
        }
        else //we have examples x dimes (as usually done)
        {
          if (X.size() > 0) //and have at least one example
            set_d(X[0]->getDim());  
          else //no example, so set the dim to 0, since we have no idea at all
          {
            set_d(0);
          }          
        }
      }
           
      // set number of examples n
      if (d>0)
      {
        if (dimensionsOverExamples) //do we have dim x examples ?
          n = X[0]->getDim(); //NOTE Pay attention: we assume, that this number is set!
        else //we have examples x dimes (as usually done)   
          n = X.size(); 
      }  

     
      // insert all values
      if (dimensionsOverExamples) //do we have dim x examples ?
      {
        for (int dim = 0; dim < d; dim++)
        {
          features[dim].insert( X[dim] );
        }
      }
      else //we have examples x dimes (as usually done)
      {
        //loop over every example to add its content
        for (int nr = 0; nr < n; nr++)
        {
          //loop over every dimension to add the specific value to the corresponding SortedVectorSparse
          for (NICE::SparseVector::const_iterator elemIt = X[nr]->begin(); elemIt != X[nr]->end(); elemIt++)
          {
            //elemIt->first: dim, elemIt->second: value
            features[elemIt->first].insert( (T) elemIt->second, nr);
          }//for non-zero-values of the feature
        }//for every new feature
      }//if dimOverEx

      //set n for the internal data structure SortedVectorSparse
      for (typename std::vector<NICE::SortedVectorSparse<T> >::iterator it = features.begin(); it != features.end(); it++)
        (*it).setN(n);
    }

#ifdef NICE_USELIB_MATIO
    //Constructor reading data from matlab-files
    template <typename T>
    FeatureMatrixT<T>::
    FeatureMatrixT(const sparse_t & _features, const int & _dim)
    {
      if (_dim < 0)
        set_d(_features.njc -1);
      else
        set_d(_dim);
      
      int nMax(0);

      for ( int i = 0; i < _features.njc-1; i++ ) //walk over dimensions
      {
        for ( int j = _features.jc[i]; j < _features.jc[i+1] && j < _features.ndata; j++ ) //walk over single features, which are sparsely represented
        {
          features[i].insert(((T*)_features.data)[j], _features.ir[ j]);
          if ((_features.ir[ j])>nMax) nMax = _features.ir[ j];
        }
      }
      for (typename std::vector<NICE::SortedVectorSparse<T> >::iterator it = features.begin(); it != features.end(); it++)
      {
        (*it).setN(nMax+1);
      }
      n = nMax+1;
      verbose = false;
    }

    //Constructor reading data from matlab-files
    template <typename T>
    FeatureMatrixT<T>::
    FeatureMatrixT(const sparse_t & _features, const std::map<int, int> & examples, const int & _dim)
    {
      if (_dim < 0)
        set_d(_features.njc -1);
      else
        set_d(_dim);
      
      int nMax(0);

      for ( int i = 0; i < _features.njc-1; i++ ) //walk over dimensions
      {
        for ( int j = _features.jc[i]; j < _features.jc[i+1] && j < _features.ndata; j++ ) //walk over single features, which are sparsely represented
        {
          int example_index = _features.ir[ j];
          std::map<int, int>::const_iterator it = examples.find(example_index);
          if ( it != examples.end() ) {
            features[i].insert(((T*)_features.data)[j], it->second /* new index */);
            if (it->second > nMax) nMax = it->second;
          }
        }
      }
      for (typename std::vector<NICE::SortedVectorSparse<T> >::iterator it = features.begin(); it != features.end(); it++)
        (*it).setN(nMax+1);
    
      n = nMax+1;
      verbose = false;
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
    int FeatureMatrixT<T>::get_n() const
    {
      return n;
    }
      
    //  Get number of dimensions
    template <typename T>
    int FeatureMatrixT<T>::get_d() const
    {
      return d;
    }
      
    //  Sets the given dimension and re-sizes internal data structure. WARNING: this will completely remove your current data!
    template <typename T>
    void FeatureMatrixT<T>::set_d(const int & _d)
    {
      d = _d; features.resize(d);
    }
    
    template <typename T>
    void FeatureMatrixT<T>::setVerbose( const bool & _verbose)
    {
      verbose = _verbose;
    }
    
    template <typename T>
    bool FeatureMatrixT<T>::getVerbose( )   const
    {
      return verbose;
    } 
    
    template <typename T>
    void FeatureMatrixT<T>::setDebug( const bool & _debug)
    {
      debug = _debug;
    }
    
    template <typename T>
    bool FeatureMatrixT<T>::getDebug( )   const
    {
      return debug;
    }     
      
    //  Matrix-like operator for element access, performs validity check
    template <typename T>
    inline T FeatureMatrixT<T>::operator()(const int row, const int col) const
    {
      if ( (row < 0) || (col < 0) || (row > d) || (col > n) )
      {
        fthrow(Exception, "FeatureMatrixT: out of bounds");
      }
      else
        return (features[row]).access(col);
    }
    
    template<class T>
    inline bool
    FeatureMatrixT<T>::operator==(const FeatureMatrixT<T> & F) const
    {
      if ( ( (*this).get_n() != F.get_n()) || ((*this).get_d() != F.get_d()) )
      {
        fthrow(Exception, "FeatureMatrixT<T>::operator== : (n != F.get_n()) || (d != F.get_d()) -- number of dimensions does not fit");
      }
      else if ((n == 0) || (d == 0))
      {
        return true;
      }
      
      for (int i = 0; i < d; i++)
      {
        for (int j = 0; j < n; j++)
        {
          // FIXME: it would be more efficient if we compare SortedVectorSparse objects here
          if(!((*this)(i,j) == F(i,j)))
            return false;
        }
      }
      return true;
    }

    template<class T>
    inline bool
    FeatureMatrixT<T>::operator!=(const FeatureMatrixT<T> & F) const
    {
      if ( ( (*this).get_n() != F.get_n()) || ((*this).get_d() != F.get_d()) )
      {
        fthrow(Exception, "FeatureMatrixT::operator!=(): (n != F.get_n()) || (d != F.get_d()) -- number of dimensions does not fit");
      }
      else if ((n == 0) || (d == 0))
      {
        return false;
      }
      
      for (int i = 0; i < d; i++)
      {
        for (int j = 0; j < n; j++)
        {
          if(!((*this)(i,j) == F(i,j)))
            return true;
        }
      }
      return false;
    }
    
    template<typename T>
    inline FeatureMatrixT<T>&
    FeatureMatrixT<T>::operator=(const FeatureMatrixT<T> & F)
    {
      (*this).set_d(F.get_d());
      
      (*this).n = F.get_n();
      
      for (int i = 0; i < (*this).get_d(); i++)
      {
        // use the operator= of SortedVectorSparse
        features[i] = F[i];
      }
      
      return *this;
    }
    
    //  Element access without validity check
    template <typename T>
    inline T FeatureMatrixT<T>::getUnsafe(const int row, const int col) const
    {
      return (features[row]).access(col);
    }

    //! Element access of original values without validity check
    template <typename T>
    inline T FeatureMatrixT<T>::getOriginal(const int row, const int col) const
    {
      return (features[row]).accessOriginal(col);
    }

    //  Sets a specified element to the given value, performs validity check
    template <typename T>
    inline void FeatureMatrixT<T>::set (const int row, const int col, const T & newElement, bool setTransformedValue)
    {
      if ( (row < 0) || (col < 0) || (row > d) || (col > n) )
      {
        return;
      }
      else
        (features[row]).set ( col, newElement, setTransformedValue );
    }
    
    //  Sets a specified element to the given value, without validity check
    template <typename T>
    inline void FeatureMatrixT<T>::setUnsafe (const int row, const int col, const T & newElement, bool setTransformedValue)
    {
      (features[row]).set ( col, newElement, setTransformedValue );
    }
    
    //  Acceess to all element entries of a specified dimension, including validity check
    template <typename T>
    void FeatureMatrixT<T>::getDimension(const int & dim, NICE::SortedVectorSparse<T> & dimension) const
    {
      if ( (dim < 0) || (dim > d) )
      {
        return;
      }
      else
        dimension = features[dim];
    }
    
    //  Acceess to all element entries of a specified dimension, without validity check
    template <typename T>
    void FeatureMatrixT<T>::getDimensionUnsafe(const int & dim, NICE::SortedVectorSparse<T> & dimension) const
    {
      dimension = features[dim];
    }
    
    // Finds the first element in a given dimension, which equals elem
    template <typename T>
    void FeatureMatrixT<T>::findFirstInDimension(const int & dim, const T & elem, int & position) const
    {
      position = -1;
      if ( (dim < 0) || (dim > d))
        return;

      std::pair< typename SortedVectorSparse<T>::elementpointer, typename SortedVectorSparse<T>::elementpointer > eit;
      eit =  features[dim].nonzeroElements().equal_range ( elem );
      position = distance( features[dim].nonzeroElements().begin(), eit.first );
      if ( elem > features[dim].getTolerance() )
        position += features[dim].getZeros();

    }
    
    //  Finds the last element in a given dimension, which equals elem
    template <typename T>
    void FeatureMatrixT<T>::findLastInDimension(const int & dim, const T & elem, int & position) const
    {
      position = -1;
      if ( (dim < 0) || (dim > d))
        return;

      std::pair< typename SortedVectorSparse<T>::const_elementpointer, typename SortedVectorSparse<T>::const_elementpointer > eit =  features[dim].nonzeroElements().equal_range ( elem );
      position = distance( features[dim].nonzeroElements().begin(), eit.second );
      if ( elem > features[dim].getTolerance() )
        position += features[dim].getZeros();
    }
    
    //  Finds the first element in a given dimension, which is larger as elem
    template <typename T>
    void FeatureMatrixT<T>::findFirstLargerInDimension(const int & dim, const T & elem, int & position) const
    {
      position = -1;
      if ( (dim < 0) || (dim > d))
        return;
      
      //no non-zero elements?
      if (features[dim].getNonZeros() <= 0)
      {
        // if element is greater than zero, than is should be added at the last position
        if (elem > features[dim].getTolerance() )
          position = this->n;
        
        //if not, position is -1
        return;
      }
      
      if (features[dim].getNonZeros() == 1)
      {
        // if element is greater than the only nonzero element, than it is larger as everything else
        if (features[dim].nonzeroElements().begin()->first <= elem)
          position = this->n;
        
        //if not, but the element is still greater than zero, than 
        else if (elem > features[dim].getTolerance() )
          position = this->n -1;
          
        return;
      }
      
      typename SortedVectorSparse<T>::const_elementpointer it =  features[dim].nonzeroElements().end(); //this is needed !!!
      it = features[dim].nonzeroElements().upper_bound ( elem ); //if all values are smaller, this does not do anything at all
      
      position = distance( features[dim].nonzeroElements().begin(), it );

      if ( elem > features[dim].getTolerance() )
      {
        //position += features[dim].getZeros();
        position += n - features[dim].getNonZeros();
      }
    }
    
    //  Finds the last element in a given dimension, which is smaller as elem
    template <typename T>
    void FeatureMatrixT<T>::findLastSmallerInDimension(const int & dim, const T & elem, int & position) const
    {
      position = -1;
      if ( (dim < 0) || (dim > d))
        return;

      typename SortedVectorSparse<T>::const_elementpointer it =  features[dim].nonzeroElements().lower_bound ( elem );
      position = distance( features[dim].nonzeroElements().begin(), it );
      if ( it->first > features[dim].getTolerance() )
        position += features[dim].getZeros();
    }
    
    //------------------------------------------------------
    // high level methods
    //------------------------------------------------------

    template <typename T>
    void FeatureMatrixT<T>::applyFunctionToFeatureMatrix ( const NICE::ParameterizedFunction *pf )
    {
      if (pf != NULL)
      {
        // REMARK: might be inefficient due to virtual calls
        if ( !pf->isOrderPreserving() )
          fthrow(Exception, "ParameterizedFunction::applyFunctionToFeatureMatrix: this function is optimized for order preserving transformations");
        
        int d = this->get_d();
        for (int dim = 0; dim < d; dim++)
        {
          std::multimap< double, typename SortedVectorSparse<double>::dataelement> & nonzeroElements = this->getFeatureValues(dim).nonzeroElements();
          for ( SortedVectorSparse<double>::elementpointer i = nonzeroElements.begin(); i != nonzeroElements.end(); i++ )
          {
            SortedVectorSparse<double>::dataelement & de = i->second;
            
            //TODO check, wether the element is "sparse" afterwards
            de.second = pf->f( dim, i->first );
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
    double FeatureMatrixT<T>:: computeSparsityRatio()
    {
      double ratio(0.0);
      for (typename std::vector<NICE::SortedVectorSparse<T> >::const_iterator it = features.begin(); it != features.end(); it++)
      {
        ratio += (*it).getZeros() / (double) (*it).getN();
      }
      if (features.size() != 0)
        ratio /= features.size();
      return ratio;
    }

    //  add a new feature and insert its elements at the end of each dimension vector
    template <typename T>
    void FeatureMatrixT<T>::add_feature( const std::vector<T> & feature, const NICE::ParameterizedFunction *pf )
    {
      if (n == 0)
      {
        set_d(feature.size());
      }
      
      if ( (int)feature.size() != d)
      {
        fthrow(Exception, "FeatureMatrixT<T>::add_feature - number of dimensions does not fit");
        return;
      }

      for (int dimension = 0; dimension < (int) features.size(); dimension++)
      {
        if (pf != NULL)
          features[dimension].insert( feature[dimension], pf->f( dimension, feature[dimension]) );
        else  
          features[dimension].insert( feature[dimension] );        
      }
      n++;
    }
    //  add a new feature and insert its elements at the end of each dimension vector
    template <typename T>
    void FeatureMatrixT<T>::add_feature(const NICE::SparseVector & feature, const ParameterizedFunction *pf )
    {
      if (n == 0)
      {
        set_d(feature.size());
      }
      
      if ( (int)feature.getDim() > d)
      {
        fthrow(Exception, "FeatureMatrixT<T>::add_feature - number of dimensions does not fit");
        return;
      }

      for (NICE::SparseVector::const_iterator it = feature.begin(); it != feature.end(); it++)
      {
        if (pf != NULL)
          features[it->first].insert( (T) it->second, pf->f( it->first, (T) it->second), n );
        else  
          features[it->first].insert( (T) it->second, n );
      }
      n++;
    }    
      
    //  add several new features and insert their elements in the already ordered structure
    template <typename T>
    void FeatureMatrixT<T>::add_features(const std::vector<std::vector<T> > & _features )
    {
      //TODO do we need the parameterized function here as well? usually, we add several features and run applyFunctionToFeatureMatrix afterwards.
      // check this please :)
      
      //TODO assure that every feature has the same dimension
      if (n == 0)
      {
        set_d(_features.size());
      }
      
      //pay attention: we assume now, that we have a vector (over dimensions) containing vectors over features (examples per dimension) - to be more efficient
      for (int dim = 0; dim < d; dim++)
      {
          features[dim].insert( _features[dim] );
      }
      
      //update the number of our features
      n += (int) _features[0].size();
    }
    
    template <typename T>
    void FeatureMatrixT<T>::set_features(const std::vector<std::vector<T> > & _features, std::vector<std::vector<int> > & permutations, const int & _dim )
    {
      features.clear();
      if (_dim < 0)
        set_d(_features.size());
      else
        set_d(_dim);
      
      if (d>0)
        n = _features[0].size();
      
      //pay attention: we assume now, that we have a vector (over dimensions) containing vectors over features (examples per dimension) - to be more efficient
      for (int dim = 0; dim < d; dim++)
      {
        features[dim].insert( _features[dim] );
      }
    
      getPermutations( permutations );
    }

    template <typename T>
    void FeatureMatrixT<T>::set_features(const std::vector<std::vector<T> > & _features, std::vector<std::map<int,int> > & permutations, const int & _dim)
    {
      features.clear();
      if (_dim < 0)
        set_d(_features.size());
      else
        set_d(_dim);
      if (d>0)
        n = _features[0].size();
           
      //pay attention: we assume now, that we have a vector (over dimensions) containing vectors over features (examples per dimension) - to be more efficient
      for (int dim = 0; dim < d; dim++)
      {
        features[dim].insert( _features[dim] );
      }
    
      getPermutations( permutations );
    }
    
    template <typename T>
    void FeatureMatrixT<T>::set_features(const std::vector<std::vector<T> > & _features, const int & _dim)
    {
      features.clear();
      if (_dim < 0)
        set_d(_features.size());
      else
        set_d(_dim);
      
      if (d>0)
        n = _features[0].size();
      
      //pay attention: we assume now, that we have a vector (over dimensions) containing vectors over features (examples per dimension) - to be more efficient
      for (int dim = 0; dim < d; dim++)
      {
        features[dim].insert( _features[dim] );
      }
    }
    
    template <typename T>
    void FeatureMatrixT<T>::set_features(const std::vector< const NICE::SparseVector * > & _features, const bool dimensionsOverExamples, const int & _dim)
    {   
      features.clear();
      if (_features.size() == 0)
      {
        std::cerr << "set_features without features" << std::endl;
      }
            
      // resize our data structure      
      if (_dim >= 0) //did the user specified the number of dimensions?
        set_d(_dim);
      else //dimensions not specified by users
      {
        if (dimensionsOverExamples) //do we have dim x examples ?
        {
          set_d(_features.size());
        }
        else //we have examples x dimes (as usually done)
        {
          if (_features.size() > 0) //and have at least one example
          {
            try{
              set_d(_features[0]->getDim());  
            }
            catch(...)
            {
              std::cerr << "FeatureMatrixT<T>::set_features -- something went wrong using getDim() of SparseVectors" << std::endl;
            }
          }
          else //no example, so set the dim to 0, since we have no idea at all
          {
            set_d(0);
          }          
        }
      }  
            
      // set number of examples n
      if (d>0)
      {
        if (dimensionsOverExamples) //do we have dim x examples ?
          n = _features[0]->getDim(); //NOTE Pay attention: we assume, that this number is set!
        else //we have examples x dimes (as usually done)   
          n = _features.size(); 
      }       
            
      // insert all values
      if (dimensionsOverExamples) //do we have dim x examples ?
      {
        for (int dim = 0; dim < d; dim++)
        {
          features[dim].insert( _features[dim] );
        }
      }
      else //we have examples x dimes (as usually done)
      {
        if ( debug )
          std::cerr << "FeatureMatrixT<T>::set_features " << n << " new examples" << std::endl;
        //loop over every example to add its content
        for (int nr = 0; nr < n; nr++)
        {
          if ( debug )
            std::cerr << "add feature nr. " << nr << " / " << _features.size() << " ";
          //loop over every dimension to add the specific value to the corresponding SortedVectorSparse
          for (NICE::SparseVector::const_iterator elemIt = _features[nr]->begin(); elemIt != _features[nr]->end(); elemIt++)
          {
            if ( debug )
              std::cerr << elemIt->first << "-" << elemIt->second << " ";
            //elemIt->first: dim, elemIt->second: value
            features[elemIt->first].insert( (T) elemIt->second, nr);
          }//for non-zero-values of the feature
          if ( debug )
            std::cerr << std::endl;
        }//for every new feature
        if ( debug )
          std::cerr << "FeatureMatrixT<T>::set_features done" << std::endl;
      }//if dimOverEx
      
      //set n for the internal data structure SortedVectorSparse
      for (typename std::vector<NICE::SortedVectorSparse<T> >::iterator it = features.begin(); it != features.end(); it++)
        (*it).setN(n);
    }

    template <typename T>
    void FeatureMatrixT<T>::getPermutations( std::vector<std::vector<int> > & permutations) const
    {
      for (int dim = 0; dim < d; dim++)
      {
        std::vector<int> perm (  (features[dim]).getPermutation() );
        permutations.push_back(perm);
      }
    }
    
    template <typename T>
    void FeatureMatrixT<T>::getPermutations( std::vector<std::map<int,int> > & permutations) const
    {
      for (int dim = 0; dim < d; dim++)
      {
        std::map<int,int> perm (  (features[dim]).getPermutationNonZeroReal() );
        permutations.push_back(perm);
      }
    }

      
    //  Prints the whole Matrix (outer loop over dimension, inner loop over features)
    template <typename T>
    void FeatureMatrixT<T>::print(std::ostream & os) const
    {
      if (os.good())
      {
        for (int dim = 0; dim < d; dim++)
        {
          features[dim].print(os);
        }
      }
    }
    
    // Computes the whole non-sparse matrix. WARNING: this may result in a really memory-consuming data-structure!
    template <typename T>
    void FeatureMatrixT<T>::computeNonSparseMatrix(NICE::MatrixT<T> & matrix, bool transpose) const
    {
      if ( transpose )
        matrix.resize(this->get_n(),this->get_d());
      else
        matrix.resize(this->get_d(),this->get_n());

      matrix.set((T)0.0);
      int dimIdx(0);
      for (typename std::vector<NICE::SortedVectorSparse<T> >::const_iterator it = features.begin(); it != features.end(); it++, dimIdx++)
      {
        std::map< int, typename NICE::SortedVectorSparse<T>::elementpointer>  nonzeroIndices= (*it).nonzeroIndices();
        for (typename std::map< int, typename NICE::SortedVectorSparse<T>::elementpointer>::const_iterator inIt = nonzeroIndices.begin(); inIt != nonzeroIndices.end(); inIt++)
        {
          int featIndex = ((*inIt).second)->second.first;
          if ( transpose ) 
            matrix(featIndex,dimIdx) =((*inIt).second)->second.second; 
          else
            matrix(dimIdx,featIndex) =((*inIt).second)->second.second; 
        }
      }
    }

    // Computes the whole non-sparse matrix. WARNING: this may result in a really memory-consuming data-structure!
    template <typename T>
    void FeatureMatrixT<T>::computeNonSparseMatrix(std::vector<std::vector<T> > & matrix, bool transpose) const
    {
      if ( transpose )
        matrix.resize(this->get_n());
      else
        matrix.resize(this->get_d());
      
      // resizing the matrix
      for ( uint i = 0 ; i < matrix.size(); i++ )
        if ( transpose )
          matrix[i] = std::vector<T>(this->get_d(), 0.0);
        else
          matrix[i] = std::vector<T>(this->get_n(), 0.0);

      int dimIdx(0);
      for (typename std::vector<NICE::SortedVectorSparse<T> >::const_iterator it = features.begin(); it != features.end(); it++, dimIdx++)
      {
        std::map< int, typename NICE::SortedVectorSparse<T>::elementpointer>  nonzeroIndices= (*it).nonzeroIndices();
        for (typename std::map< int, typename NICE::SortedVectorSparse<T>::elementpointer>::const_iterator inIt = nonzeroIndices.begin(); inIt != nonzeroIndices.end(); inIt++)
        {
          int featIndex = ((*inIt).second)->second.first;
          if ( transpose )
            matrix[featIndex][dimIdx] =((*inIt).second)->second.second; 
          else
            matrix[dimIdx][featIndex] =((*inIt).second)->second.second; 
        }
      }
    }
    
    // Swaps to specified elements, performing a validity check
    template <typename T>
    void FeatureMatrixT<T>::swap(const int & row1, const int & col1, const int & row2, const int & col2)
    {
      if ( (row1 < 0) || (col1 < 0) || (row1 > d) || (col1 > n) || (row2 < 0) || (col2 < 0) || (row2 > d) || (col2 > n))
      {
        return;
      }
      else
      {
        //swap
        T tmp = (*this)(row1, col1);
        (*this).set(row1, col1, (*this)(row2,col2));
        (*this).set(row2, col2, tmp);
      }
    }
    
    //  Swaps to specified elements, without performing a validity check
    template <typename T>
    void FeatureMatrixT<T>::swapUnsafe(const int & row1, const int & col1, const int & row2, const int & col2)
    {
      //swap
      T tmp = (*this)(row1, col1);
      (*this).set(row1, col1, (*this)(row2,col2));
      (*this).set(row2, col2, tmp);
    }

    template <typename T>
    void FeatureMatrixT<T>::hikDiagonalElements( Vector & diagonalElements ) const
    {
      int dimIdx = 0;
      // the function calculates the diagonal elements of a HIK kernel matrix
      diagonalElements.resize(n);
      diagonalElements.set(0.0);
      // loop through all dimensions
      for (typename std::vector<NICE::SortedVectorSparse<T> >::const_iterator it = features.begin(); it != features.end(); it++, dimIdx++)
      {
        std::map< int, typename NICE::SortedVectorSparse<T>::elementpointer>  nonzeroIndices= (*it).nonzeroIndices();
        // loop through all features
        for (typename std::map< int, typename NICE::SortedVectorSparse<T>::elementpointer>::const_iterator inIt = nonzeroIndices.begin(); inIt != nonzeroIndices.end(); inIt++)
        {
          int index = inIt->first;
          typename NICE::SortedVectorSparse<T>::elementpointer p = inIt->second;
          typename NICE::SortedVectorSparse<T>::dataelement de = p->second;

          diagonalElements[index] += de.second;
        }
      }
    }

    template <typename T>
    double FeatureMatrixT<T>::hikTrace() const
    {
      Vector diagonalElements;
      hikDiagonalElements( diagonalElements );
      return diagonalElements.Sum();

    }
    
    template <typename T>
    int FeatureMatrixT<T>::getNumberOfNonZeroElementsPerDimension(const int & dim) const
    {
      if ( (dim < 0) || (dim > d))
        return -1;
      return features[dim].getNonZeros();
    }

    template <typename T>
    int FeatureMatrixT<T>::getNumberOfZeroElementsPerDimension(const int & dim) const
    {
      if ( (dim < 0) || (dim > d))
        return -1;
      return n - features[dim].getNonZeros();
    }
    

    template <typename T>
    void FeatureMatrixT<T>::restore ( std::istream & is, int format )
    {
      bool b_restoreVerbose ( false );
      if ( is.good() )
      {
	if ( b_restoreVerbose ) 
	  std::cerr << " restore FeatureMatrixT" << std::endl;
	
	std::string tmp;
	is >> tmp; //class name 
	
	if ( ! this->isStartTag( tmp, "FeatureMatrixT" ) )
	{
	    std::cerr << " WARNING - attempt to restore FeatureMatrixT, but start flag " << tmp << " does not match! Aborting... " << std::endl;
	    throw;
	}   
	    
	is.precision ( std::numeric_limits<double>::digits10 + 1);
	
	bool b_endOfBlock ( false ) ;
	
	while ( !b_endOfBlock )
	{
	  is >> tmp; // start of block 
	  
	  if ( this->isEndTag( tmp, "FeatureMatrixT" ) )
	  {
	    b_endOfBlock = true;
	    continue;
	  }      
	  
	  tmp = this->removeStartTag ( tmp );
	  
	  if ( b_restoreVerbose )
	    std::cerr << " currently restore section " << tmp << " in FeatureMatrixT" << std::endl;
	  
	  if ( tmp.compare("n") == 0 )
	  {
	    is >> n;        
	    is >> tmp; // end of block 
	    tmp = this->removeEndTag ( tmp );
	  }
	  else if ( tmp.compare("d") == 0 )
	  {
	    is >> d;        
	    is >> tmp; // end of block 
	    tmp = this->removeEndTag ( tmp );
	  } 
	  else if ( tmp.compare("features") == 0 )
	  {
	    //NOTE assumes d to be read first!
	    features.resize(d);
	    //now read features for every dimension
	    for (int dim = 0; dim < d; dim++)
	    {
	      NICE::SortedVectorSparse<T> svs;
	      features[dim] = svs;          
	      features[dim].restore(is,format);
	    }
	    
	    is >> tmp; // end of block 
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
    void FeatureMatrixT<T>::store ( std::ostream & os, int format ) const
    {
      if (os.good())
      {
	// show starting point
	os << this->createStartTag( "FeatureMatrixT" ) << std::endl;
	
        os.precision (std::numeric_limits<double>::digits10 + 1);
	
	os << this->createStartTag( "n" ) << std::endl;
	os << n << std::endl;
	os << this->createEndTag( "n" ) << std::endl;
	
	
	os << this->createStartTag( "d" ) << std::endl;
	os << d << std::endl;
	os << this->createEndTag( "d" ) << std::endl;
        
        //now write features for every dimension
	os << this->createStartTag( "features" ) << std::endl;
	for (int dim = 0; dim < d; dim++)
	{
	  features[dim].store(os,format);
	}
        os << this->createEndTag( "features" ) << std::endl;
        
	// done
	os << this->createEndTag( "FeatureMatrixT" ) << std::endl;       
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
