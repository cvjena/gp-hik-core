% brief:    Unit testing of the NICE::MatlabConversion functions
% author:   Johannes Ruehle
% date:     11-04-2014 (dd-mm-yyyy)
%% test convertInt32
t = int32( 23 );
[r] = testMatlabConversionFunctionsMex( 'convertInt32', t );
assert( t == r);

t = single( 23 );
try
    [r] = testMatlabConversionFunctionsMex( 'convertInt32', t );
catch ecpn
    assert( strcmp( ecpn.message,'Expected int32'));
end

%% test logical
t = true;
[r] = testMatlabConversionFunctionsMex( 'convertLogical', t );
assert( t == r);

t = 1;
try
    [r] = testMatlabConversionFunctionsMex( 'convertLogical', t );
catch ecpn
    assert( strcmp( ecpn.message,'Expected bool'));
end

%% test convertDouble
t = double( 23 );
[r] = testMatlabConversionFunctionsMex( 'convertDouble', t );
assert( t == r);

t = single( 23 );
try
    [r] = testMatlabConversionFunctionsMex( 'convertDouble', t );
catch ecpn
    assert( strcmp( ecpn.message,'Expected double'));
end

t = double( [42, 23]);
[r] = testMatlabConversionFunctionsMex( 'convertDouble', t );
assert( t(1) == r);

%% test convertDoubleVector
t = double( 23 );
[r] = testMatlabConversionFunctionsMex( 'convertDoubleVector', t );
assert( t == r);

t = single( 23 );
try
    [r] = testMatlabConversionFunctionsMex( 'convertDoubleVector', t );
catch ecpn
    assert( strcmp( ecpn.message,'Expected double in convertDoubleVectorToNice'));
end

t = double( [42, 23]);
[r] = testMatlabConversionFunctionsMex( 'convertDoubleVector', t );
%r is a row vector, not an column vector anymore
assert( size(t,1) == size(r,2) && size(t,2) == size(r,1) && all( t == r' ) );

t = double( [123; 234]);
[r] = testMatlabConversionFunctionsMex( 'convertDoubleVector', t );
%r is a row vector, not an column vector
assert( size(t,1) == size(r,1) && size(t,1) == size(r,1) && all( t == r ) );

%% test convertDoubleMatrix
t = double( 23 );
[r] = testMatlabConversionFunctionsMex( 'convertDoubleMatrix', t );
assert( t == r);

t = single( 23 );
try
    [r] = testMatlabConversionFunctionsMex( 'convertDoubleMatrix', t );
catch ecpn
    assert( strcmp( ecpn.message,'Expected double in convertDoubleMatrixToNice'));
end

t = double( [42, 23]);
[r] = testMatlabConversionFunctionsMex( 'convertDoubleMatrix', t );
assert( size(t,1) == size(r,1) && size(t,1) == size(r,1) && all( t == r ) );

t = double( [123; 234]);
[r] = testMatlabConversionFunctionsMex( 'convertDoubleMatrix', t );
assert( size(t,1) == size(r,1) && size(t,1) == size(r,1) && all( t == r ) );

t = rand(5,5,'double');
[r] = testMatlabConversionFunctionsMex( 'convertDoubleMatrix', t );
d = r-t;
assert( size(t,1) == size(r,1) && size(t,1) == size(r,1) && sum( d(:)) == 0 );

%% test convertDoubleSparseVector

t = sparse( 5,1);
t(1) = 23; t(3) = 123; t(4) = 42;
[r] = testMatlabConversionFunctionsMex( 'convertDoubleSparseVector', t );
d = r-t;
assert( size(t,1) == size(r,1) && size(t,1) == size(r,1) && sum( d(:)) == 0 );

