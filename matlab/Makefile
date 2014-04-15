MEX=/home/matlab/7.14/bin/mex
#MEX=/home/matlab/8.2/academic/bin/mex
NICEFLAGS1=$(shell pkg-config libgp-hik-core --cflags --libs)
NICEFLAGS=$(subst -fopenmp,,$(NICEFLAGS1))

default:
	${MEX} ${NICEFLAGS} -largeArrayDims GPHIKClassifierMex.cpp ConverterMatlabToNICE.cpp ConverterNICEToMatlab.cpp
	${MEX} ${NICEFLAGS} -largeArrayDims GPHIKRegressionMex.cpp ConverterMatlabToNICE.cpp ConverterNICEToMatlab.cpp
	${MEX} ${NICEFLAGS} -largeArrayDims testMatlabConversionFunctionsMex.cpp ConverterMatlabToNICE.cpp ConverterNICEToMatlab.cpp

classification:
	${MEX} ${NICEFLAGS} -largeArrayDims GPHIKClassifierMex.cpp ConverterMatlabToNICE.cpp ConverterNICEToMatlab.cpp

regression:        
	${MEX} ${NICEFLAGS} -largeArrayDims GPHIKRegressionMex.cpp ConverterMatlabToNICE.cpp ConverterNICEToMatlab.cpp

unittest:        
	${MEX} ${NICEFLAGS} -largeArrayDims testMatlabConversionFunctionsMex.cpp ConverterMatlabToNICE.cpp ConverterNICEToMatlab.cpp