NICEFLAGS1=$(shell pkg-config libgp-hik-core --cflags --libs)
NICEFLAGS=$(subst -fopenmp,,$(NICEFLAGS1))

default:
	/home/alex/bin/matlab/7.14/bin/mex ${NICEFLAGS} -largeArrayDims GPHIK.cpp ConverterMatlabToNICE.cpp ConverterNICEToMatlab.cpp
