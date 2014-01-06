NICEFLAGS1=$(shell pkg-config libgp-hik-core --cflags --libs)
NICEFLAGS=$(subst -fopenmp,,$(NICEFLAGS1))

default:
	/home/matlab/7.14/bin/mex ${NICEFLAGS} -largeArrayDims GPHIK.cpp 
