########################################################
#					EXECUTABLE NAME					   #
########################################################

EXE				:= util_tests

########################################################
#					COMPILERS						   #
########################################################

HOST_COMPILER	:= g++
NVCC			:= nvcc -ccbin $(HOST_COMPILER)
CXX				:= $(HOST_COMPILER)

########################################################
#				INCLUDE DIRECTORIES					   #
########################################################

INCLUDES		:= -I/usr/local/cuda-7.0/include -I.

########################################################
#					LIBRARIES						   #
########################################################

CUDA_LIBS		:= -lcuda -lcublas
TEST_LIBS		:= -lgtest -lgtest_main \
				   -lpthread

LIB_DIR			:= -L/usr/local/cuda/lib64

########################################################
#					COMPILER FLAGS					   #
#                                                      #
# To enable compiler warnings, remove -w and replace   #
# with :                                               #
#			--compiler-options -Wall                   #
########################################################

CCFLAGS			:= -std=c++11 -O3 -w
CUFLAGS			:= -arch=sm_30

########################################################
#					TARGET RULES					   #
########################################################

all: tests

tests: tests

util_tests.o: util_tests.cpp 
	$(NVCC) -x cu $(INCLUDES) $(CCFLAGS) $(CUFLAGS) -o $@ -c $<

errors.o: errors.cpp 
	$(NVCC) -x cu $(INCLUDES) $(CCFLAGS) $(CUFLAGS) -o $@ -c $<

main.o : main.cpp
	$(CXX) $(INCLUDES) $(CCFLAGS) -o $@ -c $<

tests: errors.o util_tests.o main.o
	$(NVCC) $(LDFLAGS) -o $(EXE) $+ $(LIB_DIR) \
		$(CUDA_LIBS) $(TEST_LIBS)	
		
cleanobs:
	rm -rf *.o

clean:
	rm -rf *.o
	rm -rf $(EXE) 

clobber: clean
