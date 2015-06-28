########################################################
#					EXECUTABLE NAME 				   #
########################################################

EXE 			:= general_tests

########################################################
#					COMPILERS						   #
########################################################

HOST_COMPILER	:= g++
NVCC 			:= nvcc -ccbin $(HOST_COMPILER)
CXX 			:= $(HOST_COMPILER)

########################################################
#				INCLUDE DIRECTORIES 				   #
########################################################

INCLUDES 		:= -I/usr/local/cuda-7.0/include -I.

########################################################
#					LIBRARIES 						   #
########################################################

CUDA_LIBS 		:= -lcuda
TEST_LIBS 		:= -lgtest -lgtest_main \
				   -lpthread

LIB_DIR 		:= -L/usr/local/cuda/lib64

########################################################
#					COMPILER FLAGS 					   #
########################################################

CCFLAGS 		:= -std=c++11 -O3

########################################################
# 					TARGET RULES 					   #
########################################################

all: tests cleanobs 

tests: tests

general_tests.o: general_tests.cpp 
	$(NVCC) -x cu $(INCLUDES) $(CCFLAGS) -o $@ -c $<

tests: general_tests.o
	$(NVCC) $(LDFLAGS) -o $(EXE) $+ $(LIB_DIR) \
		$(CUDA_LIBS) $(TEST_LIBS)	
		
cleanobs:
	rm -rf *.o

clean:
	rm -rf *.o
	rm -rf $(EXE) 

clobber: clean
