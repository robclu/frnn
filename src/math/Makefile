########################################################
#					EXECUTABLE NAME 				   #
########################################################

EXE 			:= math_tests

########################################################
#					COMPILERS						   #
########################################################

HOST_COMPILER	:= g++
NVCC 			:= nvcc -ccbin $(HOST_COMPILER)
CXX 			:= $(HOST_COMPILER)

########################################################
#				INCLUDE DIRECTORIES 				   #
########################################################

INCLUDES 		:= -I/usr/local/cuda-7.0/include \
				   -I../Util

########################################################
#					LIBRARIES 						   #
########################################################

CUDA_LIBS 		:= -lcuda -lcublas
TEST_LIBS 		:= -lgtest -lgtest_main \
				   -lpthread

LIB_DIR 		:= -L/usr/local/cuda/lib64

########################################################
#					COMPILER FLAGS 					   #
########################################################

CCFLAGS 		:= -std=c++11 \
				   #--relaxed-constexpr

########################################################
# 					TARGET RULES 					   #
########################################################

all: tests cleanobs 

tests: tests

math.o : math.cpp
	$(NVCC) -x cu $(INLCUDES) $(CCFLAGS) -o $@ -dc $<

math_tests.o: math_tests.cpp 
	$(CXX) $(INCLUDES) $(CCFLAGS) -o $@ -c $<

errors.o: ../util/errors.cpp 
	$(NVCC) $(INCLUDES) $(CCFLAGS) -o $@ -c $<

tests: errors.o math.o math_tests.o
	$(NVCC) $(LDFLAGS) -o $(EXE) $+ $(LIB_DIR) \
		$(CUDA_LIBS) $(TEST_LIBS)	
		
cleanobs:
	rm -rf *.o

clean:
	rm -rf *.o
	rm -rf $(EXE) 

clobber: clean
