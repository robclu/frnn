########################################################
#					EXECUTABLE NAME 				   #
########################################################

EXE 			:= all_tests

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

CUDA_LIBS 		:= -lcuda -lcublas
TEST_LIBS 		:= -lgtest -lgtest_main \
				   -lpthread

LIB_DIR 		:= -L/usr/local/cuda/lib64

########################################################
#					COMPILER FLAGS 					   #
#                                                      #
# To enable compiler warnings, remove -w and replace   #
# with :                                               #
# 			--compiler-options -Wall                   #
########################################################

CCFLAGS 		:= -std=c++11 -O3 -w
CUFLAGS 		:= -arch=sm_30

########################################################
# 					TARGET RULES 					   #
########################################################

all: tests cleanobs 

tests: tests

##################### UTIL #############################

util_tests.o: util/util_tests.cpp 
	$(NVCC) -x cu $(INCLUDES) $(CCFLAGS) $(CUFLAGS) -o $@ -c $<

errors.o: util/errors.cpp 
	$(NVCC) -x cu $(INCLUDES) $(CCFLAGS) $(CUFLAGS) -o $@ -c $<

##################### GENERAL ##########################

general_tests.o : curnn/general_tests.cpp
	$(NVCC) -x cu $(INCLUDES) $(CCFLAGS) $(CUFLAGS) -o $@ -c $<

###################### MATH ############################

math.o : math/math.cpp
	$(NVCC) -x cu $(INLCUDES) $(CCFLAGS) $(CUFLAGS) -o $@ -dc $<

math_tests.o: math/math_tests.cpp 
	$(NVCC) -x cu $(INCLUDES) $(CCFLAGS) $(CUFLAGS) -o $@ -c $<

###################### MAIN ############################

main.o : main.cpp
	$(NVCC) $(INCLUDES) $(CCFLAGS) $(CUFLAGS) -o $@ -c $<

tests: errors.o util_tests.o general_tests.o \
	   math.o math_tests.o main.o
	$(NVCC) $(LDFLAGS) -o $(EXE) $+ $(LIB_DIR) \
		$(CUDA_LIBS) $(TEST_LIBS)	
		
cleanobs:
	rm -rf *.o

clean:
	rm -rf *.o
	rm -rf $(EXE) 

clobber: clean
