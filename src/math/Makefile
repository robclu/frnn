########################################################
#					EXECUTABLE NAME 				   #
########################################################

EXE 			:= mathTests

########################################################
#					COMPILERS						   #
########################################################

HOST_COMPILER	:= g++
NVCC 			:= nvcc -ccbin $(HOST_COMPILER)

########################################################
#				INCLUDE DIRECTORIES 				   #
########################################################

INCLUDES 		:= -I/usr/local/cuda-7.0/include

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

CCFLAGS 		:= -g -std=c++11 \
				   #--relaxed-constexpr

########################################################
# 					TARGET RULES 					   #
########################################################

all: tests cleanobs 

tests: tests

math.o : math.cpp
	$(NVCC) -x cu $(INLCUDES) $(CCFLAGS) -o $@ -c $<

tests_math.o: tests_math.cpp math.cpp
	$(NVCC)  -x cu $(INCLUDES) $(CCFLAGS) -o $@ -c $<

tests: math.o tests_math.o
	$(NVCC) $(LDFLAGS) -o $(EXE) $+ $(LIB_DIR) \
		$(CUDA_LIBS) $(TEST_LIBS)
	@rm -rf *.o	
		
cleanobs:
	rm -rf *.o

clean:
	rm -rf *.o
	rm -rf $(EXE) 

clobber: clean
