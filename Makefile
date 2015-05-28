TARGET_NAME     = cublrnn
SRC_DIR         = src
OBJ_DIR         = obj
REL_DIR         = release

TARGET          = $(addprefix $(REL_DIR)/,$(TARGET_NAME))

NVCC            = nvcc
CXX             = g++

CUD_INCLUDES    = /usr/local/cuda-7.0/include
CXX_INCLUDES    = 

CUD_LDFLAGS     = -lcuda
CXX_LDFLAGS     = 

CUD_FLAGS       = $(GEN_SM35)
CUD_FLAGS       += $(foreach includedir,$(CUD_INCLUDES),-I$(includedir))
CXX_FLAGS       = -std=c++11

CPP_FILES       = $(wildcard $(SRC_DIR)/*/*.cpp)
CU_FILES        = $(wildcard $(SRC_DIR)/*/*.cu)

H_FILES         = $(wildcard $(SRC_DIR)/*/*.h)
CUH_FILES       = $(wildcard $(SRC_DIR)/*/*.cuh)

OBJ_FILES       = $(addprefix $(OBJ_DIR)/,$(notdir $(CPP_FILES:.cpp=.o)))
CUO_FILES       = $(addprefix $(OBJ_DIR)/,$(notdir $(CU_FILES:.cu=.

OBJS            =  $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(CPP_FILES)))
OBJS            += $(patsubst %.cu,$(OBJ_DIR)/%.cu.o,$(notdir $(CU_FILES)))

.PHONY: all clean

all: $(TARGET)

$(TARGET) : $(OBJS)
	echo "linking rule : " -o $@ $?
	$(NVCC) $(CUD_LDFLAGS) -o $@ $?

$(OBJ_DIR)/%.cu.o : $(SRC_DIR)/*/%.cu $(CUH_FILES)
	echo ".cu.o rule : " $@ $<
	touch $@
	$(NVCC) $(CUD_FLAGS) $(CXX_FLAGS) $(CXX_INCLUDES) -c -o $@ $<

$(OBJ_DIR)/%.o : $(SRC_DIR)/*/%.cpp $(H_FILES)
	echo ".o rule : " $@ $<
	$(NVCC) $(CUD_FLAGS) $(CXX_FLAGS) $(CXX_INCLUDES) -c -o $@ $<	
	touch $@

clean:
	$(OBJ_DIR)/*.o
