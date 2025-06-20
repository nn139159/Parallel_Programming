# ========================
# Platform Detection
# ========================
UNAME_S := $(shell uname -s)
ARCH := $(shell uname -m)

# Detect OS Type
ifeq ($(UNAME_S),Linux)
    IS_LINUX := 1
    RM = rm -f
else
    IS_LINUX := 0
    RM = del /Q
endif

# Detect Architecture
ifeq ($(ARCH),aarch64)
    IS_ARM := 1
else
    IS_ARM := 0
endif

# ========================
# Compiler + Flags
# ========================
CXX := g++
NVCC := nvcc

CXXFLAGS := -O3 -std=c++17 -fopenmp
NVCCFLAGS := -O3 --compiler-bindir=/usr/bin/g++-7 -std=c++14 

# Platform-specific SIMD flags
ifeq ($(IS_ARM),1)
    CXXFLAGS += -march=native
else
    CXXFLAGS += -mavx2
endif

# ========================
# Optional CUDA build
# Usage: make USE_CUDA=1
# ========================
USE_CUDA ?= 0

# ========================
# Source Files
# ========================
COMMON_SRC := \
    Data_Processing.cpp \
    NearestNeighborCollector.cpp \
    LBPH_Serial.cpp \
    LBPH_OpenMP.cpp

MAIN_SRC := main.cpp
OMPTEST_SRC := omp_compare.cpp

ifeq ($(USE_CUDA),1)
    CUDA_SRC := LBPH_Cuda.cpp
    CUDA_CU := lbphCuda.cu
    OBJS := $(COMMON_SRC:.cpp=.o) $(CUDA_SRC:.cpp=.o) lbphCuda.o
    LINKER := $(NVCC)
    LDFLAGS += -Xcompiler "-fopenmp" -L/usr/local/cuda/lib64 -lcudart
    CXXFLAGS += -DUSE_CUDA=$(USE_CUDA)
else
    OBJS := $(COMMON_SRC:.cpp=.o)
    LINKER := $(CXX)
    LDFLAGS += -fopenmp
endif

MAIN_OBJS := $(MAIN_SRC:.cpp=.o) $(OBJS)
OMPTEST_OBJS := $(OMPTEST_SRC:.cpp=.o) $(OBJS)

MAIN_EXE := main
OMPTEST_EXE := omp_compare

# ========================
# Build Rules
# ========================
.PHONY: all clean run

all: $(MAIN_EXE) $(OMPTEST_EXE)

$(MAIN_EXE): $(MAIN_OBJS)
	$(LINKER) -o $@ $^ $(LDFLAGS)

$(OMPTEST_EXE): $(OMPTEST_OBJS)
	$(LINKER) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

ifeq ($(USE_CUDA),1)
lbphCuda.o: lbphCuda.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
endif

clean:
	$(RM) *.o *.exe *.out $(MAIN_EXE) $(OMPTEST_EXE)

