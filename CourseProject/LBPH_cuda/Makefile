# Makefile for nmake (MSVC)

#================== Compiler Settings ==================
CXX = cl.exe
NVCC = nvcc.exe

CXXFLAGS = /nologo /O2 /openmp /MD /W3 /EHsc /std:c++17
NVCCFLAGS = -O3 -std=c++17 -Xcompiler "/openmp /O2 /MD /std:c++17"

#================== Source Files ==================
CPP_SRC = main.cpp Data_Processing.cpp LightLBPH.cpp NearestNeighborCollector.cpp
CU_SRC = lbph_cuda.cu

OBJ = main.obj Data_Processing.obj LightLBPH.obj NearestNeighborCollector.obj lbph_cuda.obj
TARGET = main.exe

#================== Build Rules ==================
all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(NVCCFLAGS) $(OBJ) -o $(TARGET)

# C++ Archive compilation rules
main.obj: main.cpp
	$(CXX) $(CXXFLAGS) /c main.cpp /Fo$@

Data_Processing.obj: Data_Processing.cpp
	$(CXX) $(CXXFLAGS) /c Data_Processing.cpp /Fo$@

LightLBPH.obj: LightLBPH.cpp
	$(CXX) $(CXXFLAGS) /c LightLBPH.cpp /Fo$@

NearestNeighborCollector.obj: NearestNeighborCollector.cpp
	$(CXX) $(CXXFLAGS) /c NearestNeighborCollector.cpp /Fo$@

# CUDA Archive compilation rules
lbph_cuda.obj: lbph_cuda.cu
	$(NVCC) $(NVCCFLAGS) -c lbph_cuda.cu -o lbph_cuda.obj

clean:
	del /Q *.obj *.exe *.pdb *.exp *.lib *.bak *.o
