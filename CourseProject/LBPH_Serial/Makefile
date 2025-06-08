# Detect platform
ifeq ($(OS),Windows_NT)
    # Windows (MSYS2 or MinGW)
    EXE := .exe
    RM := del /Q
    CXX := C:/msys64/mingw64/bin/g++.exe
else
    # Assume Linux (Jetson Nano or Raspberry Pi)
    EXE :=
    RM := rm -f
    CXX := g++
endif

CXXFLAGS := -O3 -std=c++17 -Wall -fdiagnostics-color=always -g

# Source and target
SRC := main.cpp Data_Processing.cpp LightLBPH.cpp NearestNeighborCollector.cpp
TARGET := main$(EXE)

# Build rule
all: $(TARGET)

$(TARGET): $(SRC)
	"$(CXX)" $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	$(RM) $(TARGET)
