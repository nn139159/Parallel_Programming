# Detect OS and Architecture
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Default compiler and flags
CXX := g++
CXXFLAGS := -O3 -std=c++17 -Wall -fopenmp -fpermissive -fdiagnostics-color=always -g

# Platform-specific settings
ifeq ($(OS),Windows_NT)
    EXE := .exe
    RM := del /Q
    CXX := C:/msys64/mingw64/bin/g++.exe
    CXXFLAGS += -mavx2
else
    EXE :=
    RM := rm -f
    ifeq ($(UNAME_M),x86_64)
        CXXFLAGS += -mavx2
    else ifeq ($(findstring arm,$(UNAME_M)),arm)
        CXXFLAGS += -mfpu=neon
    else ifeq ($(findstring aarch64,$(UNAME_M)),aarch64)
        CXXFLAGS += -mfpu=neon
    endif
endif

# Source and target
SRC := main.cpp Data_Processing.cpp LightLBPH.cpp NearestNeighborCollector.cpp
TARGET := main$(EXE)

# Build rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	$(RM) $(TARGET)