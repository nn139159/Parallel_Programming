TARGET := pi.out

CXX := g++

CXXFLAGS := -std=c++17 -O3 -mavx2 -Wall -pthread

.PHONY: all
all: $(TARGET)

%.out: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -f *.o $(TARGET)

