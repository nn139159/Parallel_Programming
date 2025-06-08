#pragma once
#include <limits>

// Prediction collector used by LBPH: record the label with the smallest distance
class NearestNeighborCollector {
public:
    NearestNeighborCollector();

    // Initialize the collector (called before each predict)
    void init(int size);

    // Pass in the label and its distance to the input image. 
    // If return false, LBPH will terminate the search early.
    bool collect(int label, double distance);

    // Return the prediction result (the one with the smallest distance)
    int getLabel() const;

    // Return the distance of the result
    double getDistance() const;

private:
    int bestLabel;
    double minDistance;
};
