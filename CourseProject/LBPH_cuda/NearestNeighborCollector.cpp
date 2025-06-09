#include "NearestNeighborCollector.hpp"

NearestNeighborCollector::NearestNeighborCollector()
    : bestLabel(-1), minDistance(std::numeric_limits<double>::max()) {}

void NearestNeighborCollector::init(int size) {
    bestLabel = -1;
    minDistance = std::numeric_limits<double>::max();
}

bool NearestNeighborCollector::collect(int label, double distance) {
    if (distance < minDistance) {
        minDistance = distance;
        bestLabel = label;
    }
    return true; // always collect all candidates
}

int NearestNeighborCollector::getLabel() const {
    return bestLabel;
}

double NearestNeighborCollector::getDistance() const {
    return minDistance;
}
