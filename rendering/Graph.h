#ifndef NEURAL_NETWORK_GRAPH_H
#define NEURAL_NETWORK_GRAPH_H

#include <vector>
#include <string>

class Graph {
private:
    int _currentSize = 0;
    int _maxSize;
    int _targetSize;
    float _ratio;
    std::vector<float> _data;
public:
    Graph(int maxSize, int targetSize);
    void addData(float val);
    std::pair<float*, int> getData();
    void resetGraph();
};


#endif //NEURAL_NETWORK_GRAPH_H
