//
// Created by aaa on 28/09/2025.
//

#include "Graph.h"

Graph::Graph(int maxSize, int targetSize) : _maxSize(maxSize), _targetSize(targetSize), _currentSize(0) {
    _data.resize(maxSize, 0);
    _ratio = (float) _maxSize / _targetSize;
}

void Graph::addData(float val) {
    if(_currentSize < _maxSize) {
        _data[_currentSize++] = val;
        return;
    }

    size_t writePos = 0;

    for(size_t i = 0; i < _targetSize; ++i) {
        size_t readPos = i * _ratio;
        _data[writePos++] = _data[readPos];
    }
    _currentSize = _targetSize;

    _data[_currentSize++] = val;
}

std::pair<float*, int> Graph::getData() {
    return {_data.data() , _currentSize};
}

void Graph::resetGraph() {
    std::fill(_data.begin(), _data.end(), 0.0);
}