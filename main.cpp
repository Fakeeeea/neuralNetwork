#include <iostream>
#include <algorithm>
#include <cmath>

#include "NeuralNetwork.h"
#include "rendering/Renderer.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

void emLoop();

std::string title = "NeuralNetwork";
bool rendererInitialized = false;
Renderer renderer;

int main( int argc, char* argv[] ) {

#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop(emLoop, 0, 1);
#else

    while (!renderer.quit) {
        emLoop();
    }

    renderer.clean();
#endif
    return 0;
}

void emLoop () {
    if(!rendererInitialized)
        renderer.init(title), rendererInitialized = true;
    renderer.handleEvents();
    renderer.update();
    renderer.render();
}