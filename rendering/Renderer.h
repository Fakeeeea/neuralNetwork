#ifndef NEURAL_NETWORK_RENDERER_H
#define NEURAL_NETWORK_RENDERER_H

#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <SDL_opengles2.h>
#else
#include <SDL_opengl.h>
#endif
#include <SDL.h>
#include <string>
#include "NetworkHandler.h"
#include "Graph.h"
#include "../imgui/imgui.h"
#include "../imgui/imgui_impl_sdl2.h"
#include "../imgui/imgui_impl_opengl3.h"

class Renderer {
private:

#ifdef __EMSCRIPTEN__
    ImVec2 touchPanStart;
#endif

    struct Rect{
        ImVec2 pos;
        ImVec2 size;
    };

    enum graphType {
        trainAcc,
        trainCost,
        testAcc,
        testCost,
        valAcc,
        valCost,
    };

    float zoom = 1.0f;
    ImVec2 pan;

    SDL_Window *window{};
    SDL_GLContext gl_context{};

    const int nodeRadius = 20;
    const int graphHeight = 120;

    NetworkHandler networkHandler;

    bool regularize = false;
    float regularizationFactor = 3;
    int networkType = 0;
    int costType = 1;
    int miniBatchSize = 10;
    int epochNumber = 1000;
    int stepSize = 50;
    float learningRate = 0.5;
    std::vector<int> layerSizes = {2,3,2};

    std::vector<Graph> graphs;

    Rect headerViewport, settingsViewport, graphViewport, networkViewport;
    std::vector<std::vector<ImVec2>> nodeLocations;

    bool checkCursorInArea(Rect area);
    void checkPanning();
    void checkZooming();

    void drawBackground();

    void drawSettings();
    void drawNetworkTypeSelector();
    void drawLayerSizesSelector();
    void drawNetworkVariablesSelector();
    void drawNetworkControls();

    void drawHeader();
    void drawNetwork();
    void drawGraph();

    void updateSizes();
    void updateLocations();
    void updateData();
    ImVec2 transform(ImVec2 pos) const;

    void updateGraphs();
    void resetGraphs();

    void drawResults(int type);
    void drawDataPoints(int type);
    void updateAllDrawnDataPoints();
    void drawData(const std::vector<TrainingData>& inputs, const Eigen::MatrixXd& outputs, ImVec2 startingPos, ImVec2 endingPos, int type);

public:

    bool quit = false;

    void init(std::string& title);
    void handleEvents();
    void update();
    void render();
    void clean();

    Renderer();
};

#endif //NEURAL_NETWORK_RENDERER_H
