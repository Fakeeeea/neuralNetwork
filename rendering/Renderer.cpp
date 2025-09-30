#include "Renderer.h"
#include <iostream>
#include <iomanip>
#include <sstream>

#if __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>
#endif

Renderer::Renderer() = default;

void Renderer::init(std::string& title) {
    if(SDL_Init(SDL_INIT_EVERYTHING)) {
        std::cout<<"SDL Initialized\n";
    }

    const char* glsl_version = "#version 300 es";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);

    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

    float main_scale = ImGui_ImplSDL2_GetContentScaleForDisplay(0);
    window = SDL_CreateWindow(title.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1920 * main_scale, 1080 * main_scale,
                               SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_SHOWN );

    if(!window) {
        std::cout<<"Error"<<SDL_GetError()<<"\n";
    }

    gl_context = SDL_GL_CreateContext(window);

    if(gl_context == nullptr) {
        std::cout<<"Error"<<SDL_GetError()<<"\n";
    }

    if(SDL_GL_MakeCurrent(window, gl_context) != 0) {
        std::cout << "MakeCurrent Error: " << SDL_GetError() << "\n";
        return;
    }

    SDL_GL_SetSwapInterval(0);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;

    ImGui::StyleColorsDark();
    if(!ImGui_ImplSDL2_InitForOpenGL(window, gl_context)) {
        std::cout << "ImGui SDL2 Init Failed\n";
        return;
    }
    if(!ImGui_ImplOpenGL3_Init(glsl_version)) {
        std::cout << "ImGui OpenGL3 Init Failed\n";
        return;
    }

    quit = false;

    constexpr int graphTypes = 6;
    for(int i = 0; i < graphTypes; ++i) {
        graphs.emplace_back(100,90);
    }

    networkHandler = NetworkHandler(layerSizes);
    networkHandler.loadData(NetworkHandler::XOR);
    networkHandler.updateTrainingSessionSettings(miniBatchSize, epochNumber);
}

void Renderer::handleEvents() {
    SDL_Event e;
    while(SDL_PollEvent(&e)) {
        ImGui_ImplSDL2_ProcessEvent(&e);
        switch(e.type) {
            case SDL_QUIT:
                quit = true;
                break;
            case SDL_WINDOWEVENT:
                switch(e.window.event) {
                    case SDL_WINDOWEVENT_SHOWN:
                    case SDL_WINDOWEVENT_RESIZED:
                        updateSizes();
                        updateLocations();
                        break;
                } break;
#ifdef __EMSCRIPTEN__
            case SDL_FINGERDOWN:
                touchPanStart.x = e.tfinger.x;
                touchPanStart.y = e.tfinger.y;
                break;
            case SDL_FINGERMOTION: {

                int drawableW, drawableH;
                SDL_GL_GetDrawableSize(window, &drawableW, &drawableH);

                float fingerx = e.tfinger.x * drawableW;
                float fingery = e.tfinger.y * drawableH;

                if(fingerx > networkViewport.pos.x && fingerx < networkViewport.size.x && fingery > networkViewport.pos.y && fingery < networkViewport.size.y) {
                    pan.x -= (touchPanStart.x * drawableW - fingerx) * 0.5f;
                    pan.y -= (touchPanStart.y * drawableH - fingery) * 0.5f;
                }
                break;
            }
#endif
        }
    }
}

void Renderer::render() {

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();

    if(checkCursorInArea(networkViewport)) {
        checkPanning();
        checkZooming();
    }

    drawBackground();
    drawHeader();
    drawSettings();
    drawGraph();

    if(networkHandler.isNetworkCreated())
        drawNetwork();

    ImGui::Render();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    SDL_GL_SwapWindow(window);
}

void Renderer::update() {

    for(int i = 0; i < stepSize && networkHandler.status._training; ++i) {
        bool shouldUpdateGraphs = false;
        networkHandler.update(shouldUpdateGraphs);
        if(shouldUpdateGraphs) {
            updateGraphs();
        }
    }
}

void Renderer::clean() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

bool Renderer::checkCursorInArea(Rect area) {
    ImVec2 cursorPos = ImGui::GetMousePos();

    if(cursorPos.x < 0 || cursorPos.y < 0) return false;

    if(cursorPos.x > area.pos.x && cursorPos.x < (area.pos.x + area.size.x) &&
       cursorPos.y > area.pos.y && cursorPos.y < (area.pos.y + area.size.y))
        return true;
    return false;
}

void Renderer::checkZooming() {
    ImGuiIO& io = ImGui::GetIO();

    constexpr float step = 0.1f;
    constexpr float minZoom = 0.01f;
    constexpr float maxZoom = 5.0f;

    if(io.MouseWheel != 0.0f)
        zoom = std::clamp(zoom + io.MouseWheel * step, minZoom, maxZoom);
}

void Renderer::checkPanning() {
    ImGuiIO& io = ImGui::GetIO();

    constexpr int panningMultiplier = 2;

    if(ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
        ImVec2 delta = io.MouseDelta;
        pan.x += delta.x * panningMultiplier;
        pan.y += delta.y * panningMultiplier;
    }
}

void Renderer::drawBackground() {
    ImGuiIO& io = ImGui::GetIO();
    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    drawList->AddRectFilled({0,0}, {io.DisplaySize.x, io.DisplaySize.y}, IM_COL32(30,30,46,255));
}

void Renderer::drawHeader() {
    ImDrawList* drawList = ImGui::GetForegroundDrawList();

    drawList->AddRectFilled({headerViewport.pos.x, headerViewport.pos.y},
                            {headerViewport.pos.x + headerViewport.size.x, headerViewport.pos.y + headerViewport.size.y},
                            IM_COL32(43, 43, 60, 255));

    float fontSize = headerViewport.size.y - headerViewport.size.y * 0.2f;
    float shadowOffset = fontSize * 0.05f;

    ImVec2 pos((0+16), (headerViewport.size.y * 0.2f) / 2);

    drawList->AddText(
            nullptr,
            fontSize,
            ImVec2(pos.x + shadowOffset, pos.y+shadowOffset),
            IM_COL32(0,0,0,180),
            "Neural Network Visualization"
    );
    drawList->AddText(
            nullptr,
            fontSize,
            ImVec2(pos.x, pos.y),
            IM_COL32(255,200,50,255),
            "Neural Network Visualization"
    );
}

void Renderer::drawSettings() {
    ImGui::SetNextWindowPos(ImVec2(settingsViewport.pos.x, settingsViewport.pos.y));
    ImGui::SetNextWindowSize(ImVec2(settingsViewport.size.x, settingsViewport.size.y));

    ImGuiWindowFlags window_flags = 0
                                    | ImGuiWindowFlags_NoTitleBar
                                    | ImGuiWindowFlags_NoResize
                                    | ImGuiWindowFlags_NoMove
                                    | ImGuiWindowFlags_NoScrollbar
                                    | ImGuiWindowFlags_NoSavedSettings
    ;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
    ImGui::Begin("Settings", nullptr, window_flags);
    ImGui::PopStyleVar();

    ImGui::Text("Network Settings");
    ImGui::Separator();

    drawNetworkTypeSelector();
    ImGui::Separator(), ImGui::Separator();
    drawLayerSizesSelector();
    ImGui::Separator(), ImGui::Separator();
    drawNetworkVariablesSelector();
    ImGui::Separator(), ImGui::Separator();
    drawNetworkControls();

    ImGui::End();
}

void Renderer::drawNetworkTypeSelector() {
    static const char* networkTypes[]{"XOR", "SPIRAL", "LINEAR", "CIRCLE", "MOONS", "CUBE", "PYRAMID"};
    //static const char* networkTypes[]{"MNIST Images", "XOR", "SPIRAL", "LINEAR", "CIRCLE", "MOONS", "CUBE", "PYRAMID"};
    if(ImGui::Combo("Network type", &networkType, networkTypes, IM_ARRAYSIZE(networkTypes))) {
        switch(networkType) {
            /*case NetworkHandler::MNIST: layerSizes = {784, 30, 10};
                                        networkHandler.loadData("train-images.idx3-ubyte", "train-labels.idx1-ubyte", "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 0.2, true);
                                        break;*/
            case NetworkHandler::XOR:
                layerSizes = {2, 3, 2};
                networkHandler.loadData(NetworkHandler::XOR);
                break;
            case NetworkHandler::SPIRAL:
                layerSizes = {2, 4, 4, 2};
                networkHandler.loadData(NetworkHandler::SPIRAL);
                break;
            case NetworkHandler::LINEAR:
                layerSizes = {2, 3, 2};
                networkHandler.loadData(NetworkHandler::LINEAR);
                break;
            case NetworkHandler::CIRCLE:
                layerSizes = {2, 4, 2, 2};
                networkHandler.loadData(NetworkHandler::CIRCLE);
                break;
            case NetworkHandler::MOONS:
                layerSizes = {2, 3, 2};
                networkHandler.loadData(NetworkHandler::MOONS);
                break;
            case NetworkHandler::CUBE:
                layerSizes = {3, 8, 16, 12};
                networkHandler.loadData(NetworkHandler::CUBE);
                break;
            case NetworkHandler::PYRAMID:
                layerSizes = {3, 5, 10, 8};
                networkHandler.loadData(NetworkHandler::PYRAMID);
                break;
        }

        networkHandler.createNetwork(layerSizes, costType, regularizationFactor, learningRate, regularize);
        networkHandler.updateTrainingSessionSettings(miniBatchSize, epochNumber);

        updateLocations();
    }
}

void Renderer::drawLayerSizesSelector() {
    ImGui::Text("Layer sizes");
    bool updateNetwork = false;
    int layersCount = layerSizes.size();
    ImGui::TextColored(ImVec4(0,1,0,1),("Input Nodes: " + std::to_string(layerSizes.front())).c_str());
    for(int i = 1; i < layersCount-1; ++i) {

        std::string text = "Hidden Layer " + std::to_string(i);

        if(ImGui::InputInt(text.c_str(), &layerSizes[i]))
            updateNetwork = true;

        layerSizes[i] = std::max(0, std::min(1000, layerSizes[i]));

        if(layerSizes[i] == 0)
            layerSizes.erase(layerSizes.begin() + i);

    }
    if(ImGui::Button("Add hidden layer")) {
        layerSizes.insert(layerSizes.end() - 1, 1);
        updateNetwork = true;
    }

    ImGui::TextColored(ImVec4(1,0,0,1),("Output Nodes: " + std::to_string(layerSizes.back())).c_str());

    if(updateNetwork) {
        networkHandler.createNetwork(layerSizes, costType, regularizationFactor, learningRate, regularize);
        networkHandler.updateTrainingSessionSettings(miniBatchSize, epochNumber);

        updateLocations();
    }
}

void Renderer::drawNetworkVariablesSelector() {
    bool change = false;
    bool changeTraining = false;

    ImGui::TextColored(ImVec4(0.31, 0.98, 0.48, 1), "Network variables");

    if(ImGui::Checkbox("L2 Regularize", &regularize)) change = true;
    if(regularize)
        if(ImGui::InputFloat("Regularization rate", &regularizationFactor)) change = true;
    ImGui::Separator();
    static const char* costTypes[]{"Quadratic","CrossEntropy"};
    if(ImGui::Combo("Cost type", &costType, costTypes, IM_ARRAYSIZE(costTypes))) change = true;
    ImGui::Separator();
    if(ImGui::InputInt("Mini batch size", &miniBatchSize)) changeTraining = true;
    ImGui::Separator();
    if(ImGui::InputInt("Epoch number", &epochNumber)) changeTraining = true;
    ImGui::Separator();
    if(ImGui::InputFloat("Learning rate", &learningRate)) change = true;
    ImGui::Separator();
    if(ImGui::InputInt("Minibatches per frame", &stepSize)) stepSize = std::min(100000, std::max(stepSize, 1));

    if(change)
        networkHandler.updateSettings(costType, regularizationFactor, learningRate, regularize);
    if(changeTraining)
        networkHandler.updateTrainingSessionSettings(miniBatchSize, epochNumber);
}

void Renderer::drawNetworkControls() {
    if(ImGui::Button("Toggle Train", ImVec2(-FLT_MIN, 0))) {
        networkHandler.toggleTraining();
    }

    if(ImGui::Button("Reset graphs", ImVec2(-FLT_MIN, 0))) {
        resetGraphs();
    }

    if(ImGui::Button("Return to network", ImVec2(-FLT_MIN, 0))) {
        pan = ImVec2(networkViewport.size.x * 0.3, networkViewport.size.y * 0.5);
    }
}

void Renderer::drawResults(int type) {

    ImDrawList *drawList = ImGui::GetWindowDrawList();
    ImVec2 cursorPos = ImGui::GetCursorScreenPos();
    const int padding = 8;
    const float paddedSize = graphViewport.size.x;
    const float unpaddedSize = graphViewport.size.x - 2 * padding;

    ImVec2 startingPos(graphViewport.pos.x + padding,
                       cursorPos.y + padding);

    ImVec2 endingPos = ImVec2(startingPos.x+unpaddedSize,
                              startingPos.y+unpaddedSize);

    ImGui::Dummy(ImVec2(paddedSize, paddedSize));

    drawList->AddRectFilled(startingPos, endingPos, IM_COL32(100,100,100,255));

    ImVec2 startX, endX;
    ImVec2 startY, endY;

    startX = ImVec2(startingPos.x, startingPos.y + unpaddedSize/2);
    endX = ImVec2(endingPos.x, startingPos.y + unpaddedSize/2);
    startY = ImVec2(startingPos.x + unpaddedSize/2, startingPos.y);
    endY = ImVec2(startingPos.x + unpaddedSize/2, endingPos.y);

    auto drawLine = [&](ImVec2 start, ImVec2 end) {
        drawList->AddLine(start, end, IM_COL32(255,255,255,255));
        drawList->AddText(start, IM_COL32(255,0,0,255),"0.00");
    };

    drawLine(startX, endX), drawLine(startY, endY);

    switch(type) {
        case NetworkHandler::DataType::TRAIN:
            drawData(networkHandler.session.trainData, networkHandler.trainOutputs, startingPos, endingPos,
                     NetworkHandler::DataType::TRAIN); break;
        case NetworkHandler::DataType::TEST:
            drawData(networkHandler.session.testData, networkHandler.testOutputs, startingPos, endingPos,
                     NetworkHandler::DataType::TEST); break;
        case NetworkHandler::DataType::VAL:
            drawData(networkHandler.session.validationData, networkHandler.validationOutputs, startingPos, endingPos,
                     NetworkHandler::DataType::VAL); break;
    }
}

void Renderer::drawData(const std::vector<TrainingData>& inputs, const Eigen::MatrixXd& outputs, ImVec2 startingPos, ImVec2 endingPos, int type) {
    if(outputs.size() == 0)
        return;

    int numData = inputs.size();

    ImDrawList* drawList = ImGui::GetWindowDrawList();

    float size = endingPos.x - startingPos.x;

    auto traslate = [&](ImVec2 pointPos) {
        return ImVec2(startingPos.x + (pointPos.x * size),
                      startingPos.y + (pointPos.y * size));
    };

    for(int i = 0; i < numData; ++i) {

        ImVec2 point(inputs[i].input[0], inputs[i].input[1]);

        int expected, actual;
        const int labelsCount = outputs.col(i).size();

        inputs[i].target.maxCoeff(&expected);
        outputs.col(i).maxCoeff(&actual);

        float hueActual = (actual % labelsCount) / (float) labelsCount;
        ImU32 actualColor = ImGui::ColorConvertFloat4ToU32(ImColor::HSV(hueActual, 0.8f, 0.9f));

        drawList->AddCircleFilled(traslate(point), 4, actualColor);
    }
}

void Renderer::drawGraph() {
    ImGui::SetNextWindowPos(ImVec2(graphViewport.pos.x, graphViewport.pos.y));
    ImGui::SetNextWindowSize(ImVec2(graphViewport.size.x, graphViewport.size.y));

    ImGuiWindowFlags window_flags = 0
                                    | ImGuiWindowFlags_NoTitleBar
                                    | ImGuiWindowFlags_NoResize
                                    | ImGuiWindowFlags_NoMove
                                    | ImGuiWindowFlags_NoScrollbar
                                    | ImGuiWindowFlags_NoSavedSettings
    ;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
    ImGui::Begin("Graphs", nullptr, window_flags);
    ImGui::PopStyleVar();

    ImGui::Text("Training Metrics");
    ImGui::Separator();

    const ImVec2 graphSize = ImVec2(-1, graphHeight);

    const auto createPlot = [&](const char* title, const std::pair<float*, int> accuracy, const std::pair<float*, int> cost, const ImVec4& color) {
        ImGui::PushID(title);
        ImGui::Text("%s",title);
        ImGui::PushStyleColor(ImGuiCol_PlotLines, color);

        ImGui::Text("Accuracy");
        ImGui::PlotLines("##Accuracy",
                         accuracy.first, accuracy.second,
                         0, nullptr, 0.0f, 100.0f, graphSize);

        ImGui::Text("Cost");
        ImGui::PlotLines("##Cost",
                         cost.first, cost.second,
                         0, nullptr, FLT_MIN, FLT_MAX, graphSize);

        ImGui::PopStyleColor();
        ImGui::Separator();
        ImGui::PopID();
    };

    ImVec4 testColor = ImVec4(0.9f, 0.2f, 0.2f, 1.0f);
    ImVec4 trainColor = ImVec4(0.2f, 0.6f, 0.9f, 1.0f);
    ImVec4 validationColor = ImVec4(0.2f, 0.8f, 0.4f, 1.0f);

    createPlot("Train Data", graphs[trainAcc].getData(), graphs[trainCost].getData(), trainColor);
    drawResults(NetworkHandler::DataType::TRAIN);
    createPlot("Test Data", graphs[testAcc].getData(), graphs[testCost].getData(), testColor);
    drawResults(NetworkHandler::DataType::TEST);
    createPlot("Validation Data", graphs[valAcc].getData(), graphs[valCost].getData(), validationColor);
    drawResults(NetworkHandler::DataType::VAL);

    ImGui::End();
}

void Renderer::drawNetwork() {
    NeuralNetwork& nn = networkHandler.nn;
    ImDrawList* drawList = ImGui::GetBackgroundDrawList();

    int transformedRadius = nodeRadius * zoom;
    int numLayers = nodeLocations.size();


    for(int i = 1; i < numLayers; ++i) {
        int layerNodes = nodeLocations[i].size();
        int prevLayerNodes = nodeLocations[i-1].size();

        for(int j = 0; j < layerNodes; ++j) {

            ImVec2 weightEnd = transform(nodeLocations[i][j]);
            weightEnd.x -= transformedRadius;

            for(int k = 0; k < prevLayerNodes; ++k) {

                ImVec2 weightStart = transform(nodeLocations[i-1][k]);
                weightStart.x += transformedRadius;

                ImU32 color = IM_COL32(100,100,100,100);
                double weight = nn.weights[i-1](j,k);

                color = IM_COL32(((weight < 0) ? 255 * -weight : 0), ((weight >= 0.0f) ? 255 * weight : 0), 0, 255);
                drawList->AddLine(weightStart, weightEnd, color, std::min(3.0, 1.0 * abs(weight)));
            }
        }
    }

    for(int i = 0; i < numLayers; ++i) {
        int numNodes = nodeLocations[i].size();

        for(int j = 0; j < numNodes; ++j) {

            ImVec2 pos = transform(nodeLocations[i][j]);

            drawList->AddCircle(pos, transformedRadius, IM_COL32(255,184,108,255));

            if(i > 0) {
                double biasValue = nn.biases[i-1](j);

                std::ostringstream oss;
                oss << std::fixed << std::setprecision(2) << biasValue;

                std::string biasText = oss.str();

                ImVec2 textSize = ImGui::CalcTextSize(biasText.c_str());
                ImVec2 textPos = ImVec2(pos.x - textSize.x * 0.5f, pos.y - textSize.y * 0.5f);
                drawList->AddText(textPos, IM_COL32(255, 184, 108, 255), biasText.c_str());
            }
        }
    }

    if(networkHandler.isNetworkCreated() && networkHandler.status._training) {
        std::string progressMinibatch = std::to_string(networkHandler.session.minibatch) + "/" + std::to_string(networkHandler.session.miniBatches.size());
        std::string progressEpoch = std::to_string(networkHandler.session.epoch) + "/" + std::to_string(epochNumber);
        ImVec2 minibatchesPos = ImVec2(networkViewport.pos.x, networkViewport.pos.y);
        ImVec2 epochPos = ImVec2(networkViewport.pos.x + ImGui::CalcTextSize(progressMinibatch.c_str()).x + ImGui::GetFontSize() * 2, networkViewport.pos.y);
        drawList->AddText(minibatchesPos, IM_COL32(255, 184, 108, 255), progressMinibatch.c_str());
        drawList->AddText(epochPos, IM_COL32(255, 184, 108, 255), progressEpoch.c_str());
    }
}

void Renderer::updateLocations() {
    const int numLayers = layerSizes.size();
    const int layerDistance = nodeRadius * 25;
    const int nodesDistance = nodeRadius * 2.5;
    const int layerOffset = (networkViewport.size.x / 2) - (numLayers / 2) * layerDistance;

    nodeLocations.resize(numLayers);

    for(int i = 0; i < numLayers; ++i) {
        int layerX = layerOffset + i * layerDistance;
        const int numNodes = layerSizes[i];

        nodeLocations[i].resize(numNodes);

        //int nodesOffset = networkViewport.pos.y + (networkViewport.size.y - (nodesDistance * (numNodes - 1))) / 2;
        int nodesOffset = -(nodesDistance * numNodes) * 0.5f;

        for (int j = 0; j < numNodes; ++j) {
            int nodeY = nodesOffset + j * nodesDistance;
            nodeLocations[i][j] = ImVec2(layerX, nodeY);
        }
    }
}

void Renderer::updateSizes() {

#ifdef __EMSCRIPTEN__
    double css_w = (double)EM_ASM_INT({ return window.innerWidth; });
    double css_h = (double)EM_ASM_INT({ return window.innerHeight; });
    double dpr = emscripten_get_device_pixel_ratio();

    int fb_w = (int)(css_w * dpr);
    int fb_h = (int)(css_h * dpr);
    fb_w = std::max(1920, fb_w);
    fb_h = std::max(1080, fb_h);

    emscripten_set_canvas_element_size("#canvas", fb_w, fb_h);
    emscripten_set_element_css_size("#canvas", fb_w / dpr, fb_h / dpr);
    SDL_SetWindowSize(window, fb_w, fb_h);

    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2((float)fb_w, (float)fb_h);
#endif

    int x,y;
    SDL_GetWindowSize(window, &x, &y);
    ImVec2 screenSize((float)x,(float)y);

    //%
    constexpr float headerSize = 0.05f;
    constexpr float sidePanelsSize = 0.2f;

    headerViewport = {ImVec2(0, 0),
                      ImVec2(screenSize.x, screenSize.y * headerSize)};

    settingsViewport = {ImVec2(0, headerViewport.size.y),
                        ImVec2(screenSize.x* sidePanelsSize, screenSize.y - headerViewport.size.y)};

    networkViewport = {ImVec2(settingsViewport.size.x, headerViewport.size.y),
                       ImVec2(screenSize.x - (screenSize.x * sidePanelsSize)*2, screenSize.y -  headerViewport.size.y)};

    graphViewport = {ImVec2(networkViewport.size.x + networkViewport.pos.x, headerViewport.size.y),
                     ImVec2(screenSize.x * sidePanelsSize, screenSize.y - headerViewport.size.y)};

    pan = ImVec2(networkViewport.size.x * 0.3, networkViewport.size.y * 0.5);
}

void Renderer::updateGraphs() {
    networkHandler.updateOutputs();

    graphs[trainAcc].addData(networkHandler.getDatasetAccuracy(NetworkHandler::DataType::TRAIN));
    graphs[trainCost].addData(networkHandler.getDatasetCost(NetworkHandler::DataType::TRAIN));
    graphs[testAcc].addData(networkHandler.getDatasetAccuracy(NetworkHandler::DataType::TEST));
    graphs[testCost].addData(networkHandler.getDatasetCost(NetworkHandler::DataType::TEST));
    graphs[valAcc].addData(networkHandler.getDatasetAccuracy(NetworkHandler::DataType::VAL));
    graphs[valCost].addData(networkHandler.getDatasetCost(NetworkHandler::DataType::VAL));

}

void Renderer::resetGraphs() {
   for(auto & graph : graphs) {
       graph.resetGraph();
   }
}

ImVec2 Renderer::transform(ImVec2 pos) const {
    return {(pos.x + pan.x) * zoom, (pos.y + pan.y) * zoom};
}
