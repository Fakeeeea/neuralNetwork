cmake_minimum_required(VERSION 3.27)
project(neural_network)

set(CMAKE_CXX_STANDARD 17)

set(SDL2_INCLUDE_DIR C:/libs/SDL/include/SDL2)
set(SDL2_LIB_DIR C:/libs/SDL/lib)

include_directories(${SDL2_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/Eigen)
include_directories(${CMAKE_SOURCE_DIR}/imgui)
link_directories(${SDL2_LIB_DIR})

add_executable(neural_network main.cpp
        NeuralNetwork.h
        NeuralNetwork.cpp
        TrainingData.h
        TrainingData.cpp
        Cost.h
        Cost.cpp
        ActivationFunction.h
        ActivationFunction.cpp
        rendering/Renderer.h
        rendering/Renderer.cpp
        rendering/NetworkHandler.h
        rendering/NetworkHandler.cpp
        rendering/Graph.h
        rendering/Graph.cpp

        #imgui files

        imgui/imconfig.h
        imgui/imgui.cpp
        imgui/imgui.h
        imgui/imgui_demo.cpp
        imgui/imgui_draw.cpp
        imgui/imgui_impl_sdl2.h
        imgui/imgui_impl_sdl2.cpp
        imgui/imgui_impl_opengl3.cpp
        imgui/imgui_impl_opengl3.h
        imgui/imgui_impl_opengl3_loader.h
        imgui/imgui_internal.h
        imgui/imgui_stdlib.h
        imgui/imgui_stdlib.cpp
        imgui/imgui_tables.cpp
        imgui/imgui_widgets.cpp
        imgui/imstb_rectpack.h
        imgui/imstb_truetype.h
        imgui/imstb_textedit.h
        )

target_link_libraries(neural_network opengl32 mingw32 SDL2main SDL2)