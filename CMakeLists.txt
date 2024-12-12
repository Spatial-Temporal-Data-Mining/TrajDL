cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/trajdl_cpp/src/*.cpp")

pybind11_add_module(trajdl_cpp MODULE
    ${SRC_FILES}
    src/trajdl_cpp/pybind11_module.cpp
)
target_include_directories(trajdl_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/trajdl_cpp/include)
install(TARGETS trajdl_cpp DESTINATION ${SKBUILD_PROJECT_NAME})
