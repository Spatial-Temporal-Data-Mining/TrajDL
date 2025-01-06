cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

execute_process(
    COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/trajdl_cpp/src/*.cpp")

pybind11_add_module(trajdl_cpp MODULE
    ${SRC_FILES}
    src/trajdl_cpp/pybind11_module.cpp
)
target_include_directories(trajdl_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/trajdl_cpp/include)

install(TARGETS trajdl_cpp DESTINATION ${SKBUILD_PROJECT_NAME})
