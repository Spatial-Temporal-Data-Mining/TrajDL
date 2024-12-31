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

# # 测试用，可执行文件
# # 创建库，包含共享的代码
# add_library(base STATIC
#     src/trajdl_cpp/src/distance/base.cpp
# )
# target_include_directories(base PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/trajdl_cpp/include)

# add_executable(example_exe
#     src/trajdl_cpp/src/distance/frechet.cpp
# )
# target_link_libraries(example_exe PRIVATE
#     base
#     pybind11::embed
#     Python::Python
# )
# target_include_directories(example_exe PUBLIC
#     ${CMAKE_CURRENT_SOURCE_DIR}/src/trajdl_cpp/include
#     /home/mengxian/Downloads/github/TrajDL/.venv/lib/python3.12/site-packages/pybind11/include/
#     /home/mengxian/.local/share/uv/python/cpython-3.12.8-linux-x86_64-gnu/include/python3.12/
# )
# # 设置调试模式
# set(CMAKE_BUILD_TYPE Debug)

# # 添加调试标志
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
### for test

install(TARGETS trajdl_cpp DESTINATION ${SKBUILD_PROJECT_NAME})
