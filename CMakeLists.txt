cmake_minimum_required(VERSION 3.28)
project(ppo_cpp VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.4/bin/nvcc")
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install)

set(CAFFE2_USE_CUDNN ON) # Tell pytorch to use cudnn
set(CAFFE2_USE_CUSPARSELT ON) # Tell pytorch to use CUSPARSELT, You need to install it or turn off.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(APP_SOURCES
        src/carla/ac_ppo_carla.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/carla/ppo_carla_inference.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/ppo_continuous_action.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/ac_ppo_continuous_action.cpp
)

foreach(source_file  ${APP_SOURCES})

get_filename_component( source_name ${source_file} NAME_WE )
add_executable(${source_name}
        ${source_file}
        src/distributed.cpp)

target_include_directories (${source_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories (${source_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/libs/mujoco_3_2_0/include)
target_include_directories (${source_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/libs/glfw_3_4/include)
target_include_directories(${source_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/libs/ThreadPool)
target_include_directories (${source_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/libs/mujoco_3_2_0/lib)
target_include_directories(${source_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/libs)
target_include_directories(${source_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)


include(FindProtobuf)
find_package(Protobuf REQUIRED)
target_link_libraries(${source_name} PUBLIC ${Protobuf})

target_include_directories (${source_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/libs/tensorboard_logger/include)

find_library(TENSORBOARD tensorboard_logger REQUIRED)
target_link_libraries(${source_name} LINK_PUBLIC ${TENSORBOARD})

find_package(Torch PATHS /libtorch/share/cmake/Torch ${CMAKE_CURRENT_SOURCE_DIR}/libs/libtorch_2_4_1/share/cmake/Torch REQUIRED)

find_package(MPI REQUIRED)
target_include_directories(${source_name} PUBLIC ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(${source_name} PRIVATE MPI::MPI_CXX)

target_link_libraries(${source_name} PRIVATE json_spirit)

find_package(OpenCV REQUIRED)
target_link_libraries(${source_name} PRIVATE ${OpenCV_LIBS})

find_package(CUDAToolkit REQUIRED)
find_path(NCCL_INCLUDE_DIR REQUIRED NAMES nccl.h)
find_library(NCCL_LIBRARY REQUIRED NAMES nccl)
target_link_libraries(${source_name} PRIVATE ${NCCL_LIBRARY})
set(NCCL_DEBUG=OFF)
target_include_directories(${source_name} PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${NCCL_INCLUDE_DIR})
target_link_libraries(${source_name} PRIVATE CUDA::cudart)


find_library(MUJOCOLIB mujoco PATHS ${CMAKE_CURRENT_SOURCE_DIR}/libs/mujoco_3_2_0/lib REQUIRED)
target_link_libraries(${source_name} LINK_PUBLIC ${MUJOCOLIB})


find_library(GLFW glfw3 PATHS ${CMAKE_CURRENT_SOURCE_DIR}/libs/glfw_3_4/lib-mingw-w64 REQUIRED)

target_link_libraries(${source_name} LINK_PUBLIC ${GLFW})

target_link_libraries(${source_name} LINK_PUBLIC "${TORCH_LIBRARIES}")

find_package(cppzmq REQUIRED)
target_link_libraries(${source_name} LINK_PUBLIC cppzmq)

install(TARGETS ${source_name} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install/bin)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/libs/gymcpp DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install/libs)
# To install, for example, MSVC runtime libraries:
include(InstallRequiredSystemLibraries)
# To install other/non-system 3rd party required libraries:
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/FixBundle.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/FixBundle.cmake
        @ONLY
)


install(SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/FixBundle.cmake)

endforeach()