cmake_minimum_required(VERSION 3.18)
project(l4rpz LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# auto-detect GPU architecture; override with -DCMAKE_CUDA_ARCHITECTURES=86
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(l4rpz
  src/main.cpp
  src/lz4_frame.cpp
  # src/decompress.cpp # TODO: add better way to switch between cpu and gpu implementation
  src/decompress.cu
)

target_include_directories(l4rpz PRIVATE src/include)

target_compile_options(l4rpz PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra --generate-line-info>
)
