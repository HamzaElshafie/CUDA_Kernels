cmake_minimum_required(VERSION 3.24)

project(TKKernels LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# H100 TK kernels that use WGMMA/TMA generally require SM90a.
set(CMAKE_CUDA_ARCHITECTURES 90a)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optional override for TK root. If unset, we auto-discover from common locations.
set(THUNDERKITTENS_ROOT "" CACHE PATH "Path to ThunderKittens repo root")

if(NOT THUNDERKITTENS_ROOT AND DEFINED ENV{THUNDERKITTENS_ROOT})
    set(THUNDERKITTENS_ROOT "$ENV{THUNDERKITTENS_ROOT}")
endif()

if(NOT THUNDERKITTENS_ROOT)
    set(TK_CANDIDATE_PATHS
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ThunderKittens"
        "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ThunderKittens"
        "${CMAKE_CURRENT_SOURCE_DIR}/ThunderKittens"
    )
    foreach(candidate ${TK_CANDIDATE_PATHS})
        if(EXISTS "${candidate}/kittens.cuh" OR EXISTS "${candidate}/include/kittens.cuh")
            set(THUNDERKITTENS_ROOT "${candidate}")
            break()
        endif()
    endforeach()
endif()

set(TK_INCLUDE_DIR "")
set(TK_REPO_ROOT "")

# Accept either:
#   - THUNDERKITTENS_ROOT=<repo root> (contains include/kittens.cuh)
#   - THUNDERKITTENS_ROOT=<include dir> (contains kittens.cuh)
if(EXISTS "${THUNDERKITTENS_ROOT}/kittens.cuh")
    set(TK_INCLUDE_DIR "${THUNDERKITTENS_ROOT}")
    get_filename_component(TK_REPO_ROOT "${THUNDERKITTENS_ROOT}" DIRECTORY)
elseif(EXISTS "${THUNDERKITTENS_ROOT}/include/kittens.cuh")
    set(TK_REPO_ROOT "${THUNDERKITTENS_ROOT}")
    set(TK_INCLUDE_DIR "${THUNDERKITTENS_ROOT}/include")
endif()

if(NOT EXISTS "${TK_INCLUDE_DIR}/kittens.cuh")
    message(FATAL_ERROR
        "Could not find ThunderKittens.\n"
        "Set THUNDERKITTENS_ROOT via one of:\n"
        "  1) CMake: cmake -S . -B build -DTHUNDERKITTENS_ROOT=/path/to/ThunderKittens\n"
        "  2) Env var: export THUNDERKITTENS_ROOT=/path/to/ThunderKittens\n"
        "  3) Repo-local folder: ../third_party/ThunderKittens (relative to TK_kernels)")
endif()

set(TK_PROTOTYPE_DIR "${TK_REPO_ROOT}/prototype")

message(STATUS "Using ThunderKittens include dir: ${TK_INCLUDE_DIR}")
if(EXISTS "${TK_PROTOTYPE_DIR}/prototype.cuh")
    message(STATUS "Using ThunderKittens prototype dir: ${TK_PROTOTYPE_DIR}")
endif()

file(GLOB TK_CUDA_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

if(NOT TK_CUDA_SOURCES)
    message(STATUS "No .cu files found in ${CMAKE_CURRENT_SOURCE_DIR}. Add a kernel file to build.")
endif()

foreach(src ${TK_CUDA_SOURCES})
    get_filename_component(target_name "${src}" NAME_WE)
    add_executable(${target_name} "${src}")
    target_compile_definitions(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:KITTENS_HOPPER>
    )
    target_include_directories(${target_name} PRIVATE "${TK_INCLUDE_DIR}")
    if(EXISTS "${TK_PROTOTYPE_DIR}/prototype.cuh")
        target_include_directories(${target_name} PRIVATE "${TK_PROTOTYPE_DIR}")
    endif()
    target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
    )
    target_link_libraries(${target_name} PRIVATE
        CUDA::cublas
        CUDA::cuda_driver
    )
endforeach()