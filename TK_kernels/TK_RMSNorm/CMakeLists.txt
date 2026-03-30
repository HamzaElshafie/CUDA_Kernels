cmake_minimum_required(VERSION 3.24)

project(TKRMSNorm LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 90a)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

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

if(EXISTS "${THUNDERKITTENS_ROOT}/kittens.cuh")
    set(TK_INCLUDE_DIR "${THUNDERKITTENS_ROOT}")
    get_filename_component(TK_REPO_ROOT "${THUNDERKITTENS_ROOT}" DIRECTORY)
elseif(EXISTS "${THUNDERKITTENS_ROOT}/include/kittens.cuh")
    set(TK_REPO_ROOT "${THUNDERKITTENS_ROOT}")
    set(TK_INCLUDE_DIR "${THUNDERKITTENS_ROOT}/include")
endif()

if(NOT EXISTS "${TK_INCLUDE_DIR}/kittens.cuh")
    message(FATAL_ERROR
        "Could not find ThunderKittens. Set THUNDERKITTENS_ROOT or place ThunderKittens under third_party/.")
endif()

set(TK_PROTOTYPE_DIR "${TK_REPO_ROOT}/prototype")

message(STATUS "Using ThunderKittens include dir: ${TK_INCLUDE_DIR}")

file(GLOB TK_CUDA_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

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
