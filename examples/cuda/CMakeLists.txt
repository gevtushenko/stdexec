cmake_minimum_required(VERSION 3.17.0)
project(CUDAExample CXX)

find_package(CUDAToolkit)

if(CUDAToolkit_FOUND)
    file(GLOB_RECURSE SCHED_SOURCES schedulers/*.hpp)
    add_library(CUDASchedulers INTERFACE ${SCHED_SOURCES})
    target_link_libraries(CUDASchedulers INTERFACE p2300 CUDA::cudart)
    target_include_directories(CUDASchedulers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(CUDASchedulers INTERFACE -x cuda -Wno-unknown-cuda-version)

    foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        target_compile_options(CUDASchedulers INTERFACE --cuda-gpu-arch=sm_${arch})
    endforeach()

    # Examples
    file(GLOB_RECURSE EXAMPLES
            RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
            CONFIGURE_DEPENDS
            examples/*.cpp)

    foreach(example ${EXAMPLES})
        file(TO_CMAKE_PATH "${example}" example_prefix)
        string(REPLACE "/" "." example_prefix "${example_prefix}")
        get_filename_component(example_name "${example_prefix}" NAME_WLE)
        message("--> ${example_name}")

        add_executable(${example_name} ${example})
        target_link_libraries(${example_name} PRIVATE CUDASchedulers)

        get_filename_component(PARENT_DIR ../ ABSOLUTE)
        target_include_directories(${example_name} PRIVATE ${PARENT_DIR})

        if (CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_options(${example_name} PRIVATE -O3 -DNDEBUG -mtune=native)
        endif()
    endforeach()
endif()
