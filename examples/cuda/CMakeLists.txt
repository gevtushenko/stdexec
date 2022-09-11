cmake_minimum_required(VERSION 3.17.0)
project(CUDAExample CUDA)

find_package(CUDAToolkit)

if(CUDAToolkit_FOUND)
    file(GLOB_RECURSE SCHED_SOURCES schedulers/*.cuh)
    add_library(CUDASchedulers ${SCHED_SOURCES} examples/_clangd_helper_file.cu schedulers/detail/_clangd_helper_file.cu)
    target_link_libraries(CUDASchedulers PUBLIC p2300 p2300_executable_flags CUDA::cudart)
    target_include_directories(CUDASchedulers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_features(CUDASchedulers PUBLIC cxx_std_20)

    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      target_compile_options(CUDASchedulers PUBLIC -Wno-unknown-cuda-version)
      foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        target_compile_options(CUDASchedulers PUBLIC --cuda-gpu-arch=sm_${arch})
      endforeach()
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "NVCXX")
      foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        target_compile_options(CUDASchedulers PUBLIC -gpu=cc${arch})
      endforeach()
    endif()

    # Tests
    file(GLOB_RECURSE TEST_SOURCES
            RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
            CONFIGURE_DEPENDS
            test/*.cu)

    add_executable(test.CUDA ${TEST_SOURCES})

    target_include_directories(test.CUDA PRIVATE test)
    target_link_libraries(test.CUDA CUDASchedulers Catch2::Catch2)
    catch_discover_tests(test.CUDA)

    # Examples
    file(GLOB_RECURSE EXAMPLES
            RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
            CONFIGURE_DEPENDS
            examples/*.cu)

    foreach(example ${EXAMPLES})
        file(TO_CMAKE_PATH "${example}" example_prefix)
        string(REPLACE "/" "." example_prefix "${example_prefix}")
        get_filename_component(example_name "${example_prefix}" NAME_WLE)

        add_executable(${example_name} ${example})
        target_link_libraries(${example_name} PRIVATE CUDASchedulers)

        get_filename_component(PARENT_DIR ../ ABSOLUTE)
        target_include_directories(${example_name} PRIVATE ${PARENT_DIR})
    endforeach()
endif()
