cmake_minimum_required(VERSION 3.22)

# 关键：让 galp 自己成为一个可配置的顶层工程（同时也兼容被 add_subdirectory）
project(GALP LANGUAGES CXX CUDA)

# 是否顶层构建（Standalone）
set(GALP_STANDALONE OFF)
if (CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(GALP_STANDALONE ON)
endif()

# ────────────────────────────────────────────────
# 0. Standalone 时补齐全局设置（被父工程包含时不要乱改父工程）
# ────────────────────────────────────────────────
if (GALP_STANDALONE)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    option(GALP_FORCE_CLANG "Force clang/clang++ when building GALP standalone" OFF)
    if (GALP_FORCE_CLANG)
        find_program(CLANG_CXX NAMES clang++ REQUIRED)
        find_program(CLANG_C NAMES clang REQUIRED)
        set(CMAKE_C_COMPILER "${CLANG_C}" CACHE STRING "C Compiler" FORCE)
        set(CMAKE_CXX_COMPILER "${CLANG_CXX}" CACHE STRING "C++ Compiler" FORCE)
    endif()
endif()

# ────────────────────────────────────────────────
# 0.1 CUDA settings（尽量别用 CMAKE_CUDA_FLAGS，全局污染；但先保留你现有逻辑）
# ────────────────────────────────────────────────
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr") # todo

# ────────────────────────────────────────────────
# 1. Dependencies
# ────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)
find_package(nvcomp CONFIG REQUIRED)

get_target_property(NVCOMP_VERSION nvcomp::nvcomp VERSION)
get_target_property(NVCOMP_INC_DIRS nvcomp::nvcomp INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(NVCOMP_LIBS nvcomp::nvcomp INTERFACE_LINK_LIBRARIES)

message(STATUS "nvCOMP version ..........: ${NVCOMP_VERSION}")
message(STATUS "nvCOMP include dirs .....: ${NVCOMP_INC_DIRS}")
message(STATUS "nvCOMP libs .............: ${NVCOMP_LIBS}")

# ────────────────────────────────────────────────
# 2. Thrust / CCCL
# ────────────────────────────────────────────────
find_path(THRUST_INCLUDE_DIR
    NAMES thrust/version.h thrust/device_uvector.h
    HINTS ${CUDAToolkit_INCLUDE_DIRS}
)

if (NOT THRUST_INCLUDE_DIR)
    message(FATAL_ERROR "Need Thrust ≥ 2.2 (with device_uvector.h) in CUDA Toolkit")
endif()

if (NOT TARGET Thrust::Thrust)
    add_library(Thrust::Thrust INTERFACE IMPORTED)
    target_include_directories(Thrust::Thrust INTERFACE ${THRUST_INCLUDE_DIR})
endif()


# ────────────────────────────────────────────────
# 3. Options
# ────────────────────────────────────────────────
option(GALP_BUILD_TESTS "Build GALP tests" OFF)
option(GALP_BUILD_BENCHMARKS "Build GALP benchmarks" OFF)
option(GALP_VERBOSE_BUILD "Print every compile/link command" OFF)
if (GALP_VERBOSE_BUILD)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

set(COMMON_WARNINGS "")
set(CLANG_ONLY_WARNINGS
    -Wno-c++98-compat-local-type-template-args
    -Wno-c++98-compat-pedantic
    -Wno-c++98-compat
    -Wno-padded
    -Wno-float-equal
    -Wno-global-constructors
    -Wno-exit-time-destructors
    -Wno-float-conversion
    -Wno-sign-conversion
    -Wno-implicit-int-float-conversion
    -Wno-unused-parameter
    -Wno-unused-variable
)

macro(apply_warnings tgt)
    target_compile_options(${tgt} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${COMMON_WARNINGS}>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:Clang>>:${CLANG_ONLY_WARNINGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:${COMMON_WARNINGS}>)
endmacro()

add_compile_definitions(FLS_GALP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# ────────────────────────────────────────────────
# 4. GALP external dependencies
# ────────────────────────────────────────────────
add_subdirectory(external)

# ────────────────────────────────────────────────
# 5. GALP core
# ────────────────────────────────────────────────
add_subdirectory(src)

# ────────────────────────────────────────────────
# 6. tests / benchmark
# ────────────────────────────────────────────────
if (GALP_BUILD_TESTS)
    include(CTest)
    enable_testing()

    # 提供 gtest_discover_tests 宏
    include(GoogleTest)

    # standalone 构建时，如果父工程没提供 gtest targets，就自己拉取
    if (NOT TARGET GTest::gtest_main)
        include(FetchContent)
        FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.15.2
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    add_subdirectory(test)
endif()

if (GALP_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()
