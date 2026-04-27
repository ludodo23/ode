cmake_minimum_required(VERSION 3.20)
project(ode CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ─── Options ─────────────────────────────────────────────────────────────────
option(ENABLE_ASAN     "Enable AddressSanitizer" OFF)
option(ENABLE_COVERAGE "Enable coverage"         OFF)
option(ENABLE_PERF     "Enable perf-friendly flags" OFF)

# ─── Lib header-only ─────────────────────────────────────────────────────────
add_library(ode INTERFACE)
target_include_directories(ode INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(ode INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)

# ─── Tests ───────────────────────────────────────────────────────────────────
enable_testing()
add_subdirectory(tests)