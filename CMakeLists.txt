cmake_minimum_required(VERSION 3.20)
project(ode VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

option(ODE_BUILD_TESTS "Build tests" ON)

if(ODE_BUILD_TESTS)
    find_package(Catch2 3 REQUIRED)

    add_executable(ode_tests
        tests/test_fixed_steppers.cpp
        tests/test_adaptive_steppers.cpp
        tests/test_verlet.cpp
        tests/test_state_and_concepts.cpp
    )

    target_link_libraries(ode_tests PRIVATE ode Catch2::Catch2WithMain)

    include(CTest)
    include(Catch)
    catch_discover_tests(ode_tests)
endif()
