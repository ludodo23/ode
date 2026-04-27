find_package(Catch2 3 REQUIRED)

add_executable(ode_tests
    test_fixed_steppers.cpp
    test_adaptive_steppers.cpp
    test_verlet.cpp
    test_state_and_concepts.cpp
)

target_link_libraries(ode_tests PRIVATE
    ode
    Catch2::Catch2WithMain
)

target_compile_definitions(ode_tests PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:_USE_MATH_DEFINES>
)

# ─── Flags optionnels ────────────────────────────────────────────────────────
if(ENABLE_ASAN)
    target_compile_options(ode_tests PRIVATE -fsanitize=address -g)
    target_link_options(ode_tests    PRIVATE -fsanitize=address)
endif()

if(ENABLE_COVERAGE)
    target_compile_options(ode_tests PRIVATE --coverage -O0 -g)
    target_link_options(ode_tests    PRIVATE --coverage)
endif()

if(ENABLE_PERF)
    target_compile_options(ode_tests PRIVATE -g -fno-omit-frame-pointer)
endif()

# ─── CTest + Catch2 ──────────────────────────────────────────────────────────
include(CTest)
include(Catch)
catch_discover_tests(ode_tests)