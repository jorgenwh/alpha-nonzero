cmake_minimum_required(VERSION 3.16)
project(native_stockfish_C LANGUAGES CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)

# CXX flags
#set(CMAKE_CXX_FLAGS "-pthread -mavx2 -mpopcnt -msse3 -mavx512f -mavx512bw")
set(CMAKE_CXX_FLAGS "-pthread -O3 -funroll-loops -msse -mmmx -msse2 -msse3 -msse4.1 -mavx2 -mavx512 -mavx512bw -mavx512vnni")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# add definitions
#add_compile_definitions(USE_AVX2)
add_compile_definitions(USE_AVX512)
add_compile_definitions(USE_POPCNT)
add_compile_definitions(USE_PREFETCH)
add_compile_definitions(IS_64BIT)

set(SOURCES
    src/benchmark.cpp 
    src/bitboard.cpp 
    src/evaluate.cpp 
    src/misc.cpp 
    src/movegen.cpp 
    src/movepick.cpp 
    src/position.cpp
    src/search.cpp 
    src/thread.cpp 
    src/timeman.cpp 
    src/tt.cpp 
    src/uci.cpp 
    src/ucioption.cpp 
    src/tune.cpp 
    src/syzygy/tbprobe.cpp
    src/nnue/nnue_misc.cpp 
    src/nnue/features/half_ka_v2_hm.cpp 
    src/nnue/network.cpp 
    src/engine.cpp 
    src/score.cpp
    sfwrapper.cpp
)

pybind11_add_module(${PROJECT_NAME} ${SOURCES} "bindings.cpp")
