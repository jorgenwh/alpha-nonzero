cmake_minimum_required(VERSION 3.16)
project(pystockfish_C LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS 
    "-m64 -Wl,--no-as-needed -lpthread -Wall -Wcast-qual -std=c++17 -pedantic -Wextra -Wshadow -Wmissing-declarations -m64 -DUSE_PTHREADS -DNDEBUG -O3 -funroll-loops -DIS_64BIT -msse -msse3 -mpopcnt -DUSE_POPCNT -DUSE_AVX2 -mavx2 -mbmi -DUSE_SSE41 -msse4.1 -DUSE_SSSE3 -mssse3 -DUSE_SSE2 -msse2 -DGIT_SHA=c17d73c5 -DGIT_DATE=20240601 -DARCH=x86-64-avx2 -flto -flto-partition=one -flto=jobserver"
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

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
    enginewrapper.cpp
)

pybind11_add_module(${PROJECT_NAME} ${SOURCES} "bindings.cpp")
