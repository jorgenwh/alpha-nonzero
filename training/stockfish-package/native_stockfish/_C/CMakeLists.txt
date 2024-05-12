cmake_minimum_required(VERSION 3.16)
project(native_stockfish_C LANGUAGES CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)

# CXX flags
set(CMAKE_CXX_FLAGS "-O3 -march=native -pthread")

#configure_file(
#  ${CMAKE_SOURCE_DIR}/nn-ae6a388e4a1a.nnue
#  ${CMAKE_BINARY_DIR}/nn-ae6a388e4a1a.nnue
#  COPYONLY
#)

#configure_file(
#  ${CMAKE_SOURCE_DIR}/nn-baff1ede1f90.nnue
#  ${CMAKE_BINARY_DIR}/nn-baff1ede1f90.nnue
#  COPYONLY
#)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# static link libstdc++ where i choose where from
#set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")


# set sources
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
)

pybind11_add_module(${PROJECT_NAME} ${SOURCES} "bindings.cpp")
