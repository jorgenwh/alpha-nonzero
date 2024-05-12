cmake_minimum_required(VERSION 3.16)
project(native_stockfish LANGUAGES CXX)

find_package(pybind11 REQUIRED)

#configure_file(
  #${CMAKE_SOURCE_DIR}/nn-ae6a388e4a1a.nnue
  #${CMAKE_BINARY_DIR}/nn-ae6a388e4a1a.nnue
  #COPYONLY
#)

#configure_file(
#  ${CMAKE_SOURCE_DIR}/nn-baff1ede1f90.nnue
#  ${CMAKE_BINARY_DIR}/nn-baff1ede1f90.nnue
#  COPYONLY
#)

add_subdirectory("native_stockfish/_C")
