cmake_minimum_required(VERSION 3.16)
project(pystockfish LANGUAGES CXX)

find_package(pybind11 REQUIRED)

add_subdirectory("pystockfish/_C")
