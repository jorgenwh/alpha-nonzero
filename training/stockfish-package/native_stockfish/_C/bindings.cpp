#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "sfwrapper.h"

namespace py = pybind11;

PYBIND11_MODULE(native_stockfish_C, m) {
  m.doc() = "native_stockfish_C module";

  py::class_<StockfishWrapper>(m, "Stockfish")
    .def(py::init<>())
    .def("set_position", &StockfishWrapper::set_position)
    .def("set_position_ml", &StockfishWrapper::set_position_ml)
    .def("set_num_threads", &StockfishWrapper::set_num_threads)
    .def("set_ht_size", &StockfishWrapper::set_ht_size)
    .def("set_multipv", &StockfishWrapper::set_multipv)

    .def("go", &StockfishWrapper::go)
    .def("stop", &StockfishWrapper::stop)

    .def("get_evaluations", &StockfishWrapper::get_evaluations)
    .def("clear_evaluations", &StockfishWrapper::clear_evaluations)
    
    .def("side_to_move", &StockfishWrapper::side_to_move)
    .def("visualize", &StockfishWrapper::visualize)
    .def("tostring", &StockfishWrapper::tostring)
    ;
}

