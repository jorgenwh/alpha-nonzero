#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "enginewrapper.h"

namespace py = pybind11;

PYBIND11_MODULE(pystockfish_C, m) {
    m.doc() = "pystockfish_C module";

    py::class_<EngineWrapper>(m, "EngineWrapper")
        .def(py::init<>())
        .def("set_position", &EngineWrapper::set_position)
        .def("set_position_with_moves", &EngineWrapper::set_position_with_moves)
        .def("set_num_threads", &EngineWrapper::set_num_threads)
        .def("set_ht_size", &EngineWrapper::set_ht_size)
        .def("set_multipv", &EngineWrapper::set_multipv)

        .def("go", &EngineWrapper::go)
        .def("go_nodes_limit", &EngineWrapper::go_nodes_limit)
        .def("stop", &EngineWrapper::stop)

        .def("get_evaluations", &EngineWrapper::get_evaluations)
        .def("clear_evaluations", &EngineWrapper::clear_evaluations)

        .def("nodes", &EngineWrapper::nodes)
        .def("side_to_move", &EngineWrapper::side_to_move)
        .def("visualize", &EngineWrapper::visualize)
        .def("tostring", &EngineWrapper::tostring)
        .def("rule50_count", &EngineWrapper::rule50_count)
        ;
}

