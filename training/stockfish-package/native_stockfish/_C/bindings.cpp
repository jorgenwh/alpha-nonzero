#include <iostream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "engine.h"
#include "bitboard.h"
#include "misc.h"
#include "position.h"
#include "types.h"
#include "uci.h"
#include "tune.h"
#include "benchmark.h"
#include "evaluate.h"
#include "movegen.h"
#include "score.h"
#include "search.h"
#include "syzygy/tbprobe.h"
#include "ucioption.h"

namespace py = pybind11;
using namespace Stockfish;

constexpr auto StartFEN  = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
constexpr int  MaxHashMB = Is64Bit ? 33554432 : 2048;

template<typename... Ts>
struct overload: Ts... {
    using Ts::operator()...;
};

template<typename... Ts>
overload(Ts...) -> overload<Ts...>;


class StockfishWrapper {
public:
  StockfishWrapper() {
    Bitboards::init();
    Position::init();
    engine = new Engine("/home/jorgen/projects/alpha-nonzero/training/stockfish/stockfish-ubuntu-x86-64-avx2");

    auto& options = engine->get_options();

    options["Debug Log File"] << Option("", [](const Option& o) { start_logger(o); });
    options["Threads"] << Option(1, 1, 1024, [this](const Option&) { engine->resize_threads(); });
    options["Hash"] << Option(16, 1, MaxHashMB, [this](const Option& o) { engine->set_tt_size(o); });
    options["Clear Hash"] << Option([this](const Option&) { engine->search_clear(); });
    options["Ponder"] << Option(false);
    options["MultiPV"] << Option(1, 1, MAX_MOVES);
    options["Skill Level"] << Option(20, 0, 20);
    options["Move Overhead"] << Option(10, 0, 5000);
    options["nodestime"] << Option(0, 0, 10000);
    options["UCI_Chess960"] << Option(false);
    options["UCI_LimitStrength"] << Option(false);
    options["UCI_Elo"] << Option(1320, 1320, 3190);
    options["UCI_ShowWDL"] << Option(false);
    options["SyzygyPath"] << Option("<empty>", [](const Option& o) { Tablebases::init(o); });
    options["SyzygyProbeDepth"] << Option(1, 1, 100);
    options["Syzygy50MoveRule"] << Option(true);
    options["SyzygyProbeLimit"] << Option(7, 0, 7);
    options["EvalFile"] << Option(EvalFileDefaultNameBig,
                                  [this](const Option& o) { engine->load_big_network(o); });
    options["EvalFileSmall"] << Option(EvalFileDefaultNameSmall,
                                       [this](const Option& o) { engine->load_small_network(o); });


    engine->load_networks();
    engine->resize_threads();
    engine->search_clear();  // After threads are up
  }
  ~StockfishWrapper() { }

  void go() {
    Search::LimitsType limits;
    std::string token;

    limits.startTime = now();
    limits.depth = 15;

    engine->go(limits);
  }

private:
  Engine* engine;
};

PYBIND11_MODULE(native_stockfish_C, m) {
  m.doc() = "native_stockfish_C module";

  py::class_<StockfishWrapper>(m, "StockfishWrapper")
    .def(py::init<>())
    .def("go", &StockfishWrapper::go)
    ;
}

