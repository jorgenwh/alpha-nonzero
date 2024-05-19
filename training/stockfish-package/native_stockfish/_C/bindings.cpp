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
#include <assert.h>

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

class StockfishWrapper {
public:
  StockfishWrapper() {
    Bitboards::init();
    Position::init();
    engine_m = new Engine("");
    auto& options = engine_m->get_options();

    options["Debug Log File"] << Option("", [](const Option& o) { start_logger(o); });
    options["Threads"] << Option(6, 1, 1024, [this](const Option&) { engine_m->resize_threads(); });
    options["Hash"] << Option(4096, 1, 33554432 , [this](const Option& o) { engine_m->set_tt_size(o); });
    options["Clear Hash"] << Option([this](const Option&) { engine_m->search_clear(); });
    options["Ponder"] << Option(false);
    options["MultiPV"] << Option(MAX_MOVES, 1, MAX_MOVES);
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
                                  [this](const Option& o) { engine_m->load_big_network(o); });
    options["EvalFileSmall"] << Option(EvalFileDefaultNameSmall,
                                       [this](const Option& o) { engine_m->load_small_network(o); });

    engine_m->set_on_iter([this](const auto& i) { on_iter(i); });
    engine_m->set_on_update_no_moves([this](const auto& i) { on_update_no_moves(i); });
    engine_m->set_on_update_full([this](const auto& i) { on_update_full(i, engine_m->get_options()["UCI_ShowWDL"]); });
    engine_m->set_on_bestmove([this](const auto& bm, const auto& p) { on_bestmove(bm, p); });

    engine_m->load_networks();
    engine_m->resize_threads();
    engine_m->search_clear();  // After threads are up
  }
  ~StockfishWrapper() { }

  void set_position(std::string fen) {
    engine_m->set_position(fen, std::vector<std::string>());
  }

  std::vector<std::string> get_legal_moves() {
    std::vector<std::string> legal_moves;
    auto moves = engine_m->get_legal_moves();
    for (auto& move : moves) {
      legal_moves.push_back(move.to_uci());
    }
    return legal_moves;
  }

  void go() {
    Search::LimitsType limits;
    limits.infinite = true;
    engine_m->go(limits);
  }

  void stop() {
    engine_m->stop();
  }

  int side_to_move() const {
    Color side = engine_m->side_to_move();
    return side; 
  }

  int get_score() const {
    if (is_mate_m) {
      return mate_m;
    }
    return cp_m;
  }

  bool is_mate() const {
    return is_mate_m;
  }

  void visualize() const {
    std::cout << engine_m->visualize() << std::endl;
  }

private:
  Engine* engine_m;
  int cp_m = 0;
  int mate_m = 0;
  bool is_mate_m = false;

  void on_update_full(const Engine::InfoFull& info, const Option& showWDL) {
    Score score = info.score;
    if (score.is<Score::Mate>()) {
      auto side_to_move = engine_m->side_to_move();
      this->mate_m = score.get<Score::Mate>().plies;
      this->mate_m = (side_to_move == Color::WHITE) ? mate_m : -mate_m;
      this->cp_m = 0;
      this->is_mate_m = true;
    }
    if (score.is<Score::Tablebase>()) {
      std::cout << "DISASTER HELP" << std::endl;
      assert(false);
    }
    if (score.is<Score::InternalUnits>()) {
      auto side_to_move = engine_m->side_to_move();
      cp_m = score.get<Score::InternalUnits>().value;
      this->cp_m = (side_to_move == Color::WHITE) ? cp_m : -cp_m;
      this->mate_m = 0;
      this->is_mate_m = false;
    }
  }
  void on_iter(const Engine::InfoIter& info) { }
  void on_update_no_moves(const Engine::InfoShort& info) { }
  void on_bestmove(std::string_view bestmove, std::string_view ponder) { }

};

PYBIND11_MODULE(native_stockfish_C, m) {
  m.doc() = "native_stockfish_C module";

  py::class_<StockfishWrapper>(m, "Stockfish")
    .def(py::init<>())
    .def("set_position", &StockfishWrapper::set_position)
    .def("get_legal_moves", &StockfishWrapper::get_legal_moves)
    .def("go", &StockfishWrapper::go)
    .def("stop", &StockfishWrapper::stop)
    .def("side_to_move", &StockfishWrapper::side_to_move)
    .def("get_score", &StockfishWrapper::get_score)
    .def("is_mate", &StockfishWrapper::is_mate)
    .def("visualize", &StockfishWrapper::visualize)
    ;
}

