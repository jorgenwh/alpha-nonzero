#include <string>
#include <string_view>
#include <sstream>
#include <unordered_map>
#include <assert.h>
#include <iostream>

#include "enginewrapper.h"

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

using namespace Stockfish;

EngineWrapper::EngineWrapper() {
    Bitboards::init();
    Position::init();
    engine_m = new Engine("");
    auto& options = engine_m->get_options();

    options["Debug Log File"] << Option("", [](const Option& o) { start_logger(o); });
    options["Threads"] << Option(DEFAULT_THREADS, 1, 1024, [this](const Option&) { engine_m->resize_threads(); });
    options["Hash"] << Option(DEFAULT_HASH, 1, 33554432 , [this](const Option& o) { engine_m->set_tt_size(o); });
    options["Clear Hash"] << Option([this](const Option&) { engine_m->search_clear(); });
    options["Ponder"] << Option(false);
    options["MultiPV"] << Option(DEFAULT_PV, 1, MAX_MOVES);
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

    Tune::init(options);
}

EngineWrapper::~EngineWrapper() {
    if (is_going_m) { stop(); }
}

void EngineWrapper::set_position(std::string fen) {
    engine_m->set_position(fen, std::vector<std::string>());
}

void EngineWrapper::set_position_with_moves(std::string fen, std::vector<std::string> move_list) {
    engine_m->set_position(fen, move_list);
}

void EngineWrapper::go() {
    is_going_m = true;

    Search::LimitsType limits;
    limits.infinite = true;

    engine_m->go(limits);
}

void EngineWrapper::go_nodes_limit(int nodes) {
    is_going_m = true;

    Search::LimitsType limits;
    limits.nodes = nodes;

    engine_m->go(limits);
    engine_m->wait_for_search_finished();
}

void EngineWrapper::stop() {
    engine_m->stop();
    is_going_m = false;
}

void EngineWrapper::set_num_threads(int num_threads) {
    std::istringstream is;
    is.str("name Threads value " + std::to_string(num_threads));
    engine_m->get_options().setoption(is);
}

void EngineWrapper::set_ht_size(int ht_mb) {
    std::istringstream is;
    is.str("name Hash value " + std::to_string(ht_mb));
    engine_m->get_options().setoption(is);
}

void EngineWrapper::set_multipv(int multipv) {
    std::istringstream is;
    is.str("name MultiPV value " + std::to_string(multipv));
    engine_m->get_options().setoption(is);
}

std::unordered_map<std::string, std::string> EngineWrapper::get_evaluations() const {
    return evaluations_m;
}

void EngineWrapper::clear_evaluations() {
    evaluations_m.clear();
}

int EngineWrapper::nodes() const {
    return nodes_m;
}

int EngineWrapper::side_to_move() const {
    Color side = engine_m->side_to_move();
    return side;
}

std::string EngineWrapper::visualize() const {
    std::ostringstream oss;
    oss << engine_m->visualize() << "\n";
    return oss.str();
}

std::string EngineWrapper::tostring() const {
    std::ostringstream oss;
    oss << engine_m->visualize() << "\n";
    oss << "\nOptions:";
    oss << engine_m->get_options();
    return oss.str();
}

int EngineWrapper::rule50_count() const {
    return engine_m->rule50_count();
}

void EngineWrapper::on_update_full(const Engine::InfoFull& info, const Option& showWDL) {
    nodes_m = info.nodes;

    Score score = info.score;
    auto side_to_move = engine_m->side_to_move();
    std::string move = std::string(info.pv.substr(0, info.pv.find(' ')));
    if (score.is<Score::Mate>()) {
        int mate = score.get<Score::Mate>().plies;
        mate = (side_to_move == Color::WHITE) ? mate : -mate;
        evaluations_m[move] = "mate " + std::to_string(mate);
    }
    if (score.is<Score::InternalUnits>()) {
        int cp = score.get<Score::InternalUnits>().value;
        cp = (side_to_move == Color::WHITE) ? cp : -cp;
        evaluations_m[move] = "cp " + std::to_string(cp);
    }
    assert(false);
}
void EngineWrapper::on_iter(const Engine::InfoIter& info) { }
void EngineWrapper::on_update_no_moves(const Engine::InfoShort& info) { }
void EngineWrapper::on_bestmove(std::string_view bestmove, std::string_view ponder) { }
