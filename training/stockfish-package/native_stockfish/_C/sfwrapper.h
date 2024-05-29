#pragma once

#include <string>
#include <vector>
#include <string_view>
#include <unordered_map>
#include <assert.h>

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

class StockfishWrapper {
public:
  StockfishWrapper();
  ~StockfishWrapper();

  void set_position(std::string fen);
  void set_position_ml(std::string fen, std::vector<std::string> move_list);
  void set_num_threads(int num_threads);
  void set_ht_size(int ht_mb);
  void set_multipv(int multipv);

  void go();
  void stop();

  std::unordered_map<std::string, std::string> get_evaluations() const;
  void clear_evaluations();

  int side_to_move() const;
  std::string visualize() const;
  std::string tostring() const;
private:
  Engine* engine_m;
  std::unordered_map<std::string, std::string> evaluations_m;
  bool is_going_m = false;

  void on_update_full(const Engine::InfoFull& info, const Stockfish::Option& showWDL);
  void on_iter(const Engine::InfoIter& info);
  void on_update_no_moves(const Engine::InfoShort& info);
  void on_bestmove(std::string_view bestmove, std::string_view ponder);
};