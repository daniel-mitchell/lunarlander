#ifndef _MULTI_DIST_AGENT_HPP
#define _MULTI_DIST_AGENT_HPP

#include <vector>
#include <random>
#include <boost/math/constants/constants.hpp>
#include <Eigen/Core>

#include "simulator.hpp"
#include "tile_coder.hpp"

using Eigen::VectorXd;
using Eigen::Array2d;
using namespace std;

class multi_dist_agent {
  constexpr static double alphaR = 0;
  constexpr static double maxAlphaBeta = 1000;
  constexpr static bool printOutput = true;

  VectorXd max_state, min_state, max_clip_state, min_clip_state;
  hashing_tile_coder tc;
  double lambda;
  double alphaU, alphaVInit;

  double alphaDecay = 0.8;
  double epsilon = 0;
  double gamma = 1;
  bool inac = false;
  bool s = false;
  mt19937 rng;
  mt19937 rng2;
  VectorXd u, v, w;
  double maxParam = log(maxAlphaBeta);
  double maxReturn = std::numeric_limits<double>::lowest();
  int timedOut = 0;
  int i_episode;
  vector<double> returns;
  Array2d minAction, maxAction;

  double returnAmount, rBar, alphaV;
  VectorXi x;
  VectorXd eu, ev;

  bool chosenDist;
  lunar_lander_simulator::action action;
  Array2d alpha, beta;

  tile_coder make_tile_coder (double tile_weight_exponent, const std::vector<int>& subspaces);

  lunar_lander_simulator::action computeAction();
  void selectAction();
  double sampleBeta(double alphaa, double betaa);
  double sumIndices(VectorXd vec, VectorXi ind, int offset=0);
  VectorXd computeGradLog();

  void clip_state(VectorXd& state) {
    for (unsigned int i = 0; i < state.size(); ++i) {
      state(i) = std::max(min_state(i), std::min(state(i), max_state(i)));
    }
  }

public:

  multi_dist_agent(double lambda, double alpha_v, double alpha_u, double initial_value,
                     int num_features,
                     double tile_weight_exponent,
                     bool trunc_normal,
                     const std::vector<int>& subspaces, int seed);

  lunar_lander_simulator::action initialize(std::mt19937& rng, VectorXd state);

  lunar_lander_simulator::action update(std::mt19937& rng, VectorXd state,
                                        double reward, bool terminal=false, bool learn=true);

  const VectorXd& get_max_state() const { return max_state; }
  const VectorXd& get_min_state() const { return min_state; }

};

#endif
