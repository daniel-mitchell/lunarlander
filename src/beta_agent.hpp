#ifndef _BETA_AGENT_HPP
#define _BETA_AGENT_HPP

#include <vector>
#include <random>
#include <boost/math/constants/constants.hpp>
#include <Eigen/Core>

#include "simulator.hpp"
#include "beta_pg_agent.hpp"
#include "tile_coder.hpp"

using Eigen::VectorXd;

class beta_agent {

  VectorXd max_state, min_state, max_clip_state, min_clip_state;

  hashing_tile_coder tc;
  td_critic critic;
  beta_pg_actor thrust_actor;

  tile_coder make_tile_coder (double tile_weight_exponent, const std::vector<int>& subspaces);
  static beta_pg_actor make_thrust_actor (const tile_coder_base& tc, double lambda, double alpha);
  // static test_pg_actor make_rcs_actor (const tile_coder_base& tc, double lambda, double alpha);

  cart_pole_simulator::action compute_action(std::mt19937& rng, const VectorXi& features) {
    return cart_pole_simulator::action(thrust_actor.act(rng, features));
  }

  void clip_state(VectorXd& state) {
    for (unsigned int i = 0; i < state.size(); ++i) {
      state(i) = std::max(min_state(i), std::min(state(i), max_state(i)));
    }
  }

public:

  beta_agent(double lambda, double alpha_v, double alpha_u, double initial_value,
                     int num_features,
                     double tile_weight_exponent,
                     [[maybe_unused]] bool trunc_normal,
                     const std::vector<int>& subspaces)
    : max_state(2), min_state(2), max_clip_state(2), min_clip_state(2),
      tc (make_tile_coder (tile_weight_exponent, subspaces), num_features),
      critic (tc, lambda, alpha_v, initial_value),
      thrust_actor (make_thrust_actor (tc, lambda, alpha_u))
  { }

  cart_pole_simulator::action initialize(std::mt19937& rng, VectorXd state);

  cart_pole_simulator::action update(std::mt19937& rng, VectorXd state,
                                        double reward, bool terminal=false, bool learn=true);

  const VectorXd& get_max_state() const { return max_state; }
  const VectorXd& get_min_state() const { return min_state; }
  double get_mu() const { return 0; }
  double get_sigma() const { return 0; }
  double get_mu_grad() const { return 0; }
  double get_sigma_grad() const { return 0; }
  double get_td_error() const { return 0; }
  double get_direction_ratio() { return 0; }
  int get_forward_count() { return 0; }
  int get_backward_count() { return 0; }
  double get_backward_critic_value() { return 0; }
  double get_forward_critic_value() { return 0; }

};

#endif
