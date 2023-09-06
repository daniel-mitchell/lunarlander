#ifndef _MULTI_GAUSS_4_AGENT_HPP
#define _MULTI_GAUSS_4_AGENT_HPP

#include <vector>
#include <map>
#include <random>
#include <boost/math/constants/constants.hpp>
#include <Eigen/Core>

#include "simulator.hpp"
#include "policy_gradient_agent.hpp"
#include "tile_coder.hpp"

using Eigen::VectorXd;

class multi_gauss_4_agent {

  VectorXd max_state, min_state, max_clip_state, min_clip_state;

  hashing_tile_coder tc;
  td_critic critic_forward, critic_backward;
  VectorXd parameters, weights, parameter_trace, weight_trace; //critic_forward, critic_backward;
  double ii = 1;
  double alpha_u, alpha_v, alpha_uu, alpha_vv, alpha_r;
  VectorXd trace;
  double lambda;
  VectorXi oldFeatures;
  policy_gradient_actor thrust_actor_forward, thrust_actor_backward;
  bool thrust_forward;
  std::mt19937 rng2;
  double epsilon;
  double gamma = 1;
  int count_forward, count_backward;
  double last_critic_value_forward, last_critic_value_backward;
  bool weighted_dist_choice;
  double ucb_factor;
  bool continuing;
  double rBar = 0;
  int timestep = 0;
  std::map<int, int> num_forward;
  std::map<int, int> num_backward;

  tile_coder make_tile_coder (double tile_weight_exponent, const std::vector<int>& subspaces);
  static policy_gradient_actor make_thrust_actor_forward (const tile_coder_base& tc, double lambda, double alpha,
                                                      bool trunc_normal);
  static policy_gradient_actor make_thrust_actor_backward (const tile_coder_base& tc, double lambda, double alpha,
                                                      bool trunc_normal);
  // static policy_gradient_actor make_rcs_actor_cw (const tile_coder_base& tc, double lambda, double alpha,
  //                                                     bool trunc_normal);
  // static policy_gradient_actor make_rcs_actor_ccw (const tile_coder_base& tc, double lambda, double alpha,
  //                                                     bool trunc_normal);

  mountain_car_simulator::action compute_action([[maybe_unused]] std::mt19937& rng, const VectorXi& features);

  void clip_state(VectorXd& state) {
    for (unsigned int i = 0; i < state.size(); ++i) {
      state(i) = std::max(min_state(i), std::min(state(i), max_state(i)));
    }
  }

public:

  multi_gauss_4_agent([[maybe_unused]] double lambda, double lambda_c, double alpha_v, double alpha_u, double alpha_vv,
                     double alpha_uu, double alpha_r, double epsilon, double gamma,
                     double initial_value,
                     int num_features,
                     double tile_weight_exponent,
                     [[maybe_unused]] bool trunc_normal,
                     const std::vector<int>& subspaces,
                     bool weighted_dist_choice, double ucb_factor, bool continuing)
    : max_state(6), min_state(6), max_clip_state(6), min_clip_state(6),
      tc (make_tile_coder (tile_weight_exponent, subspaces), num_features),
      critic_forward (tc, lambda_c, alpha_vv, initial_value),
      critic_backward (tc, lambda_c, alpha_vv, initial_value),
      parameters(VectorXd::Constant(2*tc.get_num_features(), initial_value)),
      weights(VectorXd::Constant(tc.get_num_features(), initial_value)),
      parameter_trace(VectorXd::Constant(2*tc.get_num_features(), 0)),
      weight_trace(VectorXd::Constant(tc.get_num_features(), 0)),
      alpha_u(alpha_u), alpha_v(alpha_v), alpha_uu(alpha_uu), alpha_vv(alpha_vv), alpha_r(alpha_r),
      trace(VectorXd::Zero(tc.get_num_features())),
      lambda(lambda),
      thrust_actor_forward (make_thrust_actor_forward (tc, lambda, alpha_uu, trunc_normal)),
      thrust_actor_backward (make_thrust_actor_backward (tc, lambda, alpha_uu, trunc_normal)),
      rng2(0), epsilon(epsilon), gamma(gamma), weighted_dist_choice(weighted_dist_choice), ucb_factor(ucb_factor),
      continuing(continuing)
  { }

  mountain_car_simulator::action initialize(std::mt19937& rng, VectorXd state);

  mountain_car_simulator::action update(std::mt19937& rng, VectorXd state,
                                        double reward, bool terminal=false, bool learn=true);

  VectorXi get_state_action_features(const VectorXi& features, bool action);
  Eigen::Vector2d get_policy_probs(const VectorXi& features);

  const VectorXd& get_max_state() const { return max_state; }
  const VectorXd& get_min_state() const { return min_state; }
  double get_mu() const { return 0; }
  double get_sigma() const { return 0; }
  double get_mu_grad() const { return 0; }
  double get_sigma_grad() const { return 0; }
  double get_td_error() const { return 0; }
  double get_direction_ratio() { return ((double)count_forward)/(count_forward+count_backward); }
  int get_forward_count() { return count_forward; }
  int get_backward_count() { return count_backward; }
  double get_backward_critic_value() { return last_critic_value_backward; }
  double get_forward_critic_value() { return last_critic_value_forward; }

};

#endif
