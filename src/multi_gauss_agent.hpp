#ifndef _MULTI_GAUSS_AGENT_HPP
#define _MULTI_GAUSS_AGENT_HPP

#include <vector>
#include <map>
#include <random>
#include <boost/math/constants/constants.hpp>
#include <Eigen/Core>

#include "simulator.hpp"
#include "policy_gradient_agent.hpp"
#include "tile_coder.hpp"

using Eigen::VectorXd;

class multi_gauss_agent {

  VectorXd max_state, min_state, max_clip_state, min_clip_state;

  hashing_tile_coder tc;
  td_critic critic_forward, critic_backward;
  policy_gradient_actor thrust_actor_forward, thrust_actor_backward;
  bool thrust_forward;
  std::mt19937 rng2;
  double epsilon;
  int count_forward, count_backward;
  double last_critic_value_forward, last_critic_value_backward;
  bool weighted_dist_choice;
  double ucb_factor;
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

  cart_pole_simulator::action compute_action(std::mt19937& rng, const VectorXi& features);

  void clip_state(VectorXd& state) {
    for (unsigned int i = 0; i < state.size(); ++i) {
      state(i) = std::max(min_state(i), std::min(state(i), max_state(i)));
    }
  }

public:

  multi_gauss_agent(double lambda, double alpha_v, double alpha_u, double epsilon, double initial_value,
                     int num_features,
                     double tile_weight_exponent,
                     [[maybe_unused]] bool trunc_normal,
                     const std::vector<int>& subspaces,
                     bool weighted_dist_choice, double ucb_factor)
    : max_state(6), min_state(6), max_clip_state(6), min_clip_state(6),
      tc (make_tile_coder (tile_weight_exponent, subspaces), num_features),
      critic_forward (tc, lambda, alpha_v, initial_value),
      critic_backward (tc, lambda, alpha_v, initial_value),
      thrust_actor_forward (make_thrust_actor_forward (tc, lambda, alpha_u, trunc_normal)),
      thrust_actor_backward (make_thrust_actor_backward (tc, lambda, alpha_u, trunc_normal)),
      rng2(0), epsilon(epsilon), weighted_dist_choice(weighted_dist_choice), ucb_factor(ucb_factor)
  { }

  cart_pole_simulator::action initialize(std::mt19937& rng, VectorXd state);

  cart_pole_simulator::action update(std::mt19937& rng, VectorXd state,
                                        double reward, bool terminal=false, bool learn=true);

  const VectorXd& get_max_state() const { return max_state; }
  const VectorXd& get_min_state() const { return min_state; }
  double get_mu() const { return thrust_forward ? thrust_actor_forward.get_mu() : thrust_actor_backward.get_mu(); }
  double get_sigma() const { return thrust_forward ? thrust_actor_forward.get_sigma() :\
                                                      thrust_actor_backward.get_sigma(); }
  double get_mu_grad() const { return thrust_forward ? thrust_actor_forward.get_mu_grad() :\
                                                       thrust_actor_backward.get_mu_grad(); }
  double get_sigma_grad() const { return thrust_forward ? thrust_actor_forward.get_sigma_grad() :\
                                                      thrust_actor_backward.get_sigma_grad(); }
  double get_td_error() const { return thrust_forward ? critic_forward.get_td_error() :\
                                                       critic_backward.get_td_error(); }
  double get_direction_ratio() { return ((double)count_forward)/(count_forward+count_backward); }
  int get_forward_count() { return count_forward; }
  int get_backward_count() { return count_backward; }
  double get_backward_critic_value() { return last_critic_value_backward; }
  double get_forward_critic_value() { return last_critic_value_forward; }

};

#endif
