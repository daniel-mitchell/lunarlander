#ifndef _MULTI_GAUSS_3_AGENT_HPP
#define _MULTI_GAUSS_3_AGENT_HPP

#include <vector>
#include <map>
#include <random>
#include <boost/math/constants/constants.hpp>
#include <Eigen/Core>

#include "simulator.hpp"
#include "policy_gradient_agent.hpp"
#include "tile_coder.hpp"

using Eigen::VectorXd;

class multi_gauss_3_agent {

  VectorXd max_state, min_state, max_clip_state, min_clip_state;

  hashing_tile_coder tc;
  td_critic critic_left, critic_middle, critic_right;
  std::vector<policy_gradient_actor> thrust_actors;// thrust_actor_left, thrust_actor_middle, thrust_actor_right;
  int chosen_thrust;
  std::mt19937 rng2;
  double epsilon;
  int count_left, count_middle, count_right;
  double last_critic_value_left, last_critic_value_middle, last_critic_value_right;
  bool weighted_dist_choice;
  double ucb_factor;
  int timestep = 0;
  std::map<int, int> num_left;
  std::map<int, int> num_middle;
  std::map<int, int> num_right;

  tile_coder make_tile_coder (double tile_weight_exponent, const std::vector<int>& subspaces);
  static std::vector<policy_gradient_actor> make_thrust_actors (policy_gradient_actor actor1, policy_gradient_actor\
                                                          actor2, policy_gradient_actor actor3);
  static policy_gradient_actor make_thrust_actor_left (const tile_coder_base& tc, double lambda, double alpha,
                                                      bool trunc_normal);
  static policy_gradient_actor make_thrust_actor_middle (const tile_coder_base& tc, double lambda, double alpha,
                                                      bool trunc_normal);
  static policy_gradient_actor make_thrust_actor_right (const tile_coder_base& tc, double lambda, double alpha,
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

  multi_gauss_3_agent(double lambda, double alpha_v, double alpha_u, double epsilon, double initial_value,
                     int num_features,
                     double tile_weight_exponent,
                     [[maybe_unused]] bool trunc_normal,
                     const std::vector<int>& subspaces,
                     bool weighted_dist_choice, double ucb_factor)
    : max_state(6), min_state(6), max_clip_state(6), min_clip_state(6),
      tc (make_tile_coder (tile_weight_exponent, subspaces), num_features),
      critic_left (tc, lambda, alpha_v, initial_value),
      critic_middle (tc, lambda, alpha_v, initial_value),
      critic_right (tc, lambda, alpha_v, initial_value),
      thrust_actors (make_thrust_actors(
        make_thrust_actor_left (tc, lambda, alpha_u, trunc_normal),
        make_thrust_actor_middle (tc, lambda, alpha_u, trunc_normal),
        make_thrust_actor_right (tc, lambda, alpha_u, trunc_normal)
      )),
      // thrust_actor_left (make_thrust_actor_left (tc, lambda, alpha_u, trunc_normal)),
      // thrust_actor_middle (make_thrust_actor_middle (tc, lambda, alpha_u, trunc_normal)),
      // thrust_actor_right (make_thrust_actor_right (tc, lambda, alpha_u, trunc_normal)),
      rng2(0), epsilon(epsilon), weighted_dist_choice(weighted_dist_choice), ucb_factor(ucb_factor)
  { }

  cart_pole_simulator::action initialize(std::mt19937& rng, VectorXd state);

  cart_pole_simulator::action update(std::mt19937& rng, VectorXd state,
                                        double reward, bool terminal=false, bool learn=true);

  const VectorXd& get_max_state() const { return max_state; }
  const VectorXd& get_min_state() const { return min_state; }
  double get_mu() const { return thrust_actors[chosen_thrust].get_mu(); }
  double get_sigma() const { return thrust_actors[chosen_thrust].get_sigma(); }
  double get_mu_grad() const { return thrust_actors[chosen_thrust].get_mu_grad(); }
  double get_sigma_grad() const { return thrust_actors[chosen_thrust].get_sigma_grad(); }
  double get_td_error() const { std::vector<td_critic> crit_vec = {critic_left, critic_middle, critic_right};
                                return crit_vec[chosen_thrust].get_td_error(); }
  double get_direction_ratio() { return ((double)count_right)/(count_right+count_left+count_middle); }
  int get_forward_count() { return count_right; }
  int get_backward_count() { return count_left; }
  double get_backward_critic_value() { return last_critic_value_left; }
  double get_forward_critic_value() { return last_critic_value_right; }

};

#endif
