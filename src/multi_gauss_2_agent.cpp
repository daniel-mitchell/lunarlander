#include "multi_gauss_2_agent.hpp"

#include <limits>
#include <iostream>
#include <boost/math/special_functions/sign.hpp>

using Eigen::VectorXd;

tile_coder multi_gauss_2_agent::make_tile_coder(double tile_weight_exponent, const std::vector<int>& subspaces) {

  // const double PI = boost::math::constants::pi<double>();

  struct {
    bool state_signed, state_bounded;
    double tile_size;
    unsigned int num_tiles, num_offsets;
  } cfg[] =
      { // #signed, bounded, tile_size, num_tiles, num_offsets
        {     true,    true,      0.96,        10,          10 }, //position
        {     true,   false,       0.4,        10,          10 }, //velocity
        {     true,    true,    0.0836,        10,          10 }, //angle
        {     true,   false,     0.275,        10,          10 }  //angular velocity
      };
      // A teaching method for reinforcement learning (JA Clouse, PE Utgoff, 1992) gives bounds for velocities
      // { // #signed, bounded, tile_size, num_tiles, num_offsets
      //   {    false,    true,       5.0,         6,           2 }, // xpos
      //   {    false,    true,       5.0,         4,           2 }, // ypos
      //   {     true,    true,       2.0,         4,           4 }, // xvel
      //   {     true,    true,       2.0,         4,           4 }, // yvel
      //   {     true,   false,      PI/2,         2,           8 }, // rot
      //   {     true,    true,      PI/6,         3,           4 }  // rotvel
      // };

  const unsigned int state_dim = 4;

  VectorXd tile_size (4);
  VectorXi num_tiles (4);
  VectorXi num_offsets (4);

  for (unsigned int i = 0; i < state_dim; ++i) {

    max_state(i) = cfg[i].tile_size * cfg[i].num_tiles - 1e-8;
    min_state(i) = cfg[i].state_signed ? -max_state(i) : 0.0;

    max_clip_state(i) = cfg[i].state_bounded ? max_state(i) : std::numeric_limits<double>::infinity();
    min_clip_state(i) = cfg[i].state_signed ? -max_clip_state(i) : 0.0;

    num_tiles(i) = cfg[i].num_tiles;
    if (cfg[i].state_signed) num_tiles(i) *= 2;
    if (cfg[i].state_bounded) num_tiles(i) += 1;

    tile_size(i) = cfg[i].tile_size;
    num_offsets(i) = cfg[i].num_offsets;
  }

  return tile_coder (tile_size, num_tiles, num_offsets, subspaces, tile_weight_exponent);

}


policy_gradient_actor multi_gauss_2_agent::make_thrust_actor_inner
(const tile_coder_base& tc, double lambda, double alpha, bool trunc_normal) {
  const double max_thrust = cart_pole_simulator::MAX_THRUST();
  return policy_gradient_actor (tc, lambda, alpha,
                                -max_thrust/2, max_thrust/2, // min and max action
                                max_thrust/64, max_thrust/4, // min and max sigma
                                0, max_thrust/8, // initial mu and sigma
                                1.0, // gamma
                                trunc_normal);
}


policy_gradient_actor multi_gauss_2_agent::make_thrust_actor_outer
(const tile_coder_base& tc, double lambda, double alpha, bool trunc_normal) {
  const double max_thrust = cart_pole_simulator::MAX_THRUST();
  return policy_gradient_actor (tc, lambda, alpha,
                                -max_thrust/2, max_thrust/2, // min and max action
                                max_thrust/64, max_thrust/4, // min and max sigma
                                0, max_thrust/8, // initial mu and sigma
                                1.0, // gamma
                                trunc_normal);
}


// policy_gradient_actor multi_gauss_2_agent::make_rcs_actor_cw
// (const tile_coder_base& tc, double lambda, double alpha, bool trunc_normal) {
//   const double max_rcs = cart_pole_simulator::MAX_RCS();
//   return policy_gradient_actor (tc, lambda, alpha,
//                                 0, max_rcs, // min and max action
//                                 max_rcs/32, max_rcs, // min and max sigma
//                                 0.0, max_rcs/4, // initial mu and sigma
//                                 1.0, // gamma
//                                 trunc_normal);
// }
//
//
// policy_gradient_actor multi_gauss_2_agent::make_rcs_actor_ccw
// (const tile_coder_base& tc, double lambda, double alpha, bool trunc_normal) {
//   const double max_rcs = cart_pole_simulator::MAX_RCS();
//   return policy_gradient_actor (tc, lambda, alpha,
//                                 -max_rcs, 0, // min and max action
//                                 max_rcs/32, max_rcs, // min and max sigma
//                                 0.0, max_rcs/4, // initial mu and sigma
//                                 1.0, // gamma
//                                 trunc_normal);
// }


cart_pole_simulator::action multi_gauss_2_agent::compute_action(std::mt19937& rng, const VectorXi& features) {
  // std::cout << 4 << " " << (critic_forward.get_weights() - critic_backward.get_weights()).norm() << std::endl;
  last_critic_value_inner = critic_inner.get_value();
  last_critic_value_outer = critic_outer.get_value();
  if (weighted_dist_choice) {
    thrust_inner = ((double)rng2()/rng2.max() < critic_inner.get_value()/\
                        (critic_inner.get_value() + critic_outer.get_value()));
  } else if (epsilon > 0 && (double)rng2()/rng2.max() < epsilon) {
    thrust_inner = ((double)rng2()/rng2.max() < 0.5);
  } else {
    //thrust_inner = ((double)rng2()/rng2.max() < 0.5);
    // last_critic_value_forward = critic_forward.get_value();
    // last_critic_value_backward = critic_backward.get_value();
    int nti = 0;
    int nto = 0;
    if (ucb_factor != 0) {
      for (int i = 0; i < features.size(); i++) {
        nti += num_inner[features[i]];
        nto += num_outer[features[i]];
      }
    }
    thrust_inner = (critic_inner.get_value() + ucb_factor*sqrt(log(timestep)/nti)\
                > critic_outer.get_value() + ucb_factor*sqrt(log(timestep)/nto));
  }
  if (thrust_inner) {
    count_inner++;
    if (ucb_factor != 0) {
      for (int i = 0; i < features.size(); i++) {
        num_inner[features[i]]++;
      }
    }
  } else {
    count_outer++;
    if (ucb_factor != 0) {
      for (int i = 0; i < features.size(); i++) {
        num_outer[features[i]]++;
      }
    }
  }
  // std::cout << 5 << " " << (critic_forward.get_weights() - critic_backward.get_weights()).norm() << std::endl;
  if (thrust_inner) {
    return cart_pole_simulator::action(thrust_actor_inner.act(rng, features));
  } else {
    double act = thrust_actor_outer.act(rng, features);
    return cart_pole_simulator::action(cart_pole_simulator::MAX_THRUST()/2*boost::math::sign(act) + act);
  }
  /*return cart_pole_simulator::action(thrust_inner ? thrust_actor_inner.act(rng, features) :\
                                                          thrust_actor_outer.act(rng, features));*/
}


cart_pole_simulator::action multi_gauss_2_agent::initialize(std::mt19937& rng, VectorXd state) {
  clip_state(state);
  VectorXi features = tc.indices(state);

  critic_inner.initialize(features);
  critic_outer.initialize(features);
  thrust_actor_inner.initialize();
  thrust_actor_outer.initialize();
  count_inner = 0;
  count_outer = 0;
  timestep = 0;
  // rcs_actor_cw.initialize();
  // rcs_actor_ccw.initialize();
  // std::cout << 1 << " " << (critic_forward.get_weights() - critic_backward.get_weights()).norm() << std::endl;
  return compute_action(rng, features);
}


cart_pole_simulator::action multi_gauss_2_agent::update(std::mt19937& rng, VectorXd state,
                                                          double reward, bool terminal, bool learn) {
  // std::cout << 2 << " " << (critic_forward.get_weights() - critic_backward.get_weights()).norm() << std::endl;
  timestep++;
  clip_state(state);
  VectorXi features = tc.indices(state);

  if (learn) {
    // double td_error;
    // if (rcs_cw) {
    //   td_error = critic_cw.evaluate(features, reward, terminal);
    // } else {
    //   td_error = critic_ccw.evaluate(features, reward, terminal);
    // }
    double td_error = td_critic::evaluate_multi(critic_inner, critic_outer, thrust_inner, features, reward,\
                                                terminal);
    thrust_actor_inner.learn(td_error);
    thrust_actor_outer.learn(td_error);
    // rcs_actor_cw.learn(td_error);
    // rcs_actor_ccw.learn(td_error);
  }
  // std::cout << 3 << " " << (critic_forward.get_weights() - critic_backward.get_weights()).norm() << std::endl;

  return compute_action(rng, features);
}
