#include "multi_gauss_agent.hpp"

#include <limits>
#include <iostream>

using Eigen::VectorXd;

tile_coder multi_gauss_agent::make_tile_coder(double tile_weight_exponent, const std::vector<int>& subspaces) {

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


policy_gradient_actor multi_gauss_agent::make_thrust_actor_forward
(const tile_coder_base& tc, double lambda, double alpha, bool trunc_normal) {
  const double max_thrust = cart_pole_simulator::MAX_THRUST();
  return policy_gradient_actor (tc, lambda, alpha,
                                // 0.0, max_thrust, // min and max action
                                -max_thrust, max_thrust, // min and max action
                                max_thrust/64, max_thrust/4, // min and max sigma
                                0.5, max_thrust/8, // initial mu and sigma
                                1.0, // gamma
                                trunc_normal);
}


policy_gradient_actor multi_gauss_agent::make_thrust_actor_backward
(const tile_coder_base& tc, double lambda, double alpha, bool trunc_normal) {
  const double max_thrust = cart_pole_simulator::MAX_THRUST();
  return policy_gradient_actor (tc, lambda, alpha,
                                // -max_thrust, 0.0, // min and max action
                                -max_thrust, max_thrust, // min and max action
                                max_thrust/64, max_thrust/4, // min and max sigma
                                -0.5, max_thrust/8, // initial mu and sigma
                                1.0, // gamma
                                trunc_normal);
}


// policy_gradient_actor multi_gauss_agent::make_rcs_actor_cw
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
// policy_gradient_actor multi_gauss_agent::make_rcs_actor_ccw
// (const tile_coder_base& tc, double lambda, double alpha, bool trunc_normal) {
//   const double max_rcs = cart_pole_simulator::MAX_RCS();
//   return policy_gradient_actor (tc, lambda, alpha,
//                                 -max_rcs, 0, // min and max action
//                                 max_rcs/32, max_rcs, // min and max sigma
//                                 0.0, max_rcs/4, // initial mu and sigma
//                                 1.0, // gamma
//                                 trunc_normal);
// }


cart_pole_simulator::action multi_gauss_agent::compute_action(std::mt19937& rng, const VectorXi& features) {
  // std::cout << 4 << " " << (critic_forward.get_weights() - critic_backward.get_weights()).norm() << std::endl;
  last_critic_value_forward = critic_forward.get_value();
  last_critic_value_backward = critic_backward.get_value();
  if (weighted_dist_choice) {
    thrust_forward = ((double)rng2()/rng2.max() < critic_forward.get_value()/\
                        (critic_forward.get_value() + critic_backward.get_value()));
  } else if (epsilon > 0 && (double)rng2()/rng2.max() < epsilon) {
    thrust_forward = ((double)rng2()/rng2.max() < 0.5);
  } else {
    //thrust_forward = ((double)rng2()/rng2.max() < 0.5);
    // last_critic_value_forward = critic_forward.get_value();
    // last_critic_value_backward = critic_backward.get_value();
    int ntf = 0;
    int ntb = 0;
    if (ucb_factor != 0) {
      for (int i = 0; i < features.size(); i++) {
        ntf += num_forward[features[i]];
        ntb += num_backward[features[i]];
      }
    }
    thrust_forward = (critic_forward.get_value() + ucb_factor*sqrt(log(timestep)/ntf) >\
                     critic_backward.get_value() + ucb_factor*sqrt(log(timestep)/ntb));
  }
  if (thrust_forward) {
    count_forward++;
    if (ucb_factor != 0) {
      for (int i = 0; i < features.size(); i++) {
        num_forward[features[i]]++;
      }
    }
  } else {
    count_backward++;
    if (ucb_factor != 0) {
      for (int i = 0; i < features.size(); i++) {
        num_backward[features[i]]++;
      }
    }
  }
  // std::cout << 5 << " " << (critic_forward.get_weights() - critic_backward.get_weights()).norm() << std::endl;
  return cart_pole_simulator::action(thrust_forward ? thrust_actor_forward.act(rng, features) :\
                                                          thrust_actor_backward.act(rng, features));
}


cart_pole_simulator::action multi_gauss_agent::initialize(std::mt19937& rng, VectorXd state) {
  clip_state(state);
  VectorXi features = tc.indices(state);

  critic_forward.initialize(features);
  critic_backward.initialize(features);
  thrust_actor_forward.initialize();
  thrust_actor_backward.initialize();
  count_forward = 0;
  count_backward = 0;
  timestep = 0;
  // rcs_actor_cw.initialize();
  // rcs_actor_ccw.initialize();
  // std::cout << 1 << " " << (critic_forward.get_weights() - critic_backward.get_weights()).norm() << std::endl;
  return compute_action(rng, features);
}


cart_pole_simulator::action multi_gauss_agent::update(std::mt19937& rng, VectorXd state,
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
    double td_error = td_critic::evaluate_multi(critic_forward, critic_backward, thrust_forward, features, reward,\
                                                terminal);
    thrust_actor_forward.learn(td_error);
    thrust_actor_backward.learn(td_error);
    // rcs_actor_cw.learn(td_error);
    // rcs_actor_ccw.learn(td_error);
  }
  // std::cout << 3 << " " << (critic_forward.get_weights() - critic_backward.get_weights()).norm() << std::endl;

  return compute_action(rng, features);
}
