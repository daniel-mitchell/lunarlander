#include "multi_gauss_3_agent.hpp"

#include <limits>
#include <iostream>
#include <boost/math/special_functions/sign.hpp>

using Eigen::VectorXd;

tile_coder multi_gauss_3_agent::make_tile_coder(double tile_weight_exponent, const std::vector<int>& subspaces) {

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


std::vector<policy_gradient_actor> multi_gauss_3_agent::make_thrust_actors(policy_gradient_actor actor1,\
                                        policy_gradient_actor actor2, policy_gradient_actor actor3) {
  std::vector<policy_gradient_actor> actor_vec;
  actor_vec.push_back(actor1);
  actor_vec.push_back(actor2);
  actor_vec.push_back(actor3);
  return actor_vec;
}


policy_gradient_actor multi_gauss_3_agent::make_thrust_actor_left
(const tile_coder_base& tc, double lambda, double alpha, bool trunc_normal) {
  const double max_thrust = cart_pole_simulator::MAX_THRUST();
  return policy_gradient_actor (tc, lambda, alpha,
                                -max_thrust, -max_thrust/3, // min and max action
                                max_thrust/64, max_thrust/4, // min and max sigma
                                -2*max_thrust/3, max_thrust/8, // initial mu and sigma
                                1.0, // gamma
                                trunc_normal);
}


policy_gradient_actor multi_gauss_3_agent::make_thrust_actor_middle
(const tile_coder_base& tc, double lambda, double alpha, bool trunc_normal) {
  const double max_thrust = cart_pole_simulator::MAX_THRUST();
  return policy_gradient_actor (tc, lambda, alpha,
                                -max_thrust/3, max_thrust/3, // min and max action
                                max_thrust/64, max_thrust/4, // min and max sigma
                                0, max_thrust/8, // initial mu and sigma
                                1.0, // gamma
                                trunc_normal);
}


policy_gradient_actor multi_gauss_3_agent::make_thrust_actor_right
(const tile_coder_base& tc, double lambda, double alpha, bool trunc_normal) {
  const double max_thrust = cart_pole_simulator::MAX_THRUST();
  return policy_gradient_actor (tc, lambda, alpha,
                                max_thrust/3, max_thrust, // min and max action
                                max_thrust/64, max_thrust/4, // min and max sigma
                                2*max_thrust/3, max_thrust/8, // initial mu and sigma
                                1.0, // gamma
                                trunc_normal);
}


// policy_gradient_actor multi_gauss_3_agent::make_rcs_actor_cw
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
// policy_gradient_actor multi_gauss_3_agent::make_rcs_actor_ccw
// (const tile_coder_base& tc, double lambda, double alpha, bool trunc_normal) {
//   const double max_rcs = cart_pole_simulator::MAX_RCS();
//   return policy_gradient_actor (tc, lambda, alpha,
//                                 -max_rcs, 0, // min and max action
//                                 max_rcs/32, max_rcs, // min and max sigma
//                                 0.0, max_rcs/4, // initial mu and sigma
//                                 1.0, // gamma
//                                 trunc_normal);
// }


cart_pole_simulator::action multi_gauss_3_agent::compute_action(std::mt19937& rng, const VectorXi& features) {
  // std::cout << 4 << " " << (critic_forward.get_weights() - critic_backward.get_weights()).norm() << std::endl;
  last_critic_value_left = critic_left.get_value();
  last_critic_value_middle = critic_middle.get_value();
  last_critic_value_right = critic_right.get_value();
  if (weighted_dist_choice) {
    double value_sum = critic_left.get_value() + critic_middle.get_value() + critic_right.get_value();
    double break1 = critic_left.get_value()/value_sum;
    double break2 = critic_middle.get_value()/value_sum + break1;
    double sample = (double)rng2()/rng2.max();
    if (sample < break1) {
      chosen_thrust = 0;
    } else if (sample < break2) {
      chosen_thrust = 1;
    } else {
      chosen_thrust = 2;
    }
  } else if (epsilon > 0 && (double)rng2()/rng2.max() < epsilon) {
    std::uniform_int_distribution<> distr(0, 2);
    chosen_thrust = distr(rng2);
  } else {
    int ntl = 0;
    int ntm = 0;
    int ntr = 0;
    if (ucb_factor != 0) {
      for (int i = 0; i < features.size(); i++) {
        ntl += num_left[features[i]];
        ntm += num_middle[features[i]];
        ntr += num_right[features[i]];
      }
    }
    double left_value = critic_left.get_value() + ucb_factor*sqrt(log(timestep)/ntl);
    double middle_value = critic_middle.get_value() + ucb_factor*sqrt(log(timestep)/ntm);
    double right_value = critic_right.get_value() + ucb_factor*sqrt(log(timestep)/ntr);
    if (left_value > std::max(middle_value, right_value)) {
      chosen_thrust = 0;
    } else if (middle_value > right_value) {
      chosen_thrust = 1;
    } else {
      chosen_thrust = 2;
    }
  }
  if (chosen_thrust == 2) {
    count_right++;
    if (ucb_factor != 0) {
      for (int i = 0; i < features.size(); i++) {
        num_right[features[i]]++;
      }
    }
  } else if (chosen_thrust == 1) {
    count_middle++;
    if (ucb_factor != 0) {
      for (int i = 0; i < features.size(); i++) {
        num_middle[features[i]]++;
      }
    }
  } else {
    count_left++;
    if (ucb_factor != 0) {
      for (int i = 0; i < features.size(); i++) {
        num_left[features[i]]++;
      }
    }
  }
  // std::cout << 5 << " " << (critic_forward.get_weights() - critic_backward.get_weights()).norm() << std::endl;
  return cart_pole_simulator::action(thrust_actors[chosen_thrust].act(rng, features));
}


cart_pole_simulator::action multi_gauss_3_agent::initialize(std::mt19937& rng, VectorXd state) {
  clip_state(state);
  VectorXi features = tc.indices(state);

  critic_left.initialize(features);
  critic_middle.initialize(features);
  critic_right.initialize(features);
  thrust_actors[0].initialize();
  thrust_actors[1].initialize();
  thrust_actors[2].initialize();
  count_left = 0;
  count_middle = 0;
  count_right = 0;
  timestep = 0;
  // rcs_actor_cw.initialize();
  // rcs_actor_ccw.initialize();
  // std::cout << 1 << " " << (critic_forward.get_weights() - critic_backward.get_weights()).norm() << std::endl;
  return compute_action(rng, features);
}


cart_pole_simulator::action multi_gauss_3_agent::update(std::mt19937& rng, VectorXd state,
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
    double td_error = td_critic::evaluate_multi3(critic_left, critic_middle, critic_right, chosen_thrust, features,\
                                                reward, terminal);
    thrust_actors[0].learn(td_error);
    thrust_actors[1].learn(td_error);
    thrust_actors[2].learn(td_error);
    // rcs_actor_cw.learn(td_error);
    // rcs_actor_ccw.learn(td_error);
  }
  // std::cout << 3 << " " << (critic_forward.get_weights() - critic_backward.get_weights()).norm() << std::endl;

  return compute_action(rng, features);
}
