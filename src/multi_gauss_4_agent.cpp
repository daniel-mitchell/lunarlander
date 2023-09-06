#include "multi_gauss_4_agent.hpp"

#include <limits>
#include <iostream>

using Eigen::VectorXd;

tile_coder multi_gauss_4_agent::make_tile_coder(double tile_weight_exponent, const std::vector<int>& subspaces) {

  // const double PI = boost::math::constants::pi<double>();

  struct {
    bool state_signed, state_bounded;
    double tile_size;
    unsigned int num_tiles, num_offsets;
  } cfg[] =
      { // #signed, bounded, tile_size, num_tiles, num_offsets
        // {     true,    true,      0.96,        10,          10 }, //position
        // {     true,   false,       0.4,        10,          10 }, //velocity
        {    false,    true,      0.18,        10,          16 }, //position
        {     true,    true,     0.014,        10,          16 }  //velocity
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

  const unsigned int state_dim = 2;

  VectorXd tile_size (2);
  VectorXi num_tiles (2);
  VectorXi num_offsets (2);

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


policy_gradient_actor multi_gauss_4_agent::make_thrust_actor_forward
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


policy_gradient_actor multi_gauss_4_agent::make_thrust_actor_backward
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


// policy_gradient_actor multi_gauss_4_agent::make_rcs_actor_cw
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
// policy_gradient_actor multi_gauss_4_agent::make_rcs_actor_ccw
// (const tile_coder_base& tc, double lambda, double alpha, bool trunc_normal) {
//   const double max_rcs = cart_pole_simulator::MAX_RCS();
//   return policy_gradient_actor (tc, lambda, alpha,
//                                 -max_rcs, 0, // min and max action
//                                 max_rcs/32, max_rcs, // min and max sigma
//                                 0.0, max_rcs/4, // initial mu and sigma
//                                 1.0, // gamma
//                                 trunc_normal);
// }


VectorXi multi_gauss_4_agent::get_state_action_features(const VectorXi& features, bool action) {
  if (action) {
    //VectorXi ret = features.array() += tc.get_num_features();
    return features.array() + tc.get_num_features();
  } else {
    return features;
  }
}


Eigen::Vector2d multi_gauss_4_agent::get_policy_probs(const VectorXi& features) {
  // double exp_backward = exp(parameters.dot(get_state_action_features(features, false)));
  double pow_backward = 0;
  for (auto i: get_state_action_features(features, false)) {
    pow_backward += parameters[i];
  }
  // double exp_forward = exp(parameters.dot(get_state_action_features(features, true)));
  double pow_forward = 0;
  for (auto i: get_state_action_features(features, true)) {
    pow_forward += parameters[i];
  }
  double exp_backward = exp(pow_backward);
  double exp_forward = exp(pow_forward);
  Eigen::Vector2d ret;
  ret << exp_backward/(exp_backward + exp_forward), exp_forward/(exp_backward + exp_forward);
  return ret;
}


cart_pole_simulator::action multi_gauss_4_agent::compute_action([[maybe_unused]] std::mt19937& rng,\
                                                                    const VectorXi& features) {
  Eigen::Vector2d probs = get_policy_probs(features);
  thrust_forward = true;
  if ((double)rng2()/rng2.max() < probs[0]) {
    thrust_forward = false;
  }
/*  // std::cout << 4 << " " << (critic_forward.get_weights() - critic_backward.get_weights()).norm() << std::endl;
  // last_critic_value_forward = critic_forward.get_value();
  // last_critic_value_backward = critic_backward.get_value();
  VectorXi sa_features = get_state_action_features(features);
  last_critic_value_forward = critic_forward(features.array()).sum();
  last_critic_value_backward = critic_backward(features.array()).sum();
  if (weighted_dist_choice) {
    thrust_forward = ((double)rng2()/rng2.max() < last_critic_value_forward/\
                        (last_critic_value_forward + last_critic_value_backward));
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
    thrust_forward = (last_critic_value_forward + ucb_factor*sqrt(log(timestep)/ntf) >\
                     last_critic_value_backward + ucb_factor*sqrt(log(timestep)/ntb));
  }*/
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
  // return cart_pole_simulator::action(thrust_forward ? 1 : -1);
  return cart_pole_simulator::action(thrust_forward ? thrust_actor_forward.act(rng, features) :\
                                                          thrust_actor_backward.act(rng, features));
}


cart_pole_simulator::action multi_gauss_4_agent::initialize(std::mt19937& rng, VectorXd state) {
  clip_state(state);
  VectorXi features = tc.indices(state);

  critic_forward.initialize(features);
  critic_backward.initialize(features);
  thrust_actor_forward.initialize();
  thrust_actor_backward.initialize();
  count_forward = 0;
  count_backward = 0;
  ii = 1;
  timestep = 0;
  parameter_trace = VectorXd::Zero(2*tc.get_num_features());
  weight_trace = VectorXd::Zero(tc.get_num_features());
  // rcs_actor_cw.initialize();
  // rcs_actor_ccw.initialize();
  // std::cout << 1 << " " << (critic_forward.get_weights() - critic_backward.get_weights()).norm() << std::endl;
  oldFeatures = features;
  return compute_action(rng, features);
}


cart_pole_simulator::action multi_gauss_4_agent::update(std::mt19937& rng, VectorXd state,
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
    /*double td_error = td_critic::evaluate_multi(critic_forward, critic_backward, thrust_forward, features, reward,\
                                                terminal);*/
    /*double old_value = thrust_forward ? critic_forward(oldFeatures.array()).sum() :\
                                         critic_backward(oldFeatures.array()).sum();
    double new_value =  terminal ? 0.0 :  std::max(critic_forward(features.array()).sum(),\
                                         critic_backward(features.array()).sum());*/
    Eigen::Vector2d probs = get_policy_probs(features);
    VectorXi sa_features_backward = get_state_action_features(features, false);
    VectorXi sa_features_forward = get_state_action_features(features, true);
    double td_error = reward;// + gamma*new_value - old_value;
    if (continuing) {
      td_error -= rBar;
    }
    parameter_trace *= gamma*lambda;
    weight_trace *= gamma*lambda;
    for (auto i: oldFeatures) {
      td_error -= weights[i];
      weight_trace[i] += 1;
    }
    for (auto i: features) {
      if (!terminal) {
        td_error += gamma*weights[i]; //TODO: still keep this check for continuing?
      }
    }
    if (continuing) {
      rBar += alpha_r*td_error;
    }
    for (auto i: sa_features_backward) {
      if (!thrust_forward) {
        parameter_trace[i] += ii;
      }
      parameter_trace[i] -= ii*probs[0];
    }
    for (auto i: sa_features_forward) {
      if (thrust_forward) {
        parameter_trace[i] += ii;
      }
      parameter_trace[i] -= ii*probs[1];
    }
    // if (thrust_forward) {
    //   critic_forward += alpha_v/features.size()*td_error*trace;
    // } else {
    //   critic_backward += alpha_v/features.size()*td_error*trace;
    // }
    weights += alpha_u*td_error*weight_trace;
    parameters += alpha_v*td_error*parameter_trace;
    if (!continuing) {
      ii *= gamma;
    }
    // trace *=gamma*lambda;
    double td_error2 = td_critic::evaluate_multi(critic_forward, critic_backward, thrust_forward, features, reward,\
                                                terminal);
    thrust_actor_forward.learn(td_error2);
    thrust_actor_backward.learn(td_error2);
    // rcs_actor_cw.learn(td_error);
    // rcs_actor_ccw.learn(td_error);
  }
  // std::cout << 3 << " " << (critic_forward.get_weights() - critic_backward.get_weights()).norm() << std::endl;
  oldFeatures = features;
  return compute_action(rng, features);
}
