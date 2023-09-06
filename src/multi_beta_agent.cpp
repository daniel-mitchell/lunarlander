#include "multi_beta_agent.hpp"

#include <limits>


using Eigen::VectorXd;

tile_coder multi_beta_agent::make_tile_coder(double tile_weight_exponent, const std::vector<int>& subspaces) {

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


beta_pg_actor multi_beta_agent::make_thrust_actor_forward
(const tile_coder_base& tc, double lambda, double alpha) {
  const double max_thrust = cart_pole_simulator::MAX_THRUST();
  return beta_pg_actor (tc, lambda, alpha,
                                0.0, max_thrust, // min and max action
                                0, 0, // initial alpha and beta
                                1.0); // gamma
}


beta_pg_actor multi_beta_agent::make_thrust_actor_backward
(const tile_coder_base& tc, double lambda, double alpha) {
  const double max_thrust = cart_pole_simulator::MAX_THRUST();
  return beta_pg_actor (tc, lambda, alpha,
                                -max_thrust, 0, // min and max action
                                0, 0, // initial alpha and beta
                                1.0); // gamma
}


// test_pg_actor multi_beta_agent::make_rcs_actor_cw
// (const tile_coder_base& tc, double lambda, double alpha) {
//   const double max_rcs = lunar_lander_simulator::MAX_RCS();
//   return test_pg_actor (tc, lambda, alpha,
//                                 0, max_rcs, // min and max action
//                                 0, 0, // initial alpha and beta
//                                 1.0); // gamma
// }
//
//
// test_pg_actor multi_beta_agent::make_rcs_actor_ccw
// (const tile_coder_base& tc, double lambda, double alpha) {
//   const double max_rcs = lunar_lander_simulator::MAX_RCS();
//   return test_pg_actor (tc, lambda, alpha,
//                                 -max_rcs, 0, // min and max action
//                                 0, 0, // initial alpha and beta
//                                 1.0); // gamma
// }


cart_pole_simulator::action multi_beta_agent::compute_action(std::mt19937& rng, const VectorXi& features) {
  if (epsilon > 0 && rng2()/rng2.max() < epsilon) {
    thrust_forward = (rng2()/rng2.max() < 0.5);
  } else {
    thrust_forward = (critic_forward.get_value() > critic_backward.get_value());
  }
  return cart_pole_simulator::action(thrust_forward ? thrust_actor_forward.act(rng, features) :
                                                        thrust_actor_backward.act(rng, features));
}


cart_pole_simulator::action multi_beta_agent::initialize(std::mt19937& rng, VectorXd state) {
  clip_state(state);
  VectorXi features = tc.indices(state);

  critic_forward.initialize(features);
  critic_backward.initialize(features);
  thrust_actor_forward.initialize();
  thrust_actor_backward.initialize();

  return compute_action(rng, features);
}


cart_pole_simulator::action multi_beta_agent::update(std::mt19937& rng, VectorXd state,
                                                          double reward, bool terminal, bool learn) {
  clip_state(state);
  VectorXi features = tc.indices(state);

  if (learn) {
    // double td_error;
    // if (thrust_forward) {
    //   td_error = critic_forward.evaluate(features, reward, terminal);
    // } else {
    //   td_error = critic_backward.evaluate(features, reward, terminal);
    // }
    double td_error = beta_td_critic::evaluate_multi(critic_forward, critic_backward, thrust_forward, features,\
                                                     reward, terminal);
    thrust_actor_forward.learn(td_error);
    thrust_actor_backward.learn(td_error);
    // rcs_actor_cw.learn(td_error);
    // rcs_actor_ccw.learn(td_error);
  }

  return compute_action(rng, features);
}
