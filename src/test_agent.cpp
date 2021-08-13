#include "test_agent.hpp"

#include <limits>


using Eigen::VectorXd;

tile_coder test_agent::make_tile_coder(double tile_weight_exponent, const std::vector<int>& subspaces) {

  const double PI = boost::math::constants::pi<double>();

  struct {
    bool state_signed, state_bounded;
    double tile_size;
    unsigned int num_tiles, num_offsets;
  } cfg[] =
      { // #signed, bounded, tile_size, num_tiles, num_offsets
        {    false,    true,       5.0,         6,           2 }, // xpos
        {    false,    true,       5.0,         4,           2 }, // ypos
        {     true,    true,       2.0,         4,           4 }, // xvel
        {     true,    true,       2.0,         4,           4 }, // yvel
        {     true,   false,      PI/2,         2,           8 }, // rot
        {     true,    true,      PI/6,         3,           4 }  // rotvel
      };

  const unsigned int state_dim = 6;

  VectorXd tile_size (6);
  VectorXi num_tiles (6);
  VectorXi num_offsets (6);

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


test_pg_actor test_agent::make_thrust_actor
(const tile_coder_base& tc, double lambda, double alpha) {
  const double max_thrust = lunar_lander_simulator::MAX_THRUST();
  return test_pg_actor (tc, lambda, alpha,
                                0.0, max_thrust, // min and max action
                                0, 0, // initial alpha and beta
                                1.0); // gamma
}


test_pg_actor test_agent::make_rcs_actor_cw
(const tile_coder_base& tc, double lambda, double alpha) {
  const double max_rcs = lunar_lander_simulator::MAX_RCS();
  return test_pg_actor (tc, lambda, alpha,
                                0, max_rcs, // min and max action
                                0, 0, // initial alpha and beta
                                1.0); // gamma
}


test_pg_actor test_agent::make_rcs_actor_ccw
(const tile_coder_base& tc, double lambda, double alpha) {
  const double max_rcs = lunar_lander_simulator::MAX_RCS();
  return test_pg_actor (tc, lambda, alpha,
                                -max_rcs, 0, // min and max action
                                0, 0, // initial alpha and beta
                                1.0); // gamma
}


lunar_lander_simulator::action test_agent::compute_action(std::mt19937& rng, const VectorXi& features) {
  if (epsilon > 0 && rng2()/rng2.max() < epsilon) {
    rcs_cw = (rng2()/rng2.max() < 0.5);
  } else {
    rcs_cw = (critic_cw.get_value() > critic_ccw.get_value());
  }
  return lunar_lander_simulator::action(thrust_actor.act(rng, features), 
          rcs_cw ? rcs_actor_cw.act(rng, features) : rcs_actor_ccw.act(rng, features));
}


lunar_lander_simulator::action test_agent::initialize(std::mt19937& rng, VectorXd state) {
  clip_state(state);
  VectorXi features = tc.indices(state);

  critic_cw.initialize(features);
  critic_ccw.initialize(features);
  thrust_actor.initialize();
  rcs_actor_cw.initialize();
  rcs_actor_ccw.initialize();

  return compute_action(rng, features);
}


lunar_lander_simulator::action test_agent::update(std::mt19937& rng, VectorXd state,
                                                          double reward, bool terminal, bool learn) {
  clip_state(state);
  VectorXi features = tc.indices(state);

  if (learn) {
    double td_error;
    if (rcs_cw) {
      td_error = critic_cw.evaluate(features, reward, terminal);
    } else {
      td_error = critic_ccw.evaluate(features, reward, terminal);
    }
    thrust_actor.learn(td_error);
    rcs_actor_cw.learn(td_error);
    rcs_actor_ccw.learn(td_error);
  }

  return compute_action(rng, features);
}
