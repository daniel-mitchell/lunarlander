#include "beta_agent.hpp"

#include <limits>


using Eigen::VectorXd;

tile_coder beta_agent::make_tile_coder(double tile_weight_exponent, const std::vector<int>& subspaces) {

  // const double PI = boost::math::constants::pi<double>();

  struct {
    bool state_signed, state_bounded;
    double tile_size;
    unsigned int num_tiles, num_offsets;
  } cfg[] =
      { // #signed, bounded, tile_size, num_tiles, num_offsets
        {    false,    true,      0.18,        10,          10 }, //position
        {     true,    true,     0.014,        10,          10 }  //velocity
      };
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


beta_pg_actor beta_agent::make_thrust_actor
(const tile_coder_base& tc, double lambda, double alpha) {
  const double max_thrust = mountain_car_simulator::MAX_THRUST();
  return beta_pg_actor (tc, lambda, alpha,
                                -max_thrust, max_thrust, // min and max action
                                0, 0, // initial alpha and beta
                                1.0); // gamma
}


// test_pg_actor beta_agent::make_rcs_actor
// (const tile_coder_base& tc, double lambda, double alpha) {
//   const double max_rcs = mountain_car_simulator::MAX_RCS();
//   return test_pg_actor (tc, lambda, alpha,
//                                 -max_rcs, max_rcs, // min and max action
//                                 0, 0, // initial alpha and beta
//                                 1.0); // gamma
// }


mountain_car_simulator::action beta_agent::initialize(std::mt19937& rng, VectorXd state) {
  clip_state(state);
  VectorXi features = tc.indices(state);

  critic.initialize(features);
  thrust_actor.initialize();
  // rcs_actor.initialize();

  return compute_action(rng, features);
}


mountain_car_simulator::action beta_agent::update(std::mt19937& rng, VectorXd state,
                                                          double reward, bool terminal, bool learn) {
  clip_state(state);
  VectorXi features = tc.indices(state);

  if (learn) {
    double td_error = critic.evaluate(features, reward, terminal);
    thrust_actor.learn(td_error);
    // rcs_actor.learn(td_error);
  }

  return compute_action(rng, features);
}
