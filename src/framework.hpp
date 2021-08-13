#ifndef _FRAMEWORK_HPP
#define _FRAMEWORK_HPP

#include <random>
#include <Eigen/Core>
#include <vector>
#include <cstdio>

#include "simulator.hpp"
#include "lunar_lander_agent.hpp"
#include "test_agent.hpp"
#include "multi_dist_agent.hpp"

using Eigen::VectorXd;

template <typename AgentType>
struct framework {

  lunar_lander_simulator simulator;
// #ifdef USE_MULTI_DIST
//   multi_dist_agent agent;
// #elif defined(USE_TEST_DIST)
//   test_agent agent;
// #else
//   lunar_lander_agent agent;
// #endif
  AgentType agent;
  double dt, time_elapsed, _return;
  std::vector<double> rewards;
  int agent_time_steps;

  FILE* visualiser;

public:

  framework(const lunar_lander_simulator& simulator,
// #ifdef USE_MULTI_DIST
//   const multi_dist_agent& agent,
// #elif defined(USE_TEST_DIST)
//   const test_agent& agent,
// #else
//   const lunar_lander_agent& agent,
// #endif
  const AgentType& agent,
  double dt, int agent_time_steps)
    : simulator(simulator), agent(agent), dt(dt), agent_time_steps(agent_time_steps), visualiser(nullptr) { }

  double reward();

  void initialize_simulator(std::mt19937& rng);

  VectorXd simulator_state() const;

  void take_action(lunar_lander_simulator::action a);

  void run_episode(std::mt19937& init_rng, std::mt19937& agent_rng);

  double get_return () const { return _return; }
  const std::vector<double>& get_rewards () const { return rewards; }

  void set_visualiser (FILE* output) { visualiser = output; }
  void send_visualiser_data () const;
};

#endif
