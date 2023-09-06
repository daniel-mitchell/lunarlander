#ifndef _FRAMEWORK_HPP
#define _FRAMEWORK_HPP

#include <random>
#include <Eigen/Core>
#include <vector>
#include <cstdio>

#include "simulator.hpp"
#include "gauss_agent.hpp"
#include "beta_agent.hpp"
#include "multi_gauss_agent.hpp"
#include "multi_gauss_4_agent.hpp"
#include "multi_beta_agent.hpp"

using Eigen::VectorXd;

template <typename AgentType>
struct framework {

  mountain_car_simulator simulator;
  AgentType agent;
  double dt, time_elapsed, max_time, _return;
  std::vector<double> rewards;
  int agent_time_steps;

  // FILE* visualiser;

public:

  framework(const mountain_car_simulator& simulator, const AgentType& agent, double dt, double max_time,\
      int agent_time_steps)
    : simulator(simulator), agent(agent), dt(dt), max_time(max_time), agent_time_steps(agent_time_steps) { }

  double reward();

  void initialize_simulator(std::mt19937& rng);

  VectorXd simulator_state() const;

  void take_action(mountain_car_simulator::action a);

  void run_episode(std::mt19937& init_rng, std::mt19937& agent_rng, bool print_actions);
  double run_episodes(std::mt19937& init_rng, std::mt19937& agent_rng, bool print_actions);

  double get_return () const { return _return; }
  double get_time_elapsed () const { return time_elapsed; }
  const std::vector<double>& get_rewards () const { return rewards; }

  double get_direction_ratio() { return agent.get_direction_ratio(); }
  int get_forward_count() { return agent.get_forward_count(); }
  int get_backward_count() { return agent.get_backward_count(); }

  // void set_visualiser (FILE* output) { visualiser = output; }
  // void send_visualiser_data () const;
};

#endif
