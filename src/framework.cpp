#include <boost/math/constants/constants.hpp>
#include <random>
#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "framework.hpp"

template <typename AgentType>
double framework<AgentType>::reward() {
  double reward = 0;
  if (simulator.is_done()) {
    reward = -1;
  }
  return reward;
}


template <typename AgentType>
void framework<AgentType>::initialize_simulator(std::mt19937& rng) {
  simulator.initialize(rng);
}


template <typename AgentType>
VectorXd framework<AgentType>::simulator_state() const {
  return simulator.get_state();
}


template <typename AgentType>
void framework<AgentType>::take_action(cart_pole_simulator::action a) {
  simulator.set_action(a);
}


template <typename AgentType>
void framework<AgentType>::run_episode(std::mt19937& init_rng, std::mt19937& agent_rng, bool print_actions) {

  initialize_simulator(init_rng);
  take_action(agent.initialize(agent_rng, simulator_state()));

  _return = 0;
  rewards.clear();
  time_elapsed = 0;

  bool terminal = false;

  while (!terminal) {

    for (int i = 0; i < agent_time_steps && !terminal; ++i) {
      //send_visualiser_data();
      simulator.update(dt);
      time_elapsed += dt;
      terminal = simulator.is_done();
    }

    double _reward = reward();
    _return += _reward;
    rewards.push_back(_reward);

/*DEBUG*/
    cart_pole_simulator::action a = agent.update(agent_rng, simulator_state(), _reward, terminal);
    if (print_actions) {
      /*std::cout << agent.get_mu() << " " << agent.get_sigma() << " " << simulator.get_state()[0] << " " \
                  << simulator.get_state()[1] << " " << a.thrust << " " << agent.get_mu_grad() << " " \
                  << agent.get_sigma_grad() << " " << agent.get_td_error() << " " \
                  << agent.get_forward_critic_value() << " " << agent.get_backward_critic_value() << std::endl;*/
      std::cout << simulator.get_state()[0] << " " << simulator.get_state()[1] << " " << simulator.get_state()[2]\
                  << " " << simulator.get_state()[3] << " " << a.thrust << std::endl;
    }
    take_action(a);
/*END DEBUG*/

    // take_action(agent.update(agent_rng, simulator_state(), _reward, terminal));
    if (time_elapsed >= max_time) {
      terminal = true; //TODO: Maybe this should throw an error and have the main file exit
    }
  }

  //send_visualiser_data();
  //if (visualiser) std::fflush (visualiser);
}


template <typename AgentType>
int framework<AgentType>::run_episodes(std::mt19937& init_rng, std::mt19937& agent_rng, bool print_actions) {

  initialize_simulator(init_rng);
  take_action(agent.initialize(agent_rng, simulator_state()));

  _return = 0;
  rewards.clear();
  time_elapsed = 0;
  double prev_time = 0;
  int fails = 0;

  bool terminal = false;

  while (time_elapsed < max_time) {

    for (int i = 0; i < agent_time_steps /*&& !terminal*/; ++i) {
      //send_visualiser_data();
      simulator.update(dt);
      time_elapsed += dt;
      terminal = simulator.is_done();
    }

    double _reward = reward();
    _return += _reward;
    rewards.push_back(_reward);

    if (terminal) {
      initialize_simulator(init_rng);
      if (print_actions) {
        std::cout << time_elapsed-prev_time << std::endl;
      }
      fails++;
      prev_time = time_elapsed;
    }

/*DEBUG*/
    cart_pole_simulator::action a = agent.update(agent_rng, simulator_state(), _reward, terminal);
    // if (print_actions) {
      /*std::cout << agent.get_mu() << " " << agent.get_sigma() << " " << simulator.get_state()[0] << " " \
                  << simulator.get_state()[1] << " " << a.thrust << " " << agent.get_mu_grad() << " " \
                  << agent.get_sigma_grad() << " " << agent.get_td_error() << " " \
                  << agent.get_forward_critic_value() << " " << agent.get_backward_critic_value() << std::endl;*/
      /*std::cout << simulator.get_state()[0] << " " << simulator.get_state()[1] << " " << simulator.get_state()[2]\
                  << " " << simulator.get_state()[3] << " " << a.thrust << std::endl;*/
    // }
    take_action(a);
/*END DEBUG*/

    // take_action(agent.update(agent_rng, simulator_state(), _reward, terminal));
    if (time_elapsed >= max_time) {
      terminal = true; //TODO: Maybe this should throw an error and have the main file exit
    }
  }
  if (print_actions) {
    std::cout << ">" << time_elapsed-prev_time << std::endl; //print duration of last attempt
  }
  //send_visualiser_data();
  //if (visualiser) std::fflush (visualiser);
  return fails;
}


// template <typename AgentType>
// void framework<AgentType>::send_visualiser_data () const {
//
//   if (!visualiser) return;
//
//   const Vector2d& pos = simulator.get_lander().get_pos();
//   const Vector2d& vel = simulator.get_lander().get_vel();
//
//   std::fprintf (visualiser,
//                 "{ \"Return\": %g, \"x\": %g, \"y\": %g, \"vx\": %g, \"vy\": %g, \"rot\": %g, \"vrot\": %g, "
//                 "\"thrust\": %g, \"rcs\": %g, \"breakage\": %g, \"crashed\": %s, \"landed\": %s }\n",
//                 get_return(), pos.x(), pos.y(), vel.x(), vel.y(),
//                 simulator.get_lander().get_rot(), simulator.get_lander().get_rot_vel(),
//                 simulator.get_action().thrust, simulator.get_action().rcs,
//                 simulator.get_lander().get_breakage(),
//                 simulator.get_crashed() ? "true" : "false",
//                 simulator.get_landed() ? "true" : "false");
// }
