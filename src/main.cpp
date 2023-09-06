#include <numeric>
#include <random>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <string>
#include <sstream>
#include <iterator>
#include <regex>
#include <iostream>

#include "utility.hpp"
#include "simulator.hpp"
#include "gauss_agent.hpp"
#include "beta_agent.hpp"
#include "multi_beta_agent.hpp"
#include "multi_gauss_agent.hpp"
#include "multi_gauss_2_agent.hpp"
#include "multi_gauss_3_agent.hpp"
#include "multi_gauss_4_agent.hpp"
#include "sarsa_agent.hpp"
// #include "multi_dist_agent.hpp"
#include "framework.cpp"

double scaled_alpha (double lambda, double alpha_factor) {
  return alpha_factor * (1 - lambda);
}


int main (int argc, char* argv[]) {

  std::random_device rdev;

  unsigned int agent_seed = rdev();
  unsigned int init_seed = rdev();


  double dt = 0.02;
  double max_time = 200;
  int agent_time_steps = 1;

  int num_episodes = 1000;
  double lambda = 0.9375;//0.75;
  double lambda_c = 0.9375;//0.75;
  double alpha_v = 0.1;
  double alpha_u = 0.001;//0.1;
  double alpha_vv = 0.1;
  double alpha_uu = 0.001;//0.1;
  double alpha_r = 0.1;
  double epsilon = 0;
  double gamma = 1;
  double ucb_factor = 0;
  bool weighted_dist_choice = false;
  double initial_value = 0;//1;
  int num_features = 1<<20;
  double tile_weight_exponent = 0.5; // 1 for no weighting
  bool trunc_normal = false;
  bool print_timesteps = false;
  bool print_rewards = true;
  bool print_episodes = true;
  bool print_direction_ratios = false;
  bool continuing = false;

  std::vector<int> subspaces { 0, 1, 2, 4 };

  // bool visualize = false;
  // int visualize_from = 0;
  // int visualize_every = 1000;

  enum AgentType {GAUSS=0, BETA=1, MULTIGAUSS=2, MULTIBETA=3, MULTIGAUSS2=4, MULTIGAUSS3=5, SARSA=6, MULTIGAUSS4=7}\
                   agentType = GAUSS;

  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--agent-seed") agent_seed = std::strtoul(argv[++i], nullptr, 10);
    else if (std::string(argv[i]) == "--agent-type") agentType = AgentType(std::atoi(argv[++i]));
    else if (std::string(argv[i]) == "--init-seed") init_seed = std::strtoul(argv[++i], nullptr, 10);
    else if (std::string(argv[i]) == "--dt") dt = std::atof(argv[++i]);
    else if (std::string(argv[i]) == "--max-time") max_time = std::atof(argv[++i]);
    else if (std::string(argv[i]) == "--agent_steps") agent_time_steps = std::atoi(argv[++i]);
    else if (std::string(argv[i]) == "--episodes") num_episodes = std::atoi(argv[++i]);
    else if (std::string(argv[i]) == "--lambda") lambda = std::atof(argv[++i]);
    else if (std::string(argv[i]) == "--lambda-c") lambda_c = std::atof(argv[++i]);
    else if (std::string(argv[i]) == "--alpha-v") alpha_v = std::atof(argv[++i]);
    else if (std::string(argv[i]) == "--alpha-u") alpha_u = std::atof(argv[++i]);
    else if (std::string(argv[i]) == "--alpha-vv") alpha_vv = std::atof(argv[++i]);
    else if (std::string(argv[i]) == "--alpha-uu") alpha_uu = std::atof(argv[++i]);
    else if (std::string(argv[i]) == "--alpha-r") alpha_r = std::atof(argv[++i]);
    else if (std::string(argv[i]) == "--epsilon") epsilon = std::atof(argv[++i]);
    else if (std::string(argv[i]) == "--gamma") gamma = std::atof(argv[++i]);
    else if (std::string(argv[i]) == "--ucb-factor") ucb_factor = std::atof(argv[++i]);
    else if (std::string(argv[i]) == "--weighted-dist-choice") weighted_dist_choice = true;
    else if (std::string(argv[i]) == "--scaled-alpha-u") alpha_u = scaled_alpha (lambda, std::atof(argv[++i]));
    else if (std::string(argv[i]) == "--scaled-alpha-v") alpha_v = scaled_alpha (lambda, std::atof(argv[++i]));
    else if (std::string(argv[i]) == "--initial-value") initial_value = std::atof(argv[++i]);
    else if (std::string(argv[i]) == "--print-timesteps") print_timesteps = true;
    else if (std::string(argv[i]) == "--no-print-rewards") print_rewards = false;
    else if (std::string(argv[i]) == "--no-print-episodes") print_episodes = false;
    else if (std::string(argv[i]) == "--print-dir-ratios") print_direction_ratios = true;
    else if (std::string(argv[i]) == "--trunc-normal") trunc_normal = true;
    else if (std::string(argv[i]) == "--no-trunc-normal") trunc_normal = false;
    else if (std::string(argv[i]) == "--tile-weight-exponent") tile_weight_exponent = std::atof(argv[++i]);
    else if (std::string(argv[i]) == "--continuing") continuing = true;
    // else if (std::string(argv[i]) == "--visualize") visualize = true;
    // else if (std::string(argv[i]) == "--visualize-from") visualize_from = std::atoi(argv[++i]);
    // else if (std::string(argv[i]) == "--visualize-every") visualize_every = std::atoi(argv[++i]);
    else if (std::string(argv[i]) == "--subspaces") {
      subspaces.clear();
      std::istringstream arg(argv[++i]);
      int dim;
      char sep;
      while (arg >> dim) {
        subspaces.push_back(dim);
        if (arg >> sep && sep != ',') {
          fprintf(stderr, "Invalid subspaces argument\n");
          return 1;
        }
      }
    }
    else {
      std::fprintf (stderr, "Unknown parameter: %s\n",argv[i]);
      std::exit(1);
    }
  }

  std::mt19937 agent_rng(agent_seed);
  std::mt19937 init_rng(init_seed);
  //if (!visualize) std::fprintf(stdout, "# agent-seed = %u\n# init-seed = %u\n", agent_seed, init_seed);

//   framework f(cart_pole_simulator(),
// #ifdef USE_MULTI_DIST
//               multi_dist_agent(lambda, alpha_v, alpha_u, initial_value, num_features,
//                                  tile_weight_exponent, trunc_normal, subspaces, agent_seed),
// #elif defined(USE_TEST_DIST)
//               beta_agent(lambda, alpha_v, alpha_u, initial_value, num_features,
//                                  tile_weight_exponent, trunc_normal, subspaces),
// #else
//               gauss_agent(lambda, alpha_v, alpha_u, initial_value, num_features,
//                                  tile_weight_exponent, trunc_normal, subspaces),
// #endif
//               dt,
//               agent_time_steps);
  // trunc_normal_distribution dist = trunc_normal_distribution(0, 0.5, -1, 1);
  // for (int i = 0; i < 100; i++) {
  //   std::cout << dist(agent_rng) << std::endl;
  // }
  // exit(0);
  switch(agentType) {
    case GAUSS: {
      framework<gauss_agent> f(cart_pole_simulator(init_rng),
              gauss_agent(lambda, alpha_v, alpha_u, alpha_r, initial_value, num_features,
                                   tile_weight_exponent, trunc_normal, subspaces, continuing),
              dt, max_time,
              agent_time_steps);
      double total_return = 0;
      double total_timesteps = 0;
      if (continuing) {
        std::fprintf(stdout, "%i\n", f.run_episodes(init_rng, agent_rng, print_episodes));
      } else {
        for (int i = 0; i < num_episodes; i++) {

        // if (visualize && i >= visualize_from) {
        //   f.set_visualiser ((i - visualize_from) % visualize_every == 0 ? stdout : nullptr);
        // }

        f.run_episode(init_rng, agent_rng, i==-1);

        // if (!visualize) std::fprintf(stdout, "%g %g\n", f.get_return(), f.get_time_elapsed());
        if (print_episodes) {
          if (print_rewards) {
            std::fprintf(stdout, "%g ", f.get_return());
          }
          if (print_timesteps) {
            std::fprintf(stdout, "%g ", f.get_time_elapsed());
          }
          if (print_direction_ratios) {
            std::fprintf(stdout, "%g %d %d ", f.get_direction_ratio(), f.get_forward_count(), f.get_backward_count());
          }
          std::fprintf(stdout, "\b\n");
        } else {
          total_return += f.get_return();
          total_timesteps += f.get_time_elapsed();
        }
        // std::printf("%g\n", f.time_elapsed);
        }
        if (!print_episodes) {
          std::fprintf(stdout, "%g\n", total_timesteps);
        }
      }
      break;
    }
    case BETA: {
      framework<beta_agent> f(cart_pole_simulator(init_rng),
              beta_agent(lambda, alpha_v, alpha_u, initial_value, num_features,
                                 tile_weight_exponent, trunc_normal, subspaces),
              dt, max_time,
              agent_time_steps);
      double total_return = 0;
      double total_timesteps = 0;
      for (int i = 0; i < num_episodes; i++) {

        // if (visualize && i >= visualize_from) {
        //   f.set_visualiser ((i - visualize_from) % visualize_every == 0 ? stdout : nullptr);
        // }

        f.run_episode(init_rng, agent_rng, i==99);

        // if (!visualize) std::fprintf(stdout, "%g %g\n", f.get_return(), f.get_time_elapsed());
        if (print_episodes) {
          if (print_rewards) {
            std::fprintf(stdout, "%g ", f.get_return());
          }
          if (print_timesteps) {
            std::fprintf(stdout, "%g ", f.get_time_elapsed());
          }
          if (print_direction_ratios) {
            std::fprintf(stdout, "%g %d %d ", f.get_direction_ratio(), f.get_forward_count(), f.get_backward_count());
          }
          std::fprintf(stdout, "\b\n");
        } else {
          total_return += f.get_return();
          total_timesteps += f.get_time_elapsed();
        }
        // std::printf("%g\n", f.time_elapsed);
      }
      if (!print_episodes) {
        std::fprintf(stdout, "%g\n", total_timesteps);
      }
      break;
    }
    case MULTIBETA: {
      framework<multi_beta_agent> f(cart_pole_simulator(init_rng),
              multi_beta_agent(lambda, alpha_v, alpha_u, epsilon, initial_value, num_features,
                                 tile_weight_exponent, trunc_normal, subspaces),
              dt, max_time,
              agent_time_steps);
      double total_return = 0;
      double total_timesteps = 0;
      for (int i = 0; i < num_episodes; i++) {

        // if (visualize && i >= visualize_from) {
        //   f.set_visualiser ((i - visualize_from) % visualize_every == 0 ? stdout : nullptr);
        // }

        f.run_episode(init_rng, agent_rng, i==-1);

        // if (!visualize) std::fprintf(stdout, "%g %g\n", f.get_return(), f.get_time_elapsed());
        if (print_episodes) {
          if (print_rewards) {
            std::fprintf(stdout, "%g ", f.get_return());
          }
          if (print_timesteps) {
            std::fprintf(stdout, "%g ", f.get_time_elapsed());
          }
          if (print_direction_ratios) {
            std::fprintf(stdout, "%g %d %d ", f.get_direction_ratio(), f.get_forward_count(), f.get_backward_count());
          }
          std::fprintf(stdout, "\b\n");
        } else {
          total_return += f.get_return();
          total_timesteps += f.get_time_elapsed();
        }
        // std::printf("%g\n", f.time_elapsed);
      }
      if (!print_episodes) {
        std::fprintf(stdout, "%g\n", total_timesteps);
      }
      break;
    }
    case MULTIGAUSS: {
      framework<multi_gauss_agent> f(cart_pole_simulator(init_rng),
              multi_gauss_agent(lambda, alpha_v, alpha_u, epsilon, initial_value, num_features,
                                 tile_weight_exponent, trunc_normal, subspaces, weighted_dist_choice, ucb_factor),
              dt, max_time,
              agent_time_steps);
      double total_return = 0;
      double total_timesteps = 0;
      if (continuing) {
        std::fprintf(stdout, "%i\n", f.run_episodes(init_rng, agent_rng, print_episodes));
      } else {
        for (int i = 0; i < num_episodes; i++) {

          // if (visualize && i >= visualize_from) {
          //   f.set_visualiser ((i - visualize_from) % visualize_every == 0 ? stdout : nullptr);
          // }

          f.run_episode(init_rng, agent_rng, i==-1);

          // if (!visualize) std::fprintf(stdout, "%g %g\n", f.get_return(), f.get_time_elapsed());
          if (print_episodes) {
            if (print_rewards) {
              std::fprintf(stdout, "%g ", f.get_return());
            }
            if (print_timesteps) {
              std::fprintf(stdout, "%g ", f.get_time_elapsed());
            }
            if (print_direction_ratios) {
              std::fprintf(stdout, "%g %d %d ",f.get_direction_ratio(),f.get_forward_count(),f.get_backward_count());
            }
            std::fprintf(stdout, "\b\n");
          } else {
            total_return += f.get_return();
            total_timesteps += f.get_time_elapsed();
          }
          // std::printf("%g\n", f.time_elapsed);
        }
        if (!print_episodes) {
          std::fprintf(stdout, "%g\n", total_timesteps);
        }
      }
      break;
    }
    case MULTIGAUSS2: {
      framework<multi_gauss_2_agent> f(cart_pole_simulator(init_rng),
              multi_gauss_2_agent(lambda, alpha_v, alpha_u, epsilon, initial_value, num_features,
                                 tile_weight_exponent, trunc_normal, subspaces, weighted_dist_choice, ucb_factor),
              dt, max_time,
              agent_time_steps);
      double total_return = 0;
      double total_timesteps = 0;
      for (int i = 0; i < num_episodes; i++) {

        // if (visualize && i >= visualize_from) {
        //   f.set_visualiser ((i - visualize_from) % visualize_every == 0 ? stdout : nullptr);
        // }

        f.run_episode(init_rng, agent_rng, i==-1/*i==14||i==15||i==16*/);

        // if (!visualize) std::fprintf(stdout, "%g %g\n", f.get_return(), f.get_time_elapsed());
        if (print_episodes) {
          if (print_rewards) {
            std::fprintf(stdout, "%g ", f.get_return());
          }
          if (print_timesteps) {
            std::fprintf(stdout, "%g ", f.get_time_elapsed());
          }
          if (print_direction_ratios) {
            std::fprintf(stdout, "%g %d %d ", f.get_direction_ratio(), f.get_forward_count(), f.get_backward_count());
          }
          std::fprintf(stdout, "\b\n");
        } else {
          total_return += f.get_return();
          total_timesteps += f.get_time_elapsed();
        }
        // std::printf("%g\n", f.time_elapsed);
      }
      if (!print_episodes) {
        std::fprintf(stdout, "%g\n", total_timesteps);
      }
      break;
    }
    case MULTIGAUSS3: {
      framework<multi_gauss_3_agent> f(cart_pole_simulator(init_rng),
              multi_gauss_3_agent(lambda, alpha_v, alpha_u, epsilon, initial_value, num_features,
                                 tile_weight_exponent, trunc_normal, subspaces, weighted_dist_choice, ucb_factor),
              dt, max_time,
              agent_time_steps);
      double total_return = 0;
      double total_timesteps = 0;
      for (int i = 0; i < num_episodes; i++) {

        // if (visualize && i >= visualize_from) {
        //   f.set_visualiser ((i - visualize_from) % visualize_every == 0 ? stdout : nullptr);
        // }

        f.run_episode(init_rng, agent_rng, i==-1/*i==14||i==15||i==16*/);

        // if (!visualize) std::fprintf(stdout, "%g %g\n", f.get_return(), f.get_time_elapsed());
        if (print_episodes) {
          if (print_rewards) {
            std::fprintf(stdout, "%g ", f.get_return());
          }
          if (print_timesteps) {
            std::fprintf(stdout, "%g ", f.get_time_elapsed());
          }
          if (print_direction_ratios) {
            std::fprintf(stdout, "%g %d %d ", f.get_direction_ratio(), f.get_forward_count(), f.get_backward_count());
          }
          std::fprintf(stdout, "\b\n");
        } else {
          total_return += f.get_return();
          total_timesteps += f.get_time_elapsed();
        }
        // std::printf("%g\n", f.time_elapsed);
      }
      if (!print_episodes) {
        std::fprintf(stdout, "%g\n", total_timesteps);
      }
      break;
    }
    case SARSA: {
      framework<sarsa_agent> f(cart_pole_simulator(init_rng),
              sarsa_agent(lambda, alpha_v, alpha_u, alpha_r, epsilon, gamma, initial_value, num_features,
                        tile_weight_exponent, trunc_normal, subspaces, weighted_dist_choice, ucb_factor, continuing),
              dt, max_time,
              agent_time_steps);
      double total_return = 0;
      double total_timesteps = 0;
      if (continuing) {
        std::fprintf(stdout, "%i\n", f.run_episodes(init_rng, agent_rng, print_episodes));
      } else {
        for (int i = 0; i < num_episodes; i++) {

          // if (visualize && i >= visualize_from) {
          //   f.set_visualiser ((i - visualize_from) % visualize_every == 0 ? stdout : nullptr);
          // }

          f.run_episode(init_rng, agent_rng, i==-1);

          // if (!visualize) std::fprintf(stdout, "%g %g\n", f.get_return(), f.get_time_elapsed());
          if (print_episodes) {
            if (print_rewards) {
              std::fprintf(stdout, "%g ", f.get_return());
            }
            if (print_timesteps) {
              std::fprintf(stdout, "%g ", f.get_time_elapsed());
            }
            if (print_direction_ratios) {
              std::fprintf(stdout, "%g %d %d ", f.get_direction_ratio(), f.get_forward_count(), f.get_backward_count());
            }
            std::fprintf(stdout, "\b\n");
          } else {
            total_return += f.get_return();
            total_timesteps += f.get_time_elapsed();
          }
          // std::printf("%g\n", f.time_elapsed);
        }
        if (!print_episodes) {
          std::fprintf(stdout, "%g\n", total_timesteps);
        }
      }
      break;
    }
    case MULTIGAUSS4: {
      framework<multi_gauss_4_agent> f(cart_pole_simulator(init_rng),
            multi_gauss_4_agent(lambda, lambda_c, alpha_v, alpha_u, alpha_vv, alpha_uu, alpha_r, epsilon, gamma,
            initial_value, num_features, tile_weight_exponent, trunc_normal, subspaces, weighted_dist_choice,
            ucb_factor, continuing),
            dt, max_time,
            agent_time_steps);
      double total_return = 0;
      double total_timesteps = 0;
      if (continuing) {
        std::fprintf(stdout, "%i\n", f.run_episodes(init_rng, agent_rng, print_episodes));
      } else {
        for (int i = 0; i < num_episodes; i++) {

          // if (visualize && i >= visualize_from) {
          //   f.set_visualiser ((i - visualize_from) % visualize_every == 0 ? stdout : nullptr);
          // }

          f.run_episode(init_rng, agent_rng, i==-1);

          // if (!visualize) std::fprintf(stdout, "%g %g\n", f.get_return(), f.get_time_elapsed());
          if (print_episodes) {
            if (print_rewards) {
              std::fprintf(stdout, "%g ", f.get_return());
            }
            if (print_timesteps) {
              std::fprintf(stdout, "%g ", f.get_time_elapsed());
            }
            if (print_direction_ratios) {
              std::fprintf(stdout, "%g %d %d ", f.get_direction_ratio(), f.get_forward_count(),\
                  f.get_backward_count());
            }
            std::fprintf(stdout, "\b\n");
          } else {
            total_return += f.get_return();
            total_timesteps += f.get_time_elapsed();
          }
          // std::printf("%g\n", f.time_elapsed);
        }
        if (!print_episodes) {
          std::fprintf(stdout, "%g\n", total_timesteps);
        }
      }
      break;
    }
  }

  // for (int i = 0; i < num_episodes; i++) {
  //
  //   if (visualize && i >= visualize_from) {
  //     f.set_visualiser ((i - visualize_from) % visualize_every == 0 ? stdout : nullptr);
  //   }
  //
  //   f.run_episode(init_rng, agent_rng);
  //
  //   if (!visualize) std::fprintf(stdout, "%g\n", f.get_return());
  //   // std::printf("%g\n", f.time_elapsed);
  // }

  return 0;
}
