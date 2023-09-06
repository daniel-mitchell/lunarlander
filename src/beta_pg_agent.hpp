#ifndef _BETA_PG_AGENT_HPP
#define _BETA_PG_AGENT_HPP

#include <cmath>
#include <random>
#include <Eigen/Core>
#include <deque>
#include <utility>

#include "tile_coder.hpp"
#include "policy_gradient_agent.hpp"
#include "utility.hpp"

using Eigen::VectorXd;

class beta_linear_function_approx {

  VectorXd feature_weights;
  VectorXd weights;
  double falloff;
  unsigned int trace_len;
  std::deque<std::pair<VectorXi, double> > trace;

  constexpr static double maxAlphaBeta = 1000;
  double maxParam = log(maxAlphaBeta);

public:

  beta_linear_function_approx(const tile_coder_base& tc, double falloff, double initial_value=0, double threshold=0.05)
    : feature_weights(tc.get_feature_weights()),
      weights(VectorXd::Constant(tc.get_num_features(), initial_value/feature_weights.sum())),
      falloff(falloff),
      trace_len(falloff > 0 ? (unsigned int)(std::ceil(std::log(threshold) / std::log(falloff))) : 1)
  {}

  void initialize() { trace.clear(); }

  void add_features(const VectorXi& features, double scaling=1.0);

  double value(const VectorXi& features)  const;

  void update(double delta);
};

class beta_td_critic {

  double alpha;
  double gamma;
  linear_function_approx value;
  VectorXi features;

public:

  beta_td_critic(const tile_coder_base& tc, double lambda, double alpha, double initial_value=0, double gamma=1)
    : alpha(alpha), gamma(gamma), value(tc, gamma*lambda, initial_value) {}

  void initialize(const VectorXi& new_features) {
    features = new_features;
    value.initialize();
  }

  double get_value();

  double evaluate(const VectorXi& new_features, double reward, bool terminal=false);

  static double evaluate_multi(beta_td_critic& critic_1, beta_td_critic& critic_2, bool direction, \
                                    const VectorXi& new_features, double reward, bool terminal);
};

class beta_pg_actor {
  double alphaStep;
  double min_action, max_action;
  // double min_sigma, sigma_range;
  // bool use_trunc_normal;
  beta_linear_function_approx alpha, beta;
  VectorXi features;
  double action;
  double actionThreshold = 0.001;

  bool inac = false;
  bool s = false;

public:

  beta_pg_actor(const tile_coder_base& tc, double lambda, double alphaStep,
                        double min_action, double max_action,
                        double initial_alpha, double initial_beta,
                        double gamma)
    : alphaStep(alphaStep), min_action(min_action), max_action(max_action),
      alpha(tc, gamma*lambda, initial_alpha), beta(tc, gamma*lambda, initial_beta)
  {}

  void initialize() {
    alpha.initialize();
    beta.initialize();
  }

  beta_distribution action_dist() const;

  double act(std::mt19937& rng, const VectorXi& new_features) {
    features = new_features;
    action = action_dist()(rng);
    action = std::min(std::max(action, actionThreshold), 1 - actionThreshold);
    return action;
  }

  void learn(double td_error);

};

#endif
