#include <boost/math/special_functions/fpclassify.hpp>
#include <limits>
#include <algorithm>

#include "beta_pg_agent.hpp"

void beta_linear_function_approx::add_features(const VectorXi& features, double scaling) {
  if (trace.size() >= trace_len) trace.pop_back();
  trace.push_front(std::make_pair(features, scaling));
}


double beta_linear_function_approx::value(const VectorXi& features) const {
  double value = 0;
  for (int i = 0; i < features.size(); ++i) {
    value += weights(features(i)) * feature_weights(i);
  }
  return value;
}

//Capping
void beta_linear_function_approx::update(double delta) {
  for (unsigned int i = 0; i < trace.size(); ++i) {
    const VectorXi& features = trace[i].first;
    const double amount = delta * trace[i].second;
    for (int j = 0; j < features.size(); ++j) {
      // weights(features(j)) += amount * feature_weights(j);
      weights(features(j)) = std::min(weights(features(j)) + (amount * feature_weights(j)), maxParam);
    }
    delta *= falloff;
  }
  //TODO: This is slow
  // weights = weights.cwiseMin(maxParam);
}

// Scaling
// void beta_linear_function_approx::update(double delta) {
//   bool scale = false;
//   double scaleFactor = 0;
//   for (unsigned int i = 0; i < trace.size(); ++i) {
//     const VectorXi& features = trace[i].first;
//     const double amount = delta * trace[i].second;
//     for (int j = 0; j < features.size(); ++j) {
//       weights(features(j)) += amount * feature_weights(j);
//       if (weights(features(j)) > maxParam) {
//         scale = true;
//         scaleFactor = std::max(scaleFactor, weights(features(j)));
//       }
//     }
//     delta *= falloff;
//   }
//   if (scale) {
//     weights *= (maxParam/scaleFactor);
//   //weights *= (1/scaleFactor);
//   }
// }

double beta_td_critic::get_value() {
  return value.value(features);
}

double beta_td_critic::evaluate(const VectorXi& new_features, double reward, bool terminal) {
  double old_value = value.value (features);
  double new_value = terminal ? 0.0 : value.value (new_features);
  double td_error = reward + gamma*new_value - old_value;

  value.add_features(features);
  value.update(alpha * td_error);

  features = new_features;

  return td_error;
}

double beta_td_critic::evaluate_multi(beta_td_critic& critic_1, beta_td_critic& critic_2, bool direction, \
                                    const VectorXi& new_features, double reward, bool terminal) {
  double old_value = direction ? critic_1.get_value() : critic_2.get_value();
  double new_value = terminal ? 0.0 :\
                      std::max(critic_1.value.value(new_features), critic_2.value.value(new_features));
  double td_error = reward + critic_1.gamma*new_value - old_value; //Both critics should have the same gamma
  
  if (direction) {
    critic_1.value.add_features(critic_1.features);
    critic_2.value.add_features(VectorXi::Zero(critic_2.features.size()));
  } else {
    critic_1.value.add_features(VectorXi::Zero(critic_1.features.size()));
    critic_2.value.add_features(critic_2.features);
  }
  critic_1.value.update(critic_1.alpha * td_error);
  critic_2.value.update(critic_2.alpha * td_error);

  critic_1.features = new_features;
  critic_2.features = new_features;

  return td_error;
}

beta_distribution beta_pg_actor::action_dist() const {

  double alpha_value = exp(alpha.value(features)) + 1;
  double beta_value = exp(beta.value(features)) + 1;
  //Alternate to clipping or scaling on beta_linear_function_approx::update
  // if (!boost::math::isfinite(alpha_value)) {
  //   alpha_value = 1;
  //   beta_value = std::numeric_limits<double>::min();
  // }
  // if (!boost::math::isfinite(beta_value)) {
  //   alpha_value = std::numeric_limits<double>::min();
  //   beta_value = 1;
  // }
  return beta_distribution(alpha_value, beta_value);
}


void beta_pg_actor::learn(const double td_error) {

  const beta_distribution dist = action_dist();

  double alphaGradLog = log(action) + boost::math::digamma(dist.alpha() + dist.beta())\
                            - boost::math::digamma(dist.alpha())*(dist.alpha() - 1);

  double betaGradLog = log(1 - action) + boost::math::digamma(dist.alpha() + dist.beta())\
                            - boost::math::digamma(dist.beta())*(dist.beta() - 1);

  alpha.add_features(features, alphaGradLog);
  if (s) {
    alpha.update(alphaStep * td_error * dist.variance());
  }
  alpha.update(alphaStep * td_error);

  beta.add_features(features, betaGradLog);
  if (s) {
    beta.update(alphaStep * td_error * dist.variance());
  }
  beta.update(alphaStep * td_error);
}
