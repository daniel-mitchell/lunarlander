#ifndef _UTILITY_HPP
#define _UTILITY_HPP

#include <cmath>
#include <random>

#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/complement.hpp>

inline double norm_pdf(double x) {
  using namespace std;
  return exp(-x*x/2.0) / sqrt(2*boost::math::constants::pi<double>());
}

inline double norm_cdf(double x) {
  using namespace std;
  return 0.5 * boost::math::erfc(-x/(sqrt(2)));
}

inline double norm_cdfc(double x) {
  using namespace std;
  return 0.5 * boost::math::erfc(x/(sqrt(2)));
}

inline double norm_quantile(double p) {
  return -sqrt(2) * boost::math::erfc_inv(2*p);
}

class normal_distribution {

  std::normal_distribution<double> dist;

public:

  normal_distribution(double mu, double sigma) : dist(mu, sigma) {}

  double mean() const { return dist.mean(); }

  double sigma() const { return dist.stddev(); }

  double pdf(double x) const {
    return norm_pdf((x-mean())/sigma());
  }

  double cdf(double x) const {
    return norm_cdf((x-mean())/sigma());
  }

  double cdfc(double x) const {
    return norm_cdfc((x-mean())/sigma());
  }

  template <class Engine> double operator()(Engine& rng) { return dist(rng); }

};


class trunc_normal_distribution {

  double _mu, _sigma;
  double _alpha, _beta;

  double cdf_alpha, cdf_beta;
	double delta;

public:

  trunc_normal_distribution(double mu, double sigma, double a, double b)
    : _mu(mu), _sigma(sigma), _alpha((a-mu)/sigma), _beta((b-mu)/sigma),
      cdf_alpha(norm_cdf(_alpha)), cdf_beta(norm_cdf(_beta)),
      delta(cdf_beta - cdf_alpha)
  {}

  double mu() const { return _mu; }
  double sigma() const { return _sigma; }
  double alpha() const { return _alpha; }
  double beta() const { return _beta; }
  double norm_constant() const { return delta; }

  double pdf(double x) const {
    const double std_x = (x - mu()) / sigma();
    if (std_x < _alpha || std_x > _beta) return 0;
    return norm_pdf(std_x) / delta;
  }

  double cdf(double x) const {
    const double std_x = (x - mu()) / sigma();
    if (std_x < _alpha) return 0;
    if (std_x > _beta) return 1;
    return (norm_cdf(std_x) - cdf_alpha) / delta;
  }

  template <class Engine> double operator()(Engine& rng) const {
    double cdf_x = std::uniform_real_distribution<double>(cdf_alpha, cdf_beta)(rng);
    return norm_quantile(cdf_x) * sigma() + mu();
  }

};

class beta_distribution {

  double _alpha, _beta;
  boost::math::beta_distribution<> dist;

public:

  beta_distribution(double alpha, double beta) : _alpha(alpha), _beta(beta), dist(alpha, beta) {}

  double mean() const { return boost::math::mean(dist); }
  // double mean() const { return _alpha/(_alpha+_beta); }

  double variance() const { return boost::math::variance(dist); }
  // double sigma() const { return sqrt((_alpha*_beta)/((_alpha+_beta)*(_alpha+_beta)*(_alpha+_beta+1))); }

  double alpha() const { return _alpha; }

  double beta() const { return _beta; }

  double pdf(double x) const {
    return boost::math::pdf(dist, x);
  }

  double cdf(double x) const {
    return boost::math::cdf(dist, x);
  }

  double cdfc(double x) const {
    return boost::math::cdf(boost::math::complemented2_type<boost::math::beta_distribution<>, double>(dist, x));
  }

  template <class Engine> double operator()(Engine& rng) {
    std::gamma_distribution<double> gamma1(_alpha, 1);
    std::gamma_distribution<double> gamma2(_beta, 1);
    double sample1 = gamma1(rng);
    double sample2 = gamma2(rng);
    //Sampling z from beta(a, b) is z = (x/(x+y)) where x is sampled from gamma(a,1) and y from gamma(b,1)
    return sample1/(sample1+sample2);
  }

};

#endif
