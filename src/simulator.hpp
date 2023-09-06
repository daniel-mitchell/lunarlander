#ifndef _SIMULATOR_HPP
#define _SIMULATOR_HPP

#include <vector>
#include <random>
#include <Eigen/Core>

using Eigen::VectorXd;
using Eigen::Matrix2d;

class cart_pole_simulator {

public:

  struct action {
    double thrust;
    action(double thrust = 0) : thrust(thrust) {}
  };

private:

  static constexpr double GRAVITY = 9.8;
  static constexpr double MASS_CART = 1.0;
  static constexpr double MASS_POLE = 0.1;
  static constexpr double TOTAL_MASS = MASS_CART + MASS_POLE;
  static constexpr double LENGTH = 0.5;
  static constexpr double POLE_MASS_LENGTH = MASS_POLE * LENGTH;
  static constexpr double MIN_ACTION = -1.0;
  static constexpr double MAX_ACTION = 1.0;
  static constexpr double FORCE_MAG = 1.0; // Scales action
  static constexpr double TAU = 0.02; // TODO: Time between state updates, refactor
  static constexpr bool EULER_KINEMATICS_INTEGRATOR = true; // true for euler, false for semi-implicit euler

  static constexpr double MIN_START_POSITION = -0.05;
  static constexpr double MAX_START_POSITION = 0.05;
  static constexpr double MIN_START_VELOCITY = -0.05;
  static constexpr double MAX_START_VELOCITY = 0.05;
  static constexpr double MIN_START_ANGLE = -0.05;
  static constexpr double MAX_START_ANGLE = 0.05;
  static constexpr double MIN_START_ANGULAR_VELOCITY = -0.05;
  static constexpr double MAX_START_ANGULAR_VELOCITY = 0.05;
  static constexpr double THESHOLD_ANGLE = 0.2095;
  static constexpr double MAX_ANGLE = 2*THESHOLD_ANGLE;
  static constexpr double THESHOLD_POSITION = 2.4;
  static constexpr double MAX_POSITION = 2*THESHOLD_POSITION;
  static constexpr double MIN_ANGLE = -MAX_ANGLE;
  static constexpr double MIN_POSITION = -MAX_POSITION;

  double position;
  double velocity;
  double angle;
  double angular_velocity;
  action current_action;
  bool done = false;

public:

  static double MAX_THRUST () { return MAX_ACTION; }

  cart_pole_simulator(std::mt19937 &rng);

  void initialize(std::mt19937 &rng);

  void update([[maybe_unused]] double dt);

  void set_action(const action& new_action);
  const action& get_action () const { return current_action; }

  bool is_done() { return done; }

  VectorXd get_state() const;

};

#endif
