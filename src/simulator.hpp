#ifndef _SIMULATOR_HPP
#define _SIMULATOR_HPP

#include <vector>
#include <random>
#include <Eigen/Core>

using Eigen::VectorXd;
using Eigen::Matrix2d;

class mountain_car_simulator {

public:

  struct action {
    double thrust;
    action(double thrust = 0) : thrust(thrust) {}
  };

private:

  static constexpr double MAX_ACTION = 1.0;
  static constexpr double MIN_ACTION = -MAX_ACTION;
  static constexpr double MIN_POSITION = 0;//-1.2;
  static constexpr double MAX_POSITION = 1.8;//0.6;
  static constexpr double POSITION_OFFSET = -1.2; //Used for calculation of sinusoidal curve
  static constexpr double MIN_START_POSITION = 0.6;//-0.6;
  static constexpr double MAX_START_POSITION = 0.8;//-0.4;
  static constexpr double MAX_SPEED = 0.07;
  static constexpr double MIN_SPEED = -MAX_SPEED;
  static constexpr double GOAL_POSITION = 1.65;//0.45;
  static constexpr double POWER = 0.0015;

  double position;
  double velocity;
  action current_action;
  bool done = false;

public:

  static double MAX_THRUST () { return MAX_ACTION; }

  mountain_car_simulator(std::mt19937 &rng);

  void initialize(std::mt19937 &rng);

  void update([[maybe_unused]] double dt);

  void set_action(const action& new_action);
  const action& get_action () const { return current_action; }

  bool is_done() { return done; }

  VectorXd get_state() const;

};

#endif
