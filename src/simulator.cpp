#include "simulator.hpp"

#include <utility>
#include <cstdlib>
#include <cmath>

#include <Eigen/Geometry>


mountain_car_simulator::mountain_car_simulator (std::mt19937 &rng) {
  initialize(rng);
}


void mountain_car_simulator::initialize(std::mt19937 &rng) {
  position = (rng()/rng.max())*(MAX_START_POSITION - MIN_START_POSITION) + MIN_START_POSITION;
  velocity = 0;
  done = false;

  //update(0.0);
}


void mountain_car_simulator::update(double dt) {
  velocity += dt*(current_action.thrust * POWER - 0.0025*cos(3*(position+POSITION_OFFSET))); //Taken from OpenAI Gym
  velocity = std::min(MAX_SPEED, std::max(velocity, MIN_SPEED));
  position += dt*velocity;
  position = std::min(MAX_POSITION, std::max(position, MIN_POSITION));

  if (position == MIN_POSITION && velocity < 0) {
    velocity = 0;
  }

  if (position >= GOAL_POSITION) {
    done = true;
  }
}

void mountain_car_simulator::set_action(const action& new_action) {
  current_action.thrust = std::max (-MAX_THRUST(), std::min(MAX_THRUST(), new_action.thrust));
}

VectorXd mountain_car_simulator::get_state() const {
  VectorXd state(2);
  state(0) = position;
  state(1) = velocity;
  return state;
}