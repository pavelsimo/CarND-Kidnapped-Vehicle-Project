/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    /// set the number of particles
    num_particles = 100;

    /// initialize all particles to first position
    std::random_device rd;
    std::mt19937 gen(rd());

    double std_x = std[0];
    normal_distribution<double> dist_x(x, std_x);

    double std_y = std[1];
    normal_distribution<double> dist_y(y, std_y);

    double std_theta = std[2];
    normal_distribution<double> dist_theta(theta, std_theta);

    for (int i = 0; i < num_particles; ++i) {
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1;
        particles.push_back(p);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    std::random_device rd;
    std::mt19937 gen(rd());

    for (auto &p: particles) {
        double std_x = std_pos[0];
        normal_distribution<double> dist_x(p.x, std_x);

        double std_y = std_pos[1];
        normal_distribution<double> dist_y(p.y, std_y);

        double std_theta = std_pos[2];
        normal_distribution<double> dist_theta(p.theta, std_theta);

        double x_0 = dist_x(gen);
        double y_0 = dist_y(gen);
        double theta_0 = dist_theta(gen);

        double c1 = velocity / (yaw_rate + 0.00000001);
        double theta_f = theta_0 + yaw_rate * delta_t;
        double x_f = x_0 + c1 * (std::sin(theta_f) - std::sin(theta_0));
        double y_f = y_0 + c1 * (std::cos(theta_0) - std::cos(theta_f));
        p.x = x_f;
        p.y = y_f;
        p.theta = theta_f;
        assert(!std::isnan(p.x));
        assert(!std::isnan(p.y));
        assert(!std::isnan(p.theta));
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.
    for (int i = 0; i < observations.size(); ++i) {
        double best = std::numeric_limits<double>::max();
        int k = 0;
        for (int j = 0; j < predicted.size(); ++j) {
            double d = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
            if (d < best) {
                k = j;
                best = d;
            }
        }
        observations[i] = predicted[k];
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    double std_x = std_landmark[0];
    double std_y = std_landmark[1];

    /// convert the map landmarks to observations
    for (auto &p: particles) {
        /// transform from vehicle coordinates to map coordinates
        std::vector<LandmarkObs> observations_m;
        std::vector<LandmarkObs> predicted_landmarks;
        for (const auto &obs: observations) {
            LandmarkObs obs_m;
            obs_m.id = obs.id;
            obs_m.x = p.x + obs.x * cos(p.theta) - obs.y * sin(p.theta);
            obs_m.y = p.y + obs.x * sin(p.theta) + obs.y * cos(p.theta);
            observations_m.push_back(obs_m);
            predicted_landmarks.push_back(obs_m);
        }

        /// extracting landmarks in range...
        std::vector<LandmarkObs> landmarks_in_range;
        for (auto &l: map_landmarks.landmark_list) {
            LandmarkObs obs;
            obs.id = l.id_i;
            obs.x = l.x_f;
            obs.y = l.y_f;
            double d = dist(obs.x, obs.y, p.x, p.y);
            if (d <= sensor_range) {
                landmarks_in_range.push_back(obs);
            }
        }

        /// landmarks not in range
        if (landmarks_in_range.empty()) {
            std::cout << "No Landmarks in Range [" << p.x << "," << p.y << "," << p.theta << "]" << '\n';
            continue;
        }

        /// data association between observations and map landmarks
        dataAssociation(landmarks_in_range, predicted_landmarks);

        /// multivariable-gaussian probability density to update the particle weights
        for (int i = 0; i < observations_m.size(); ++i) {
            double x = observations_m[i].x;
            double y = observations_m[i].y;
            double mean_x = predicted_landmarks[i].x;
            double mean_y = predicted_landmarks[i].y;
            double c1 = 1.0 / (2 * M_PI * std_x * std_y);
            double c2 = ((x - mean_x) * (x - mean_x) / (2 * std_x * std_x))
                        + ((y - mean_y) * (y - mean_y) / (2 * std_y * std_y));
            double prob = c1 * std::exp(-c2);
            p.weight *= prob;
        }
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::random_device rd;
    std::mt19937 gen(rd());

    weights.clear();
    double sum_w = 0;
    for (auto &p: particles) {
        sum_w += p.weight;
    }

    if (sum_w == 0) {
        weights = vector<double>(particles.size(), 1);
    } else {
        for (auto &p: particles) {
            p.weight /= sum_w;
            weights.push_back(p.weight);
        }
    }

    std::discrete_distribution<> d(weights.begin(), weights.end());
    vector<Particle> new_particles;
    new_particles.reserve(num_particles);
    for (int i = 0; i < num_particles; ++i) {
        new_particles.push_back(particles[d(gen)]);
    }
    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations,
                                         const std::vector<double> &sense_x, const std::vector<double> &sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
