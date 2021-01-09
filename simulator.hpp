#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <array>
#include <random>
#include "./toml11/toml.hpp"
#include "./particle.hpp"
#include "./new_particle.hpp"

namespace generalized_langevin {
    class Simulator {
        public:
            Simulator(const std::string& input_setup_file_path);
            void run() noexcept;
        private:
            //係数と物理定数
            double friction_coefficient;
            double memory_coefficient;//摩擦核の係数
            double K_b;
            double omega;//調和ポテンシャルの核振動数

            Particle particle1;
            Particle particle2;
            std::ofstream out;
            std::mt19937 random_engine;
            std::size_t step_num;
            std::size_t save_step_num;
            double delta_t;
            double temperature;
            std::array<double,3> xi_t, xi_tph;
            std::normal_distribution<> xi_engine;

            void step() noexcept;
            std::array<double,3> calculate_coordinate(Particle target_particle, Particle another_particle, std::size_t step_index) noexcept;
            std::array<double,3> calculate_velocity(Particle target_particle, Particle another_particle, NewParticle new_particle, std::size_t step_index) noexcept;
            //座標と速度の計算に使う関数
            std::array<double,3> grad_harmonic_potential(Particle p1, Particle p2);
            std::array<double,3> grad_to_force(std::array<double,3> grad);
            //摩擦項を求める関数
            std::array<double,3> calculate_friction(Particle p, std::size_t step_index);
            double memory_func(double coefficient, double time);
            double calculate_Iprime(Particle p, std::size_t step_index);
            void write_coordinate() noexcept;
    };//Simulator

    Simulator::Simulator(const std::string& input_setup_file_path) {
        const auto input_setup_file = toml::parse(input_setup_file_path);

        //定数の読み込み
        friction_coefficient = toml::find<double>(input_setup_file, "constants", "friction_coefficient");
        memory_coefficient = toml::find<double>(input_setup_file, "constants", "memory_coefficient");
        K_b = toml::find<double>(input_setup_file, "constants", "K_b");
        omega = toml::find<double>(input_setup_file, "constants", "omega");

        //粒子の初期化
        particle1.x = toml::find<double>(input_setup_file, "particle1", "x");
        particle1.y = toml::find<double>(input_setup_file, "particle1", "y");
        particle1.z = toml::find<double>(input_setup_file, "particle1", "z");
        particle1.vx[0] = 0.0;
        particle1.vy[0] = 0.0;
        particle1.vz[0] = 0.0;
        particle1.mass = toml::find<double>(input_setup_file, "particle1", "mass");
        particle2.x = toml::find<double>(input_setup_file, "particle2", "x");
        particle2.y = toml::find<double>(input_setup_file, "particle2", "y");
        particle2.z = toml::find<double>(input_setup_file, "particle2", "z");
        particle2.vx[0] = 0.0;
        particle2.vy[0] = 0.0;
        particle2.vz[0] = 0.0;
        particle2.mass = toml::find<double>(input_setup_file, "particle2", "mass");

        const auto project_name = toml::find<std::string>(input_setup_file, "meta_data", "project_name");
        const auto working_path = toml::find<std::string>(input_setup_file, "meta_data", "working_path");
        out.open(working_path + "/" + project_name + ".xyz");
        if(!out) {
            std::cerr << "cannot open:" << working_path + "/" + project_name + ".xyz" << std::endl;
            std::exit(1);
        }

        const auto random_seed = toml::find<std::size_t>(input_setup_file, "meta_data", "random_seed");
        random_engine.seed(random_seed);

        step_num = toml::find<std::size_t>(input_setup_file, "meta_data", "step_num");
        save_step_num = toml::find<std::size_t>(input_setup_file, "meta_data", "save_step_num");
        temperature = toml::find<double>(input_setup_file, "meta_data", "temperature");
        delta_t = toml::find<double>(input_setup_file, "meta_data", "delta_t");

        std::normal_distribution<> init_xi_engine(0.0, std::sqrt((2.0*friction_coefficient*K_b*temperature*delta_t)/particle1.mass));
        xi_engine = init_xi_engine;

        xi_t = {
            xi_engine(random_engine),
            xi_engine(random_engine),
            xi_engine(random_engine)
        };
        xi_tph = {
            xi_engine(random_engine),
            xi_engine(random_engine),
            xi_engine(random_engine)
        };    
    }//constructor

    void Simulator::run() noexcept {
        write_coordinate();
        for (std::size_t step_index = 1; step_index <= step_num; ++step_index) {
            step();
            if(step_index%save_step_num == 0) {
                write_coordinate();
            }
        }
    }
    
}//generalized_langevin

#endif