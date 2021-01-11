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
            std::normal_distribution<> xi_engine1;
            std::normal_distribution<> xi_engine2;
            std::array<std::array<double,3>, 2> xi_t;
            std::array<std::array<double,3>, 2> xi_tph;

            void step(std::size_t step_index) noexcept;
            //memo:Particleに参照渡しを指定
            std::array<double,3> calculate_coordinate(Particle& target_particle,std::size_t particle_index, Particle& another_particle, std::size_t step_index) noexcept;
            std::array<double,3> calculate_velocity(Particle& target_particle,std::size_t particle_index, Particle& another_particle, NewParticle& new_particle, std::size_t step_index) noexcept;
            //座標と速度の計算に使う関数
            std::array<double,3> grad_harmonic_potential(Particle& p1, Particle& p2);
            std::array<double,3> grad_harmonic_potential(Particle& p1, NewParticle& p2); //memo: オーバーロード
            std::array<double,3> grad_to_force(std::array<double,3> grad);
            //摩擦項を求める関数
            std::array<double,3> calculate_friction(Particle& p, std::size_t step_index);
            double memory_func(double coefficient, double time);
            std::array<double,3> calculate_Iprime(Particle& p, std::size_t step_index);
            void write_coordinate() noexcept;

            //粒子と更新用のクラスの変換
            NewParticle toNewParticle(Particle& p);
            //更新用クラスから粒子を更新する
            void update_particle(Particle& target, NewParticle new_p, std::size_t step_index);
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

        std::normal_distribution<> init_xi_engine1(0.0, std::sqrt((2.0*friction_coefficient*K_b*temperature*delta_t)/particle1.mass));
        xi_engine1 = init_xi_engine1;
        std::normal_distribution<> init_xi_engine2(0.0, std::sqrt((2.0*friction_coefficient*K_b*temperature*delta_t)/particle2.mass));
        xi_engine2 = init_xi_engine2;

        xi_t[0] = {
            xi_engine1(random_engine),
            xi_engine1(random_engine),
            xi_engine1(random_engine)
        };
        xi_tph[0] = {
            xi_engine1(random_engine),
            xi_engine1(random_engine),
            xi_engine1(random_engine)
        };
        xi_t[1] = {
            xi_engine2(random_engine),
            xi_engine2(random_engine),
            xi_engine2(random_engine)
        };
        xi_tph[1] = {
            xi_engine2(random_engine),
            xi_engine2(random_engine),
            xi_engine2(random_engine)
        };   
    }//constructor

    void Simulator::run() noexcept {
        write_coordinate();
        for (std::size_t step_index = 1; step_index <= step_num; ++step_index) {
            step(step_index);
            if(step_index%save_step_num == 0) {
                write_coordinate();
            }
        }
    }

    NewParticle Simulator::toNewParticle(Particle& p) {
        NewParticle res;
        res.x = p.x;
        res.y = p.y;
        res.z = p.z;
        res.vx = p.vx[0];
        res.vy = p.vy[0];
        res.vz = p.vz[0];
        res.mass = p.mass;

        return res;
    }

    void Simulator::update_particle(Particle& target, NewParticle new_p, std::size_t step_index) {
        target.x = new_p.x;
        target.y = new_p.y;
        target.z = new_p.z;
        target.vx[step_index] = new_p.vx;
        target.vy[step_index] = new_p.vy;
        target.vz[step_index] = new_p.vz;
    }
    
    void Simulator::step(std::size_t step_index) noexcept {
        NewParticle new_particle1 = toNewParticle(particle1);
        NewParticle new_particle2 = toNewParticle(particle2);

        //次の時刻の座標を求める
        const auto [new_x1, new_y1, new_z1] = calculate_coordinate(particle1, 0, particle2, step_index);
        new_particle1.x = new_x1;
        new_particle1.y = new_y1;
        new_particle1.z = new_z1;
        const auto [new_x2, new_y2, new_z2] = calculate_coordinate(particle2, 1, particle1, step_index);
        new_particle2.x = new_x2;
        new_particle2.y = new_y2;
        new_particle2.z = new_z2;

        //次の時刻の速度を求める
        const auto [new_vx1, new_vy1, new_vz1] = calculate_velocity(particle1, 0, particle2, new_particle2, step_index);
        new_particle1.vx = new_vx1;
        new_particle1.vy = new_vy1;
        new_particle1.vz = new_vz1;
        const auto [new_vx2, new_vy2, new_vz2] = calculate_velocity(particle2, 1, particle1, new_particle1, step_index);
        new_particle2.vx = new_vx2;
        new_particle2.vy = new_vy2;
        new_particle2.vz = new_vz2;

        update_particle(particle1, new_particle1, step_index);
        update_particle(particle2, new_particle2, step_index);

        xi_t[0] = xi_tph[0];
        xi_tph[0] = {
            xi_engine1(random_engine),
            xi_engine1(random_engine),
            xi_engine1(random_engine)
        };
        xi_t[1] = xi_tph[1];
        xi_tph[1] = {
            xi_engine2(random_engine),
            xi_engine2(random_engine),
            xi_engine2(random_engine)
        };
    }

    std::array<double,3> Simulator::calculate_coordinate(Particle& target_particle,std::size_t particle_index, Particle& another_particle, std::size_t step_index) noexcept {
        //外力項を求める
        std::array<double,3> grad = grad_harmonic_potential(target_particle, another_particle);
        std::array<double,3> f = grad_to_force(grad);

        //摩擦項を求める
        std::array<double,3> I = calculate_friction(target_particle, step_index);

        //更新後の座標を速度Verlet法で求める
        const double next_x = target_particle.x + target_particle.vx[step_index - 1]*delta_t + ((delta_t*delta_t)/2.0)*(f[0]/target_particle.mass - I[0] + target_particle.mass*xi_t[particle_index][0]);
        const double next_y = target_particle.y + target_particle.vy[step_index - 1]*delta_t + ((delta_t*delta_t)/2.0)*(f[1]/target_particle.mass - I[1] + target_particle.mass*xi_t[particle_index][1]);
        const double next_z = target_particle.z + target_particle.vz[step_index - 1]*delta_t + ((delta_t*delta_t)/2.0)*(f[2]/target_particle.mass - I[2] + target_particle.mass*xi_t[particle_index][2]);

        return {next_x, next_y, next_z};
    }

    std::array<double,3> Simulator::calculate_velocity(Particle& target_particle,std::size_t particle_index, Particle& another_particle, NewParticle& new_particle, std::size_t step_index) noexcept {
        //外力項
        std::array<double,3> grad = grad_harmonic_potential(target_particle, another_particle);
        std::array<double,3> next_grad = grad_harmonic_potential(target_particle, new_particle);
        std::array<double,3> f = grad_to_force(grad);
        std::array<double,3> next_f = grad_to_force(next_grad);

        //摩擦項
        std::array<double,3> I = calculate_friction(target_particle, step_index);
        std::array<double,3> Iprime = calculate_Iprime(target_particle, step_index);

        //更新後の速度を求める
        double term1 = 1 - delta_t/2.0 + (delta_t/2.0)*(delta_t/2.0);
        const double next_vx = term1 * (target_particle.vx[step_index - 1] + (delta_t/2.0)*(((f[0] + next_f[0])/target_particle.mass) - (I[0] + Iprime[0]) + (xi_t[particle_index][0] + xi_tph[particle_index][0])));
        const double next_vy = term1 * (target_particle.vy[step_index - 1] + (delta_t/2.0)*(((f[1] + next_f[1])/target_particle.mass) - (I[1] + Iprime[1]) + (xi_t[particle_index][1] + xi_tph[particle_index][1])));
        const double next_vz = term1 * (target_particle.vz[step_index - 1] + (delta_t/2.0)*(((f[2] + next_f[2])/target_particle.mass) - (I[2] + Iprime[2]) + (xi_t[particle_index][2] + xi_tph[particle_index][2])));

        return {next_vx, next_vy, next_vz};
    }

    std::array<double,3> Simulator::grad_harmonic_potential(Particle& p1, Particle& p2) {
        const double grad_x = p1.mass*omega*omega*(p1.x - p2.x);
        const double grad_y = p1.mass*omega*omega*(p1.y - p2.y);
        const double grad_z = p1.mass*omega*omega*(p1.z - p2.z);

        return {grad_x, grad_y, grad_z};
    }

    std::array<double,3> Simulator::grad_harmonic_potential(Particle& p1, NewParticle& p2) {
        const double grad_x = p1.mass*omega*omega*(p1.x - p2.x);
        const double grad_y = p1.mass*omega*omega*(p1.y - p2.y);
        const double grad_z = p1.mass*omega*omega*(p1.z - p2.z);

        return {grad_x, grad_y, grad_z};
    }

    std::array<double,3> Simulator::grad_to_force(std::array<double,3> grad) {
        std::array<double,3> f;
        for (std::size_t index = 0; index <= 2; ++index) {
            f[index] = grad[index] * -1.0;
        }

        return f;
    }

    double Simulator::memory_func(double coefficient, double time) {
        double res = std::exp(-1.0*coefficient*time);
        return res;
    }

    std::array<double,3> Simulator::calculate_friction(Particle& p, std::size_t step_index) {
        double time = static_cast<double>(step_index)*delta_t;
        
        double friction_x = 0.0;
        for (std::size_t i = 0; i < step_index; ++i) {
            double term1 = memory_func(memory_coefficient, time - static_cast<double>(i)*delta_t)*p.vx[i];
            double term2 = memory_func(memory_coefficient, time - static_cast<double>(i + 1)*delta_t)*p.vx[i + 1];
            friction_x += (term1 + term2)*(delta_t/2.0);
        }

        double friction_y = 0.0;
        for (std::size_t i = 0; i < step_index; ++i) {
            double term1 = memory_func(memory_coefficient, time - static_cast<double>(i)*delta_t)*p.vy[i];
            double term2 = memory_func(memory_coefficient, time - static_cast<double>(i + 1)*delta_t)*p.vy[i + 1];
            friction_y += (term1 + term2)*(delta_t/2.0);
        }

        double friction_z = 0.0;
        for (std::size_t i = 0; i < step_index; ++i) {
            double term1 = memory_func(memory_coefficient, time - static_cast<double>(i)*delta_t)*p.vz[i];
            double term2 = memory_func(memory_coefficient, time - static_cast<double>(i + 1)*delta_t)*p.vz[i + 1];
            friction_z += (term1 + term2)*(delta_t/2.0);
        }

        return {friction_x, friction_y, friction_z};
    }

    std::array<double,3> Simulator::calculate_Iprime(Particle& p, std::size_t step_index) {
        double time = static_cast<double>(step_index + 1)*delta_t;
        
        double iprime_x = 0.0;
        for (std::size_t i = 0; i < step_index; ++i) {
            double term1 = memory_func(memory_coefficient, time - static_cast<double>(i)*delta_t)*p.vx[i];
            double term2 = memory_func(memory_coefficient, time - static_cast<double>(i + 1)*delta_t)*p.vx[i + 1];
            iprime_x += (term1 + term2)*(delta_t/2.0);
        }

        double iprime_y = 0.0;
        for (std::size_t i = 0; i < step_index; ++i) {
            double term1 = memory_func(memory_coefficient, time - static_cast<double>(i)*delta_t)*p.vy[i];
            double term2 = memory_func(memory_coefficient, time - static_cast<double>(i + 1)*delta_t)*p.vy[i + 1];
            iprime_x += (term1 + term2)*(delta_t/2.0);
        }

        double iprime_z = 0.0;
        for (std::size_t i = 0; i < step_index; ++i) {
            double term1 = memory_func(memory_coefficient, time - static_cast<double>(i)*delta_t)*p.vz[i];
            double term2 = memory_func(memory_coefficient, time - static_cast<double>(i + 1)*delta_t)*p.vz[i + 1];
            iprime_x += (term1 + term2)*(delta_t/2.0);
        }

        return {iprime_x, iprime_y, iprime_z};
    }

    void Simulator::write_coordinate() noexcept {
        out << "2" <<std::endl;
        out << std::endl;
        out << "C " << particle1.x << " " << particle1.y << " " << particle1.z << std::endl;
        out << "C " << particle2.x << " " << particle2.y << " " << particle2.z << std::endl;
    }

}//generalized_langevin

#endif