#ifndef HOPPER_V5_H
#define HOPPER_V5_H

#include <gymcpp/gym.h>
#include <gymcpp/mujoco/mujoco_env.h>
#include <random>
#include <string>
#include <tuple>
#include <torch/torch.h>
#include <cassert>
#include <algorithm>
#include <chrono>
#include <limits>

using namespace std;
using namespace torch;

class HopperV5Env final: public Environment {
protected:
    MujocoEnv mujoco_env_;

    mjtNum ctrl_cost_weight_, forward_reward_weight_, healthy_reward_;
    bool terminate_when_unhealthy_;
    mjtNum healthy_z_range_min_, healthy_z_range_max_;
    mjtNum healthy_state_range_min_, healthy_state_range_max_;
    mjtNum healthy_angle_range_min_, healthy_angle_range_max_, reset_noise_scale_;
    bool exclude_current_positions_from_observation_;

    Tensor obs_;
    uniform_real_distribution<> dist_;
    std::mt19937 gen_;
    int max_episode_steps_{1000};
    int elapsed_step_{max_episode_steps_ + 1};

    static constexpr int observation_space_{11};
    static constexpr int action_space_{3};
    static constexpr float action_space_min_{-1.0};
    static constexpr float action_space_max_{1.0};

public:

    explicit HopperV5Env(const string& xml, const string& render_mode="rgb_array"s):
    mujoco_env_(xml, 4, true, render_mode), forward_reward_weight_(1.0), ctrl_cost_weight_(1e-3),
    healthy_reward_(1.0), terminate_when_unhealthy_(true), healthy_state_range_min_(-100.0),
    healthy_state_range_max_(100.0), healthy_z_range_min_(0.7),
    healthy_z_range_max_(std::numeric_limits<mjtNum>::infinity()), healthy_angle_range_min_(-0.2),
    healthy_angle_range_max_(0.2), reset_noise_scale_(5e-3), exclude_current_positions_from_observation_(true),
    dist_(-reset_noise_scale_, reset_noise_scale_)
    {
        assert((mujoco_env_.model_->nq == 6 and mujoco_env_.model_->nu == 3) && "An incorrect hopper.xml file is loaded.");
        obs_ = torch::zeros({observation_space_});
        random_device rd;  // Will be used to obtain a seed for the random number engine
        gen_.seed(rd()); // Standard mersenne_twister_engine seeded with rd()
    }

    [[nodiscard]] int get_observation_space() const override {
        return observation_space_;
    }
    [[nodiscard]] int get_action_space() const override {
        return action_space_;
    }
    [[nodiscard]] float get_action_space_min() const override {
        return action_space_min_;
    }
    [[nodiscard]] float get_action_space_max() const override {
        return action_space_max_;
    }

    void mujocoResetModel() {
        for (int i = 0; i < mujoco_env_.model_->nq; ++i) {
            mujoco_env_.data_->qpos[i] = mujoco_env_.init_qpos_[i] + dist_(gen_);
        }
        for (int i = 0; i < mujoco_env_.model_->nv; ++i) {
            mujoco_env_.data_->qvel[i] = mujoco_env_.init_qvel_[i] + dist_(gen_);
        }
    }

    void mujocoReset() {
        mj_resetData(mujoco_env_.model_, mujoco_env_.data_);
        mujocoResetModel();
        mj_forward(mujoco_env_.model_, mujoco_env_.data_);
    }

    Tensor reset(const int seed) override {
        // Negative seed indicates we do not want to set the seed.
        if (seed > 0) {
            gen_.seed(seed);
        }

        elapsed_step_ = 0;
        mujocoReset();
        writeState();
        return obs_;
    }

    tuple<Tensor, float, bool, bool> step(const Tensor& action) override {
        const auto x_position_before = mujoco_env_.data_->qpos[0];
        mujoco_env_.mujocoStep(action);
        const auto x_position_after = mujoco_env_.data_->qpos[0];

        const mjtNum x_velocity = (x_position_after - x_position_before) / mujoco_env_.dt_;

        writeState();
        const bool currently_healthy = is_healthy();
        float reward = get_rew(x_velocity, currently_healthy);
        bool terminated = (!currently_healthy) and (terminate_when_unhealthy_);

        // reward and done
        ++elapsed_step_;
        const bool truncate = (elapsed_step_ >= max_episode_steps_);

        return make_tuple(obs_, reward, terminated, truncate);
    }

private:
    [[nodiscard]] bool is_healthy() const
    {
        bool healthy = true;
        const mjtNum z = mujoco_env_.data_->qpos[1];
        const mjtNum angle = mujoco_env_.data_->qpos[2];

        if (z < healthy_z_range_min_ || z > healthy_z_range_max_) {
            healthy = false;
        }

        if (angle < healthy_angle_range_min_ || angle > healthy_angle_range_max_) {
            healthy = false;
        }

        for (int i = 2; i < mujoco_env_.model_->nq; ++i) {
            if (mujoco_env_.data_->qpos[i] < healthy_state_range_min_) {
                healthy = false;
            }
            if (mujoco_env_.data_->qpos[i] > healthy_state_range_max_) {
                healthy = false;
            }
        }
        for (int i = 0; i < mujoco_env_.model_->nv; ++i) {
            if (mujoco_env_.data_->qvel[i] < healthy_state_range_min_) {
                healthy = false;
            }
            if (mujoco_env_.data_->qvel[i] > healthy_state_range_max_) {
                healthy = false;
            }
        }

        return healthy;
    }

    [[nodiscard]] mjtNum healthy_reward(const bool currently_healthy) const
    {
        return static_cast<mjtNum>(currently_healthy) * healthy_reward_;
    }

    [[nodiscard]] float get_rew(const mjtNum x_velocity, const bool currently_healthy) const
    {
        const mjtNum forward_rew = x_velocity * forward_reward_weight_;
        const mjtNum healthy_rew = healthy_reward(currently_healthy);
        const mjtNum rewards = forward_rew + healthy_rew;
        // ctrl_cost
        mjtNum ctrl_cost = 0.0;
        for (int i = 0; i < mujoco_env_.model_->nu; ++i) {
            const auto double_act = mujoco_env_.data_->ctrl[i];
            ctrl_cost += ctrl_cost_weight_ * double_act * double_act;
        }

        const mjtNum reward = rewards - ctrl_cost;

        return static_cast<float>(reward);
    }

    void writeState() const {
        auto obs_accessor = obs_.accessor<float,1>();
        int index = 0;
        for (int i = 1; i < mujoco_env_.model_->nq; ++i) {
            obs_accessor[index] = static_cast<float>(mujoco_env_.data_->qpos[i]);
            index++;
        }
        for (int i = 0; i < mujoco_env_.model_->nv; ++i) {
            obs_accessor[index] = static_cast<float>(std::clamp(mujoco_env_.data_->qvel[i], -10.0, 10.0));
            index++;
        }
    }
};

#endif //HOPPER_V5_H