#ifndef ANT_V5_H
#define ANT_V5_H

#include <gymcpp/gym.h>
#include <gymcpp/mujoco/mujoco_env.h>
#include <random>
#include <string>
#include <tuple>
#include <torch/torch.h>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <vector>

using namespace std;
using namespace torch;

class AntV5Env final: public Environment {
protected:
    MujocoEnv mujoco_env_;

    mjtNum ctrl_cost_weight_, forward_reward_weight_, contact_cost_weight_, healthy_reward_;
    int main_body_;
    bool terminate_when_unhealthy_;
    mjtNum healthy_z_range_min_, healthy_z_range_max_;
    mjtNum contact_force_min_, contact_force_max_, reset_noise_scale_;
    bool exclude_current_positions_from_observation_;
    bool include_cfrc_ext_in_observation;

    Tensor obs_;
    uniform_real_distribution<> dist_;
    normal_distribution<> vel_dist_;
    std::mt19937 gen_;
    int max_episode_steps_{1000};
    int elapsed_step_{max_episode_steps_ + 1};

    static constexpr int observation_space_{105};
    static constexpr int action_space_{8};
    static constexpr float action_space_min_{-1.0};
    static constexpr float action_space_max_{1.0};

public:

    explicit AntV5Env(const string& xml, const string& render_mode="rgb_array"s):
    mujoco_env_(xml, 5, true, render_mode), forward_reward_weight_(1.0), ctrl_cost_weight_(0.5),
    contact_cost_weight_(5e-4), healthy_reward_(1.0), main_body_(1), terminate_when_unhealthy_(true),
    healthy_z_range_min_(0.2), healthy_z_range_max_(1.0), contact_force_min_(-1.0), contact_force_max_(1.0),
    reset_noise_scale_(0.1), exclude_current_positions_from_observation_(true), include_cfrc_ext_in_observation(true),
    dist_(-reset_noise_scale_, reset_noise_scale_), vel_dist_(0.0, 1.0)
    {
        assert((mujoco_env_.model_->nq == 15 and mujoco_env_.model_->nu == 8) && "An incorrect ant.xml file is loaded.");
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
            mujoco_env_.data_->qvel[i] = mujoco_env_.init_qvel_[i] + reset_noise_scale_ * vel_dist_(gen_);
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
        const auto x_position_before = mujoco_env_.data_->xpos[3 * main_body_];
        mujoco_env_.mujocoStep(action);
        const auto x_position_after = mujoco_env_.data_->xpos[3 * main_body_];

        const mjtNum x_velocity = (x_position_after - x_position_before) / mujoco_env_.dt_;

        writeState();
        float reward = get_rew(x_velocity);
        bool terminated = (!is_healthy()) and (terminate_when_unhealthy_);

        // reward and done
        ++elapsed_step_;
        const bool truncate = (elapsed_step_ >= max_episode_steps_);

        return make_tuple(obs_, reward, terminated, truncate);
    }

private:
    [[nodiscard]] bool is_healthy() const
    {
        bool in_heath_range =    (mujoco_env_.data_->qpos[2] >= healthy_z_range_min_)
                              && (mujoco_env_.data_->qpos[2] <= healthy_z_range_max_);
        bool is_finite = true;
        for (int i = 0; i < mujoco_env_.model_->nq; ++i) {
            if (!std::isfinite(mujoco_env_.data_->qpos[i]))
            {
                is_finite = false; // found a NaN or Inf
                break;
            }
        }
        for (int i = 0; i < mujoco_env_.model_->nv; ++i) {
            if (!std::isfinite(mujoco_env_.data_->qvel[i]))
            {
                is_finite = false; // found a NaN or Inf
                break;
            }
        }
        return in_heath_range && is_finite;
    }

    [[nodiscard]] mjtNum healthy_reward() const
    {
        return static_cast<mjtNum>(is_healthy()) * healthy_reward_;
    }

    [[nodiscard]] vector<mjtNum> contact_forces() const
    {
        vector<mjtNum> clipped_contact_forces((mujoco_env_.model_->nbody - 1) * 6);
        for (int i = 6; i < mujoco_env_.model_->nbody * 6; ++i)
        {
            clipped_contact_forces[i-6] = std::clamp(mujoco_env_.data_->cfrc_ext[i],
                                                        contact_force_min_,
                                                        contact_force_max_);
        }
        return clipped_contact_forces;
    }

    [[nodiscard]] float get_rew(const mjtNum x_velocity) const
    {
        const mjtNum forward_rew = x_velocity * forward_reward_weight_;
        const mjtNum healthy_rew = healthy_reward();
        const mjtNum rewards = forward_rew + healthy_rew;
        // ctrl_cost
        mjtNum ctrl_cost = 0.0;
        for (int i = 0; i < mujoco_env_.model_->nu; ++i) {
            const auto double_act = mujoco_env_.data_->ctrl[i];
            ctrl_cost += ctrl_cost_weight_ * double_act * double_act;
        }

        mjtNum contact_cost = 0.0;
        const vector<mjtNum> clipped_contact_forces = contact_forces();
        for (const mjtNum force: clipped_contact_forces) {
            contact_cost += force * force;
        }
        contact_cost = contact_cost_weight_ * contact_cost;

        const mjtNum costs = ctrl_cost + contact_cost;

        const mjtNum reward = rewards - costs;

        return static_cast<float>(reward);
    }

    void writeState() const {
        auto obs_accessor = obs_.accessor<float,1>();
        int index = 0;
        for (int i = 2; i < mujoco_env_.model_->nq; ++i) {
            obs_accessor[index] = static_cast<float>(mujoco_env_.data_->qpos[i]);
            index++;
        }
        for (int i = 0; i < mujoco_env_.model_->nv; ++i) {
            obs_accessor[index] = static_cast<float>(mujoco_env_.data_->qvel[i]);
            index++;
        }
        vector<mjtNum> clipped_contact_forces = contact_forces();
        for (int i = 0; i < (mujoco_env_.model_->nbody - 1) * 6; ++i) {
            obs_accessor[index] = static_cast<float>(clipped_contact_forces[i]);
            index++;
        }
    }
};

#endif //ANT_V5_H