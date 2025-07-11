#ifndef HUMANOID_H
#define HUMANOID_H

#include <gymcpp/gym.h>
#include <gymcpp/mujoco/mujoco_env.h>
#include <random>
#include <string>
#include <tuple>
#include <torch/torch.h>
#include <cassert>

using namespace std;
using namespace torch;

class HumanoidV4Env final : public Environment {
protected:
    MujocoEnv mujoco_env_;
    bool terminate_when_unhealthy_, use_contact_force_;
    mjtNum ctrl_cost_weight_, forward_reward_weight_, healthy_reward_;
    mjtNum healthy_z_min_, healthy_z_max_;
    uniform_real_distribution<> dist_;
    Tensor obs_;
    std::mt19937 gen_;
    int max_episode_steps_{1000};
    int elapsed_step_{max_episode_steps_ + 1};

    static constexpr int observation_space_{376};
    static constexpr int action_space_{17};
    static constexpr float action_space_min_{-0.4};
    static constexpr float action_space_max_{0.4};

public:

    explicit HumanoidV4Env(const string& xml, const string& render_mode="rgb_array"s):
    mujoco_env_(xml, 5, true, render_mode), terminate_when_unhealthy_(true),
    use_contact_force_(false), ctrl_cost_weight_(0.1),
    forward_reward_weight_(1.25), healthy_reward_(5.0), healthy_z_min_(1.0), healthy_z_max_(2.0), dist_(-1e-2, 1e-2)
    {
        assert((mujoco_env_.model_->nq == 24 and mujoco_env_.model_->nu == 17) && "An incorrect humanoid.xml file is loaded.");
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
        const auto& before = getMassCenter();
        mujoco_env_.mujocoStep(action);
        const auto& after = getMassCenter();

        // ctrl_cost
        mjtNum ctrl_cost = 0.0;
        for (int i = 0; i < mujoco_env_.model_->nu; ++i) {
            const auto double_act = mujoco_env_.data_->ctrl[i];
            ctrl_cost += ctrl_cost_weight_ * double_act * double_act;
        }

        // xv and yv
        const mjtNum xv = (after[0] - before[0]) / mujoco_env_.dt_;
        // y velocity is not used in the reward so we don't compute it

        // reward and done
        const mjtNum healthy_reward = terminate_when_unhealthy_ or isHealthy() ? healthy_reward_ : 0.0;
        const auto reward = static_cast<float>(xv * forward_reward_weight_ + healthy_reward - ctrl_cost);
        ++elapsed_step_;
        const bool terminate = (terminate_when_unhealthy_ ? !isHealthy() : false);
        const bool truncate = (elapsed_step_ >= max_episode_steps_);

        writeState();

        return make_tuple(obs_, reward, terminate, truncate);
    }

private:
    [[nodiscard]] bool isHealthy() const {
        return healthy_z_min_ < mujoco_env_.data_->qpos[2] && mujoco_env_.data_->qpos[2] < healthy_z_max_;
    }

    [[nodiscard]] std::array<mjtNum, 2> getMassCenter() const {
        mjtNum mass_sum = 0.0;
        mjtNum mass_x = 0.0;
        mjtNum mass_y = 0.0;
        for (int i = 0; i < mujoco_env_.model_->nbody; ++i) {
            const mjtNum mass = mujoco_env_.model_->body_mass[i];
            mass_sum += mass;
            mass_x += mass * mujoco_env_.data_->xipos[i * 3 + 0];
            mass_y += mass * mujoco_env_.data_->xipos[i * 3 + 1];
        }
        return {mass_x / mass_sum, mass_y / mass_sum};
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
        for (int i = 0; i < 10 * mujoco_env_.model_->nbody; ++i) {
            obs_accessor[index] = static_cast<float>(mujoco_env_.data_->cinert[i]);
            index++;
        }
        for (int i = 0; i < 6 * mujoco_env_.model_->nbody; ++i) {
            obs_accessor[index] = static_cast<float>(mujoco_env_.data_->cvel[i]);
            index++;
        }
        for (int i = 0; i < mujoco_env_.model_->nv; ++i) {
            obs_accessor[index] = static_cast<float>(mujoco_env_.data_->qfrc_actuator[i]);
            index++;
        }
        for (int i = 0; i < 6 * mujoco_env_.model_->nbody; ++i) {
            obs_accessor[index] = static_cast<float>(mujoco_env_.data_->cfrc_ext[i]);
            index++;
        }
    }
};

#endif //HUMANOID_H