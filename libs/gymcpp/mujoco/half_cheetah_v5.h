#ifndef HALF_CHEETAH_V5_H
#define HALF_CHEETAH_V5_H

#include <gymcpp/gym.h>
#include <gymcpp/mujoco/mujoco_env.h>
#include <random>
#include <string>
#include <tuple>
#include <torch/torch.h>
#include <cassert>
#include <chrono>
#include <thread>


using namespace std;
using namespace torch;

class HalfCheetahV5Env final: public Environment {
protected:
    MujocoEnv mujoco_env_;
    mjtNum ctrl_cost_weight_, forward_reward_weight_;
    uniform_real_distribution<> dist_;
    uniform_real_distribution<> dist_sleep_;
    normal_distribution<> vel_dist_;
    bool exclude_current_positions_from_observation_;
    Tensor obs_;
    std::mt19937 gen_;
    int max_episode_steps_{1000};
    int elapsed_step_{max_episode_steps_ + 1};

    static constexpr int observation_space_{17};
    static constexpr int action_space_{6};
    static constexpr float action_space_min_{-1.0};
    static constexpr float action_space_max_{1.0};

public:

    explicit HalfCheetahV5Env(const string& xml, const string& render_mode="rgb_array"s):
    mujoco_env_(xml, 5, true, render_mode), ctrl_cost_weight_(0.1),
    exclude_current_positions_from_observation_(true), forward_reward_weight_(1.0), dist_(-0.1, 0.1), dist_sleep_(0.0, 1.0),
    vel_dist_(0.0, 0.1)
    {
        assert((mujoco_env_.model_->nq == 9 and mujoco_env_.model_->nu == 6) && "An incorrect half_cheetah.xml file is loaded.");
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
            mujoco_env_.data_->qvel[i] = mujoco_env_.init_qvel_[i] + vel_dist_(gen_);
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

        // ctrl_cost
        mjtNum ctrl_cost = 0.0;
        for (int i = 0; i < mujoco_env_.model_->nu; ++i) {
            const auto double_act = mujoco_env_.data_->ctrl[i];
            ctrl_cost += ctrl_cost_weight_ * double_act * double_act;
        }

        const mjtNum forward_reward = forward_reward_weight_ * x_velocity;

        // reward and done
        const auto reward = static_cast<float>(forward_reward - ctrl_cost);
        ++elapsed_step_;
        const bool truncate = (elapsed_step_ >= max_episode_steps_);

        writeState();
        // float prob = dist_sleep_(gen_);
        // if (prob < 0.125) {
        //     std::this_thread::sleep_for(std::chrono::milliseconds(10));
        // }

        return make_tuple(obs_, reward, false, truncate);
    }

private:
    void writeState() const {
        auto obs_accessor = obs_.accessor<float,1>();
        int index = 0;
        for (int i = 1; i < mujoco_env_.model_->nq; ++i) {
            obs_accessor[index] = static_cast<float>(mujoco_env_.data_->qpos[i]);
            index++;
        }
        for (int i = 0; i < mujoco_env_.model_->nv; ++i) {
            obs_accessor[index] = static_cast<float>(mujoco_env_.data_->qvel[i]);
            index++;
        }
    }
};

#endif //HALF_CHEETAH_V5_H