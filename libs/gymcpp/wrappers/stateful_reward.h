#ifndef STATEFUL_REWARD_PY_H
#define STATEFUL_REWARD_PY_H

#include <cmath>
#include <iostream>

#include <torch/torch.h>
#include <gymcpp/gym.h>

class NormalizeReward final : public EnvironmentWrapper
{
private:
    shared_ptr<EnvironmentWrapper> env_;
    float mean_;
    float var_;
    float accumulated_reward_;
    float count_;
    float gamma_;
    float epsilon_;

public:
    bool update_running_mean_ = true;

    explicit NormalizeReward(const shared_ptr<EnvironmentWrapper> &env, const float gamma=0.99,
                             const at::ScalarType dtype=torch::kFloat32, const float epsilon=1e-8):
    env_(env)
    {
        mean_ = 0.0f;
        var_ = 1.0f;
        accumulated_reward_ = 0.0f;
        count_ = epsilon;
        gamma_ = gamma;
        epsilon_ = epsilon;
    }

    [[nodiscard]] int get_observation_space() const override {
        return env_->get_observation_space();
    }
    [[nodiscard]] int get_action_space() const override {
        return env_->get_action_space();
    }
    [[nodiscard]] float get_action_space_min() const override {
        return env_->get_action_space_min();
    }
    [[nodiscard]] float get_action_space_max() const override {
        return env_->get_action_space_max();
    }

    Tensor reset(const int seed) override {
        torch::NoGradGuard no_grad;
        const Tensor obs = env_->reset(seed);
        return obs;
    }

    tuple<Tensor, float, bool, bool, optional<env_info>> step(const Tensor& actions) override
    {
        torch::NoGradGuard no_grad;
        auto [obs, reward, termination, truncation, info] = env_->step(actions);

        accumulated_reward_ = accumulated_reward_ * gamma_ * (1.0f - static_cast<float>(termination)) + reward;

        return make_tuple(obs, normalize(reward), termination, truncation, info);
    }

    float normalize(const float reward) {
        if (update_running_mean_) {
            update(accumulated_reward_);
        }
        return reward / std::sqrt(var_ + epsilon_);
    }

    void update(const float x) {
        // Using float here since rewards are always of size 1.
        // the batch size is always 1 with our setup so the mean is just the sample and the variance is 0.
        constexpr float batch_var = 0.0f;
        constexpr float batch_count = 1.0f;

        const float delta = x - mean_;
        const float tot_count = count_ + batch_count;
        const float new_mean = mean_ + delta * batch_count / tot_count;
        const float m_a = var_ * count_;
        constexpr float m_b = batch_var * batch_count;
        const float M2 = m_a + m_b + (delta * delta) * count_ * batch_count / tot_count;
        const float new_var = M2 / tot_count;
        const float new_count = tot_count;

        count_ = new_count;
        mean_ = new_mean;
        var_ = new_var;

    }
};

#endif //STATEFUL_REWARD_PY_H
