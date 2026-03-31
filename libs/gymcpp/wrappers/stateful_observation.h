#ifndef STATEFUL_OBSERVATION_H
#define STATEFUL_OBSERVATION_H

#include <optional>

#include <torch/torch.h>
#include <gymcpp/gym.h>

class NormalizeObservation final : public EnvironmentWrapper
{
private:
    shared_ptr<EnvironmentWrapper> env_;
    torch::Tensor mean_;
    torch::Tensor var_;
    float count_;
    float epsilon_;

public:
    bool update_running_mean_ = true;

    NormalizeObservation(const shared_ptr<EnvironmentWrapper> &env, const int observation_space,
                         const at::ScalarType dtype=torch::kFloat64, const float epsilon=1e-4):
    env_(env)
    {
        mean_ = torch::zeros({observation_space}, dtype);
        var_ = torch::ones({observation_space}, dtype);
        count_ = epsilon;
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
        return observation(obs);
    }

    tuple<Tensor, float, bool, bool, optional<env_info>> step(const Tensor& actions) override
    {
        auto [obs, reward, termination, truncation, info] = env_->step(actions);
        return make_tuple(observation(obs), reward, termination, truncation, info);
    }

    Tensor observation(const Tensor& x) {
        torch::NoGradGuard no_grad;
        if (update_running_mean_) {
            update(x);
        }
        return (x - mean_) / torch::sqrt(var_ + epsilon_);
    }

    void update(const Tensor& x) {
        // the batch size is always 1 with our setup so the mean is just the sample and the variance is 0.
        const Tensor batch_mean = x;
        // Use unbiased=false to follow gymnasium which uses np.var which defaults to unbiased=0 (called ddof there)
        const Tensor batch_var = torch::zeros_like(x);  // Usually variance across batch size
        constexpr float batch_count = 1.0f;

        const Tensor delta = x - mean_;  // Usually mean across batch size instead of x
        const float tot_count = count_ + batch_count;
        const Tensor new_mean = mean_ + delta * batch_count / tot_count;
        const Tensor m_a = var_ * count_;
        const Tensor m_b = batch_var * batch_count;
        const Tensor M2 = m_a + m_b + (delta * delta) * count_ * batch_count / tot_count;
        const Tensor new_var = M2 / tot_count;
        const float new_count = tot_count;

        count_ = new_count;
        mean_ = new_mean;
        var_ = new_var;

    }
};

#endif //STATEFUL_OBSERVATION_H
