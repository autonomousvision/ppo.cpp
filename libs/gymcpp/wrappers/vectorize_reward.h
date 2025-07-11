#ifndef VECTORIZE_REWARD_H
#define VECTORIZE_REWARD_H

#include <torch/torch.h>
#include <gymcpp/gym.h>
#include <functional>

class TransformReward final : public EnvironmentWrapper
{
  shared_ptr<EnvironmentWrapper> env_;
  std::function<float(float)> func_;

public:
  TransformReward(const shared_ptr<EnvironmentWrapper> &env, const std::function<float(float)>& func):
  env_(env), func_(func) {
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
    return make_tuple(obs, func_(reward), termination, truncation, info);
  }
};

#endif //VECTORIZE_REWARD_H
