#ifndef TRANSFORM_OBSERVATION_H
#define TRANSFORM_OBSERVATION_H

#include <functional>
#include <optional>

#include <torch/torch.h>
#include <gymcpp/gym.h>

class TransformObservation final : public EnvironmentWrapper
{
  shared_ptr<EnvironmentWrapper> env_;
  std::function<Tensor(Tensor)> func_;

public:
  TransformObservation(const shared_ptr<EnvironmentWrapper> &env, const std::function<Tensor(Tensor)>& func):
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
    return observation(obs);
  }

  tuple<Tensor, float, bool, bool, optional<env_info>> step(const Tensor& actions) override
  {
    torch::NoGradGuard no_grad;
    auto [obs, reward, termination, truncation, info] = env_->step(actions);
    return make_tuple(observation(obs), reward, termination, truncation, info);
  }

  [[nodiscard]] Tensor observation(const Tensor& x) const {
    return func_(x);
  }
};
#endif //TRANSFORM_OBSERVATION_H
