#ifndef COMMON_H
#define COMMON_H

#include <optional>

#include <torch/torch.h>
#include <gymcpp/gym.h>

using namespace std;

class RecordEpisodeStatistics final : public EnvironmentWrapper
{
private:
    chrono::steady_clock::time_point episode_start_time_;
    float episode_return_ = 0.0f;
    int episode_length_ = 0;
    shared_ptr<Environment> env_;

public:

    explicit RecordEpisodeStatistics(const shared_ptr<Environment> &env): env_(env) {
        episode_start_time_ = chrono::steady_clock::now();
    }

    Tensor reset(const int seed) override {
        torch::NoGradGuard no_grad;
        Tensor obs = env_->reset(seed);

        episode_return_ = 0.0f;
        episode_length_ = 0;
        episode_start_time_ = chrono::steady_clock::now();
        return obs;
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

    tuple<Tensor, float, bool, bool, optional<env_info>> step(const Tensor& actions) override
    {
        torch::NoGradGuard no_grad;
        auto [observation, reward, termination, truncation] = env_->step(actions);

        optional<env_info> info = nullopt;

        episode_return_ += reward;
        episode_length_ += 1;

        if (termination or truncation)
        {
            const std::chrono::duration<float> episode_time_length = chrono::steady_clock::now() - episode_start_time_;
            info = env_info{ episode_return_, episode_length_, episode_time_length.count()};
        }

        return make_tuple(observation, reward, termination, truncation, info);
    }
};

// TODO unify with other wrappers
class RecordEpisodeStatisticsCarla final : public EnvironmentWrapperCarla
{
private:
    chrono::steady_clock::time_point episode_start_time_;
    float episode_return_ = 0.0f;
    int episode_length_ = 0;
    shared_ptr<EnvironmentCarla> env_;

public:

    explicit RecordEpisodeStatisticsCarla(const shared_ptr<EnvironmentCarla> &env): env_(env) {
        episode_start_time_ = chrono::steady_clock::now();
    }

    unordered_map<string, Tensor> reset(const int seed) override {
        torch::NoGradGuard no_grad;
        unordered_map<string, Tensor> obs = env_->reset(seed);

        episode_return_ = 0.0f;
        episode_length_ = 0;
        episode_start_time_ = chrono::steady_clock::now();
        return obs;
    }

    [[nodiscard]] vector<int> get_observation_space() const override {
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

    tuple<unordered_map<string, Tensor>, float, bool, bool, optional<env_info>> step(const Tensor& actions) override
    {
        torch::NoGradGuard no_grad;
        auto [observation, reward, termination, truncation] = env_->step(actions);

        optional<env_info> info = nullopt;

        episode_return_ += reward;
        episode_length_ += 1;

        if (termination or truncation)
        {
            const std::chrono::duration<float> episode_time_length = chrono::steady_clock::now() - episode_start_time_;
            info = env_info{ episode_return_, episode_length_, episode_time_length.count()};
        }

        return make_tuple(observation, reward, termination, truncation, info);
    }
};

#endif //COMMON_H
