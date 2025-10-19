#ifndef GYM_H
#define GYM_H
#include <memory>
#include <tuple>
#include <vector>
#include <future>
#include <optional>
#include <unordered_map>
#include <string>

#include <torch/torch.h>
#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio.hpp>

using namespace std;
using namespace torch;

struct env_info
{
    float r; // return
    int l;  // length of episode
    float t; // Episode time length in seconds.
};

class Environment {
public:
    virtual tuple<Tensor, float, bool, bool> step(const Tensor& action) = 0;
    virtual Tensor reset(int seed) = 0;
    [[nodiscard]] virtual int get_observation_space() const = 0;
    [[nodiscard]] virtual int get_action_space() const = 0;
    [[nodiscard]] virtual float get_action_space_min() const = 0;
    [[nodiscard]] virtual float get_action_space_max() const = 0;
    virtual ~Environment() = default;
};


class EnvironmentWrapper {
public:
    virtual tuple<Tensor, float, bool, bool, optional<env_info>> step(const Tensor& action) = 0;
    virtual Tensor reset(int seed) = 0;
    [[nodiscard]] virtual int get_observation_space() const = 0;
    [[nodiscard]] virtual int get_action_space() const = 0;
    [[nodiscard]] virtual float get_action_space_min() const = 0;
    [[nodiscard]] virtual float get_action_space_max() const = 0;
    virtual ~EnvironmentWrapper() = default;
};


class EnvironmentCarla {
public:
    virtual tuple<unordered_map<string, Tensor>, float, bool, bool> step(const Tensor& action) = 0;
    virtual unordered_map<string, Tensor> reset(int seed) = 0;
    [[nodiscard]] virtual vector<int> get_observation_space() const = 0;
    [[nodiscard]] virtual int get_action_space() const = 0;
    [[nodiscard]] virtual float get_action_space_min() const = 0;
    [[nodiscard]] virtual float get_action_space_max() const = 0;
    virtual ~EnvironmentCarla() = default;
};

class EnvironmentWrapperCarla {
public:
    virtual tuple<unordered_map<string, Tensor>, float, bool, bool, optional<env_info>> step(const Tensor& action) = 0;
    virtual unordered_map<string, Tensor> reset(int seed) = 0;
    [[nodiscard]] virtual vector<int> get_observation_space() const = 0;
    [[nodiscard]] virtual int get_action_space() const = 0;
    [[nodiscard]] virtual float get_action_space_min() const = 0;
    [[nodiscard]] virtual float get_action_space_max() const = 0;
    virtual ~EnvironmentWrapperCarla() = default;
};



// Comparable to gymnasiums SyncVectorEnv.
class SeqVectorEnv {
private:
    std::vector<shared_ptr<EnvironmentWrapper>> env_array_;
    Tensor obs_;
    Tensor rewards_;
    Tensor terminations_;
    Tensor truncations_;
    vector<optional<env_info>> infos_;
    vector<bool> autoreset_envs_;
    const bool clip_actions_;

public:
    const unsigned int num_envs_;

    SeqVectorEnv(const std::vector<shared_ptr<EnvironmentWrapper>>& env_array, const bool clip_actions):
    env_array_(env_array), clip_actions_(clip_actions), num_envs_(env_array.size())
    {
        for (int i = 0; i < num_envs_; ++i) {
            autoreset_envs_.push_back(false);
            infos_.emplace_back();
        }
        obs_ = torch::zeros({num_envs_, env_array_[0]->get_observation_space()});
        rewards_ = torch::zeros({num_envs_});
        terminations_ = torch::zeros({num_envs_});
        truncations_ = torch::zeros({num_envs_});
    }

    ~SeqVectorEnv() = default;

    Tensor reset(const int seed) {
        torch::NoGradGuard no_grad;
        for (int i = 0; i < env_array_.size(); ++i) {
            const auto next_obs = env_array_[i]->reset(seed+i);
            obs_.index({i}) = next_obs;
            autoreset_envs_[i] = false;
        }
        return obs_;
    }

    [[nodiscard]] unsigned int get_num_envs() const {
        return num_envs_;
    }

    [[nodiscard]] int get_observation_space() const {
        return env_array_[0]->get_observation_space();
    }
    [[nodiscard]] int get_action_space() const {
        return env_array_[0]->get_action_space();
    }
    [[nodiscard]] float get_action_space_min() const {
        return env_array_[0]->get_action_space_min();
    }
    [[nodiscard]] float get_action_space_max() const {
        return env_array_[0]->get_action_space_max();
    }

    tuple<Tensor, Tensor, Tensor, Tensor, vector<optional<env_info>>> step(const Tensor& actions) {
        torch::NoGradGuard no_grad;
        Tensor clipped_actions;
        if (clip_actions_) {
             clipped_actions = torch::clamp(actions, env_array_[0]->get_action_space_min(), env_array_[0]->get_action_space_max());
        }
        else {
            clipped_actions = actions;
        }
        for (int i = 0; i < env_array_.size(); ++i) {
            if (autoreset_envs_.at(i)) {
                // -1 == do not reseed the environment
                const auto next_obs = env_array_[i]->reset(-1);
                obs_.slice(0, i,i+1) = next_obs;
                rewards_.accessor<float,1>()[i] = 0.0f;
                terminations_.accessor<float,1>()[i] = static_cast<float>(false);
                truncations_.accessor<float,1>()[i] = static_cast<float>(false);
                infos_[i] = nullopt;
                autoreset_envs_.at(i) = false;
            }
            else {
                auto [next_obs, reward, termination, truncation, info] = env_array_[i]->step(clipped_actions.index({i}));
                obs_.slice(0, i,i+1) = next_obs;
                rewards_.accessor<float,1>()[i] = reward;
                terminations_.accessor<float,1>()[i] = static_cast<float>(termination);
                truncations_.accessor<float,1>()[i] = static_cast<float>(truncation);

                infos_[i] = info;
                autoreset_envs_.at(i) = (termination or truncation);
            }
        }
        return make_tuple(obs_, rewards_, terminations_, truncations_, infos_);
    }
};

// Comparable to gymnasiums SyncVectorEnv.
class SeqVectorEnvCarla {
private:
    std::vector<shared_ptr<EnvironmentWrapperCarla>> env_array_;
    unordered_map<string, Tensor> obs_;
    Tensor rewards_;
    Tensor terminations_;
    Tensor truncations_;
    vector<optional<env_info>> infos_;
    vector<bool> autoreset_envs_;
    const bool clip_actions_;

public:
    const unsigned int num_envs_;

    SeqVectorEnvCarla(const std::vector<shared_ptr<EnvironmentWrapperCarla>>& env_array, const bool clip_actions):
    env_array_(env_array), clip_actions_(clip_actions), num_envs_(env_array.size())
    {
        for (int i = 0; i < num_envs_; ++i) {
            autoreset_envs_.push_back(false);
            infos_.emplace_back();
        }
        auto obs_space = env_array_[0]->get_observation_space();
        obs_["bev_semantics"] = torch::zeros({num_envs_, obs_space[0], obs_space[1], obs_space[2]}, torch::kUInt8);
        obs_["measurements"] = torch::zeros({num_envs_, obs_space[3]});
        obs_["value_measurements"] = torch::zeros({num_envs_, obs_space[4]});
        rewards_ = torch::zeros({num_envs_});
        terminations_ = torch::zeros({num_envs_});
        truncations_ = torch::zeros({num_envs_});
    }

    ~SeqVectorEnvCarla() = default;

    unordered_map<string, Tensor> reset(const int seed) {
        torch::NoGradGuard no_grad;
        for (int i = 0; i < env_array_.size(); ++i) {
            auto next_obs = env_array_[i]->reset(seed+i);
            obs_["bev_semantics"].index_put_({i}, next_obs["bev_semantics"]);
            obs_["measurements"].index_put_({i}, next_obs["measurements"]);
            obs_["value_measurements"].index_put_({i}, next_obs["value_measurements"]);
            autoreset_envs_[i] = false;
        }
        return obs_;
    }

    [[nodiscard]] unsigned int get_num_envs() const {
        return num_envs_;
    }

    [[nodiscard]] vector<int> get_observation_space() const {
        return env_array_[0]->get_observation_space();
    }
    [[nodiscard]] int get_action_space() const {
        return env_array_[0]->get_action_space();
    }
    [[nodiscard]] float get_action_space_min() const {
        return env_array_[0]->get_action_space_min();
    }
    [[nodiscard]] float get_action_space_max() const {
        return env_array_[0]->get_action_space_max();
    }

    tuple<unordered_map<string, Tensor>, Tensor, Tensor, Tensor, vector<optional<env_info>>> step(const Tensor& actions) {
        torch::NoGradGuard no_grad;
        Tensor clipped_actions;
        if (clip_actions_) {
             clipped_actions = torch::clamp(actions, env_array_[0]->get_action_space_min(), env_array_[0]->get_action_space_max());
        }
        else {
            clipped_actions = actions;
        }
        for (int i = 0; i < env_array_.size(); ++i) {
            if (autoreset_envs_.at(i)) {
                // -1 == do not reseed the environment
                auto next_obs = env_array_[i]->reset(-1);
                obs_["bev_semantics"].index_put_({i}, next_obs["bev_semantics"]);
                obs_["measurements"].index_put_({i},next_obs["measurements"]);
                obs_["value_measurements"].index_put_({i}, next_obs["value_measurements"]);

                rewards_.accessor<float,1>()[i] = 0.0f;
                terminations_.accessor<float,1>()[i] = static_cast<float>(false);
                truncations_.accessor<float,1>()[i] = static_cast<float>(false);
                infos_[i] = nullopt;
                autoreset_envs_.at(i) = false;
            }
            else {
                auto [next_obs, reward, termination, truncation, info] = env_array_[i]->step(clipped_actions.index({i}));
                obs_["bev_semantics"].index_put_({i}, next_obs["bev_semantics"]);
                obs_["measurements"].index_put_({i}, next_obs["measurements"]);
                obs_["value_measurements"].index_put_({i}, next_obs["value_measurements"]);
                rewards_.accessor<float,1>()[i] = reward;
                terminations_.accessor<float,1>()[i] = static_cast<float>(termination);
                truncations_.accessor<float,1>()[i] = static_cast<float>(truncation);

                infos_[i] = info;
                autoreset_envs_.at(i) = (termination or truncation);
            }
        }

        unordered_map<string, Tensor> obs_copy;
        obs_copy["bev_semantics"] = obs_["bev_semantics"].detach().clone();
        obs_copy["measurements"] = obs_["measurements"].detach().clone();
        obs_copy["value_measurements"] = obs_["value_measurements"].detach().clone();

        return make_tuple(obs_copy, rewards_.detach().clone(), terminations_.detach().clone(), truncations_.detach().clone(), infos_);
    }
};


// Comparable to gymnasiums AsyncVectorEnv but implemented with threads instead of mutli-processes.
class ParVectorEnv {
private:
    std::vector<shared_ptr<EnvironmentWrapper>> env_array_;
    Tensor obs_;
    Tensor rewards_;
    Tensor terminations_;
    Tensor truncations_;
    // NOTE: All vectors can be written to in parallel except for vector<bool>, so I use int here.
    vector<int> autoreset_envs_;
    vector<optional<env_info>> infos_;
    const bool clip_actions_;
    std::unique_ptr<boost::asio::thread_pool> pool_;
    std::vector<std::future<void>> results_;

public:
    const unsigned int num_envs_;

    ParVectorEnv(const std::vector<shared_ptr<EnvironmentWrapper>>& env_array, const bool clip_actions):
    env_array_(env_array), clip_actions_(clip_actions), num_envs_(env_array.size())
    {
        for (int i = 0; i < num_envs_; ++i) {
            autoreset_envs_.push_back(false);
            results_.emplace_back();
            infos_.emplace_back();
        }
        obs_ = torch::zeros({num_envs_, env_array_[0]->get_observation_space()});
        rewards_ = torch::zeros({num_envs_});
        terminations_ = torch::zeros({num_envs_});
        truncations_ = torch::zeros({num_envs_});
        pool_ = make_unique<boost::asio::thread_pool>(num_envs_);
    }

    ~ParVectorEnv() = default;

    Tensor reset(const int seed) {
        torch::NoGradGuard no_grad;
        for (int i = 0; i < env_array_.size(); ++i) {
            const auto next_obs = env_array_[i]->reset(seed+i);
            obs_.index({i}) = next_obs;
            autoreset_envs_[i] = false;
        }
        return obs_;
    }

    [[nodiscard]] unsigned int get_num_envs() const {
        return num_envs_;
    }

    tuple<Tensor, Tensor, Tensor, Tensor, vector<optional<env_info>>> step(const Tensor& actions) {
        torch::NoGradGuard no_grad;
        Tensor clipped_actions;
        if (clip_actions_) {
             clipped_actions = torch::clamp(actions, env_array_[0]->get_action_space_min(), env_array_[0]->get_action_space_max());
        }
        else {
            clipped_actions = actions;
        }
        for (int i = 0; i < env_array_.size(); ++i) {
            // Important to copy i by value since it will change during the loop
            auto forward_env_i = [this, &clipped_actions, i]
            {
                if (autoreset_envs_[i]) {
                    // -1 == do not reseed the environment
                    const auto next_obs = env_array_[i]->reset(-1);
                    obs_.slice(0, i,i+1) = next_obs;
                    rewards_.accessor<float,1>()[i] = 0.0f;
                    terminations_.accessor<float,1>()[i] = static_cast<float>(false);
                    truncations_.accessor<float,1>()[i] = static_cast<float>(false);
                    infos_[i] = nullopt;
                    autoreset_envs_[i] = false;
                }
                else {
                    auto [next_obs, reward, termination, truncation, info] = env_array_[i]->step(clipped_actions.index({i}));
                    obs_.slice(0, i,i+1) = next_obs;
                    rewards_.accessor<float,1>()[i] = reward;
                    terminations_.accessor<float,1>()[i] = static_cast<float>(termination);
                    truncations_.accessor<float,1>()[i] = static_cast<float>(truncation);
                    infos_[i] = info;
                    autoreset_envs_[i] = (termination or truncation);
                }
            };
            results_.at(i) = boost::asio::post(pool_->get_executor(), boost::asio::use_future(forward_env_i));
        }
        // Wait until all environments are finished.
        for (int i = 0; i < env_array_.size(); ++i) {
            results_.at(i).get();
        }

        return make_tuple(obs_, rewards_, terminations_, truncations_, infos_);
    }
};

#endif //GYM_H
