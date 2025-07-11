#ifndef CARLA_GYM_H_
#define CARLA_GYM_H_

#include <memory>
#include <filesystem>
#include <iostream>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <array>

#include <zmq.hpp>
#include <zmq_addon.hpp>
#include <torch/torch.h>

#include <gymcpp/gym.h>
#include <carla/carla_config.h>

using namespace std;
using namespace torch;


class CarlaEnv final : public EnvironmentCarla {
protected:
  zmq::context_t context_;
  zmq::socket_t socket_;
  bool initialized_ = false;
  const int port_;
  bool termination_ = false;
  bool truncation_ = false;
  string comm_root_;
  unordered_map<string, Tensor> obs_;
  GlobalConfig config_;

public:
  // channels, height, width, measurements, value_measurements
  vector<int> observation_space_ = vector<int>{9, 256, 256, 8, 10};  // TODO make configurable
  static constexpr int action_space_{2};
  static constexpr float action_space_min_{-1.0};
  static constexpr float action_space_max_{1.0};

  explicit CarlaEnv(const GlobalConfig& config, const string& comm_root, const int port) : port_(port), comm_root_(comm_root), config_(config)
  {
    observation_space_[0] = config_.obs_num_channels;
    observation_space_[1] = config_.bev_semantics_height;
    observation_space_[2] = config_.bev_semantics_width;
    observation_space_[3] = config_.obs_num_measurements;
    observation_space_[4] = config_.num_value_measurements;
    socket_ = zmq::socket_t(context_, zmq::socket_type::pair);
    const auto bev_options = TensorOptions().dtype(kUInt8);
    obs_ = {{"bev_semantics", torch::zeros({observation_space_[0], observation_space_[1], observation_space_[2]}, bev_options)},
              {"measurements", torch::zeros({observation_space_[3]})},
              {"value_measurements", torch::zeros({observation_space_[4]})}};
  }

  [[nodiscard]] vector<int> get_observation_space() const override {
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

  unordered_map<string, Tensor> reset(const int seed) override {
    // CARLA env is seeded in the python code

    if (initialized_ == false)
    {
      const filesystem::path comm_folder = filesystem::path(comm_root_) / "comm_files";
      filesystem::create_directories(comm_folder);
      const filesystem::path file(to_string(port_) + ".lock");
      const filesystem::path full_path = comm_folder / file;
      socket_.bind("ipc://" + full_path.string());
      cout << "Connecting to leaderboard gym, port: " << file.string() << endl;

      zmq::message_t msg;
      const auto init_result = socket_.recv(msg, zmq::recv_flags::none);
      if(!init_result) {
        throw runtime_error("Connection to CARLA leaderboard failed.");
      }
      cout << msg.to_string() << endl;
      initialized_ = true;
    }

    vector<zmq::message_t> recv_msgs;
    const zmq::recv_result_t result = zmq::recv_multipart(socket_, back_inserter(recv_msgs));
    auto *p_bev_semantics = static_cast<uint8_t *>(recv_msgs[0].data());
    auto *p_measurements = static_cast<float *>(recv_msgs[1].data());
    auto *p_value_measurements = static_cast<float *>(recv_msgs[2].data());
    const auto *p_reward = static_cast<float *>(recv_msgs[3].data());
    const auto *p_termination = static_cast<bool *>(recv_msgs[4].data());
    const auto *p_truncation = static_cast<bool *>(recv_msgs[5].data());
    const auto *p_n_steps = static_cast<int32_t *>(recv_msgs[6].data());
    const auto *p_suggest = static_cast<int32_t *>(recv_msgs[7].data());

    // Needs to be done before calling allocate because allocate check IsDone()
    termination_ = p_termination[0];
    truncation_ = p_truncation[0];

    assert(result && "recv failed");
    // Convert pointers to tensor.
    writeState(p_bev_semantics, p_measurements, p_value_measurements);

    return obs_;
  }


  tuple<unordered_map<string, Tensor>, float, bool, bool> step(const Tensor& action) override {
    // Send action
    auto action_accessor = action.accessor<float,1>();
    // Convert to a format that we can construct a zmq message from. Action spaces are usually small so this should be fast.
    std::array<float, action_space_> copy{};
    for (int i = 0; i < action_space_; ++i) {
      copy[i] = action_accessor[i];
    }
    zmq::message_t action_message(action_accessor.data(), action_space_ * sizeof(float));
    socket_.send(action_message, zmq::send_flags::none);

    // Receive next state
    vector<zmq::message_t> recv_msgs;
    const zmq::recv_result_t result = zmq::recv_multipart(socket_, back_inserter(recv_msgs));
    assert(result && "recv failed");

    auto *p_bev_semantics = static_cast<uint8_t *>(recv_msgs[0].data());
    auto *p_measurements = static_cast<float *>(recv_msgs[1].data());
    auto *p_value_measurements = static_cast<float *>(recv_msgs[2].data());
    const auto *p_reward = static_cast<float *>(recv_msgs[3].data());
    const auto *p_termination = static_cast<bool *>(recv_msgs[4].data());
    const auto *p_truncation = static_cast<bool *>(recv_msgs[5].data());
    // Special variables for Roach. Not implemented for now.
    const auto *p_n_steps = static_cast<int32_t *>(recv_msgs[6].data());
    const auto *p_suggest = static_cast<int32_t *>(recv_msgs[7].data());

    writeState(p_bev_semantics, p_measurements, p_value_measurements);
    return make_tuple(obs_, p_reward[0], p_termination[0], p_truncation[0]);
  }

  void writeState(uint8_t* p_bev_semantics, float* p_measurements, float* p_value_measurements) {
    std::copy_n(p_bev_semantics, observation_space_[0] * observation_space_[1] * observation_space_[2], obs_["bev_semantics"s].contiguous().data_ptr<uint8_t>());
    std::copy_n(p_measurements, observation_space_[3], obs_["measurements"s].contiguous().data_ptr<float>());
    std::copy_n(p_value_measurements, observation_space_[4], obs_["value_measurements"s].contiguous().data_ptr<float>());
  }
};

#endif  // CARLA_GYM_H_