#ifndef CARLA_GYM_H_
#define CARLA_GYM_H_

#include <memory>
#include <filesystem>
#include <iostream>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <array>
#include <chrono>

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
  array<int, 3> camera_shape_;
  array<int, 2> lidar_shape_;
  string comm_root_;
  unordered_map<string, Tensor> obs_;
  GlobalCarlaConfig config_;
  uint64_t num_recv_{0};

public:
  // channels, height, width, measurements, value_measurements
  vector<int> observation_space_ = vector<int>{9, 256, 256, 8, 10};
  static constexpr int action_space_{2};
  static constexpr float action_space_min_{-1.0};
  static constexpr float action_space_max_{1.0};

  explicit CarlaEnv(const GlobalCarlaConfig& config, const string& comm_root, const int port) : port_(port), comm_root_(comm_root), config_(config)
  {

    camera_shape_ = {3, config.cam_configs[0].height,
      config.cam_configs[0].width + config.cam_configs[1].width + config.cam_configs[2].width};
    lidar_shape_ = {static_cast<int>((config.lidar_max_x - config.lidar_min_x) * config.pixels_per_meter),
                    static_cast<int>((config.lidar_max_y - config.lidar_min_y) * config.pixels_per_meter)};

    observation_space_[0] = config_.obs_num_channels;
    observation_space_[1] = config_.bev_semantics_height;
    observation_space_[2] = config_.bev_semantics_width;
    observation_space_[3] = config_.obs_num_measurements;
    observation_space_[4] = config_.num_value_measurements;

    socket_ = zmq::socket_t(context_, zmq::socket_type::pair);
    const auto uint8_option = TensorOptions().dtype(kUInt8);
    obs_ = {{"bev_semantics", torch::zeros({observation_space_[0], observation_space_[1], observation_space_[2]}, uint8_option)},
              {"measurements", torch::zeros({observation_space_[3]})},
              {"value_measurements", torch::zeros({observation_space_[4]})}};

    if (config.use_sensorimotor == true) {
      observation_space_.push_back(camera_shape_[0]);  // idx 5
      observation_space_.push_back(camera_shape_[1]);  // idx 6
      observation_space_.push_back(camera_shape_[2]);  // idx 7
      observation_space_.push_back(lidar_shape_[0]);  // idx 8
      observation_space_.push_back(lidar_shape_[1]);  // idx 9
      observation_space_.push_back(1); // Compass idx 10
      observation_space_.push_back(1); // Speed idx 11
      observation_space_.push_back(2); // GPS idx 12
      observation_space_.push_back(2); // Target Point idx 13
      observation_space_.push_back(2); // Next Target Point idx 14

      obs_["rgb"] = torch::zeros({observation_space_[5], observation_space_[6], observation_space_[7]}, uint8_option);
      obs_["lidar"] = torch::zeros({observation_space_[8], observation_space_[9]}, uint8_option);
      obs_["compass"] = torch::zeros({observation_space_[10]});
      obs_["speed"] = torch::zeros({observation_space_[11]});
      obs_["gps"] = torch::zeros({observation_space_[12]});
      obs_["target_point"] = torch::zeros({observation_space_[13]});
      obs_["target_point_next"] = torch::zeros({observation_space_[14]});
    }
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
  [[nodiscard]] GlobalCarlaConfig get_config() const override {
    return config_;
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
    num_recv_ += 1;
    assert(result && "recv failed");

    for (auto& [key, tensor] : obs_) {
      tensor = tensor.contiguous();
    }
    const uint64_t *num_sent;
    if (config_.use_sensorimotor == true) {
      auto *p_bev_semantics = static_cast<uint8_t *>(recv_msgs[0].data());
      auto *p_measurements = static_cast<float *>(recv_msgs[1].data());
      auto *p_value_measurements = static_cast<float *>(recv_msgs[2].data());
      // Convert pointers to tensor.
      writeStatePrivileged(p_bev_semantics, p_measurements, p_value_measurements);

      auto *p_rgb = static_cast<uint8_t *>(recv_msgs[6].data());
      auto *p_lidar = static_cast<uint8_t *>(recv_msgs[7].data());
      auto *p_compass = static_cast<float *>(recv_msgs[8].data());
      auto *p_speed = static_cast<float *>(recv_msgs[9].data());
      auto *p_gps = static_cast<float *>(recv_msgs[10].data());
      auto *p_target_point = static_cast<float *>(recv_msgs[11].data());
      auto *p_target_point_next = static_cast<float *>(recv_msgs[12].data());
      writeStateSensorimotor(p_rgb, p_lidar, p_compass, p_speed, p_gps, p_target_point, p_target_point_next);

      num_sent = static_cast<uint64_t *>(recv_msgs[15].data());
    }
    else {
      auto *p_bev_semantics = static_cast<uint8_t *>(recv_msgs[0].data());
      auto *p_measurements = static_cast<float *>(recv_msgs[1].data());
      auto *p_value_measurements = static_cast<float *>(recv_msgs[2].data());

      const auto *p_n_steps = static_cast<int32_t *>(recv_msgs[6].data());
      const auto *p_suggest = static_cast<int32_t *>(recv_msgs[7].data());
      num_sent = static_cast<uint64_t *>(recv_msgs[8].data());

      // Convert pointers to tensor.
      writeStatePrivileged(p_bev_semantics, p_measurements, p_value_measurements);
    }

    if (num_recv_ != num_sent[0]) {
      throw std::invalid_argument("Communication breakdown, Leaderboard send more frames than client consumed. num_recv:" + std::to_string(num_recv_) +  ", num_sent: " + std::to_string(num_sent[0]));
    }


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
    zmq::message_t action_message(copy.data(), action_space_ * sizeof(float));
    socket_.send(action_message, zmq::send_flags::none);

    // Receive next state
    vector<zmq::message_t> recv_msgs;
    const zmq::recv_result_t result = zmq::recv_multipart(socket_, back_inserter(recv_msgs));
    assert(result && "recv failed");
    num_recv_ += 1;

    const uint64_t *num_sent;
    const float *p_reward;
    const bool *p_termination;
    const bool *p_truncation;

    for (auto& [key, tensor] : obs_) {
      tensor = tensor.contiguous();
    }

    if (config_.use_sensorimotor == true) {
      auto *p_bev_semantics = static_cast<uint8_t *>(recv_msgs[0].data());
      auto *p_measurements = static_cast<float *>(recv_msgs[1].data());
      auto *p_value_measurements = static_cast<float *>(recv_msgs[2].data());
      writeStatePrivileged(p_bev_semantics, p_measurements, p_value_measurements);

      p_reward = static_cast<float *>(recv_msgs[3].data());
      p_termination = static_cast<bool *>(recv_msgs[4].data());
      p_truncation = static_cast<bool *>(recv_msgs[5].data());

      auto *p_rgb = static_cast<uint8_t *>(recv_msgs[6].data());
      auto *p_lidar = static_cast<uint8_t *>(recv_msgs[7].data());
      auto *p_compass = static_cast<float *>(recv_msgs[8].data());
      auto *p_speed = static_cast<float *>(recv_msgs[9].data());
      auto *p_gps = static_cast<float *>(recv_msgs[10].data());
      auto *p_target_point = static_cast<float *>(recv_msgs[11].data());
      auto *p_target_point_next = static_cast<float *>(recv_msgs[12].data());
      writeStateSensorimotor(p_rgb, p_lidar, p_compass, p_speed, p_gps, p_target_point, p_target_point_next);

      num_sent = static_cast<uint64_t *>(recv_msgs[15].data());
      // auto start = static_cast<int64_t *>(recv_msgs[16].data())[0];
      // auto now = std::chrono::steady_clock::now();
      // uint64_t end = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
      // uint64_t diff_ns = end - start;
      // double diff_mu = diff_ns / 1000000000.0;
      // std::cout << "Transmission took: " << diff_mu << " seconds (s) \n";
    }
    else {
      auto *p_bev_semantics = static_cast<uint8_t *>(recv_msgs[0].data());
      auto *p_measurements = static_cast<float *>(recv_msgs[1].data());
      auto *p_value_measurements = static_cast<float *>(recv_msgs[2].data());
      writeStatePrivileged(p_bev_semantics, p_measurements, p_value_measurements);

      p_reward = static_cast<float *>(recv_msgs[3].data());
      p_termination = static_cast<bool *>(recv_msgs[4].data());
      p_truncation = static_cast<bool *>(recv_msgs[5].data());
      // Special variables for Roach. Not implemented for now.
      const auto *p_n_steps = static_cast<int32_t *>(recv_msgs[6].data());
      const auto *p_suggest = static_cast<int32_t *>(recv_msgs[7].data());
      num_sent = static_cast<uint64_t *>(recv_msgs[8].data());
    }

    if (num_recv_ != num_sent[0]) {
      throw std::invalid_argument("Communication breakdown, Leaderboard send more frames than client consumed. num_recv:" + std::to_string(num_recv_) +  ", num_sent: " + std::to_string(num_sent[0]));
    }

    return make_tuple(obs_, p_reward[0], p_termination[0], p_truncation[0]);
  }

  void writeStatePrivileged(uint8_t* p_bev_semantics, float* p_measurements, float* p_value_measurements) {
    std::copy_n(p_bev_semantics, observation_space_[0] * observation_space_[1] * observation_space_[2], obs_["bev_semantics"s].data_ptr<uint8_t>());
    std::copy_n(p_measurements, observation_space_[3], obs_["measurements"s].data_ptr<float>());
    std::copy_n(p_value_measurements, observation_space_[4], obs_["value_measurements"s].data_ptr<float>());
  }

  void writeStateSensorimotor(uint8_t* rgb, uint8_t* lidar, float* p_compass, float* p_speed, float* p_gps, float* p_target_point, float* p_target_point_next) {
    std::copy_n(rgb, observation_space_[5] * observation_space_[6] * observation_space_[7], obs_["rgb"s].data_ptr<uint8_t>());
    std::copy_n(lidar, observation_space_[8] * observation_space_[9], obs_["lidar"s].data_ptr<uint8_t>());
    std::copy_n(p_compass, observation_space_[10], obs_["compass"s].data_ptr<float>());
    std::copy_n(p_speed, observation_space_[11], obs_["speed"s].data_ptr<float>());
    std::copy_n(p_gps, observation_space_[12], obs_["gps"s].data_ptr<float>());
    std::copy_n(p_target_point, observation_space_[13], obs_["target_point"s].data_ptr<float>());
    std::copy_n(p_target_point_next, observation_space_[14], obs_["target_point_next"s].data_ptr<float>());
  }
};

#endif  // CARLA_GYM_H_