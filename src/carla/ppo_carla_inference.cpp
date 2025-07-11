// Code that gets the observations from eval_agent.py passes it throught the cpp model and sends the actions back.

#include <iostream>
#include <iomanip>
#include <tuple>
#include <filesystem>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <optional>
#include <regex>

#include <rl_utils.h>
#include <gymcpp/carla/carla_gym.h>
#include <distributed.h>
#include <carla/carla_model.h>
#include <carla/carla_config.h>
#include <args.hxx>

#include <torch/torch.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <zmq.hpp>

using namespace std;
using namespace torch;
using namespace torch::indexing;
using namespace std::literals;

int main(int argc, char** argv) {
  ios_base::sync_with_stdio(false);  // Faster print
  // TODO tune for CALRA Can be slightly faster to turn off multi-threading in libtorch. Might depend on model size.

  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);

  args::ArgumentParser parser("parser");
  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
  args::ValueFlag<string> path_to_conf_file_arg(parser, "path_to_conf_file", "Path to the folder containing model weights and config.json.", {"path_to_conf_file"}, args::Options::Required);
  args::ValueFlag<string> ipc_path_arg(parser, "ipc_path", "Path to folder that is used to sync inter process communication.", {"ipc_path"}, args::Options::Required);
  args::ValueFlag<int> port_arg(parser, "port", "Port to connect to eval_agent.py with.", {"port"}, args::Options::Required);

  try
  {
    parser.ParseCLI(argc, argv);
  }
  catch (const args::Help&)
  {
    cout << parser;
    throw std::runtime_error("Display help.");
  }
  catch (const args::ParseError& e)
  {
    cerr << e.what() << std::endl;
    cerr << parser;
    throw std::runtime_error("Could not parse arguments.");
  }

  // Update config with argparse arguments.
  // Unfortunately we have to do this manually for every variable since c++ doesn't support reflections.
  auto path_to_conf_file = args::get(path_to_conf_file_arg);
  auto ipc_path = args::get(ipc_path_arg);
  auto port = args::get(port_arg);

  GlobalConfig config;
  auto model_folder = filesystem::path(path_to_conf_file);
  auto config_path = model_folder / "config.json";
  config.update_from_json(config_path.string());

  zmq::context_t context;
  zmq::socket_t socket = zmq::socket_t(context, zmq::socket_type::pair);

  const filesystem::path comm_folder = filesystem::path(ipc_path);
  filesystem::create_directories(comm_folder);
  const filesystem::path full_path = comm_folder / (to_string(port) + ".lock");
  socket.bind("ipc://" + full_path.string());
  zmq::message_t msg("Connected to eval_agent.py."s);
  socket.send(msg, zmq::send_flags::none);
  cout << "Connecting to eval_agent.py, port: " << port << endl;
  zmq::message_t answer;
  const auto reply = socket.recv(answer, zmq::recv_flags::none);
  if(!reply) {
    cerr << "Establishing connected to eval_agent.py failed." << endl;
    return 1;
  }
  cout << "Connected to eval_agent.py, port: " << port << endl;

  cout << "Deterministic actions: " << answer.to_string() << endl;
  string sample_type = answer.to_string();

  auto device = Device(kCUDA, 0);
  cudaSetDevice(0);

  vector<Agent> agents;
  int num_models = 0;

  for (const auto& dirEntry : filesystem::recursive_directory_iterator(model_folder)) {
    auto filename = dirEntry.path().filename().string();
    if (filename.starts_with("model") and filename.ends_with(".pth")) {
      auto agent = Agent(config, CarlaEnv::action_space_, CarlaEnv::action_space_max_, CarlaEnv::action_space_min_);
      agent->to(device);
      agent->eval();
      torch::load(agent, dirEntry.path().string(), device);
      agents.push_back(agent);
      num_models++;
    }
  }

  if (num_models == 0) {
    cerr << "No model file was found in the selected path:" << model_folder.string() << endl;
    // socket.close();
    // context.close();
    return 2;
  }

  // // When running the agent in parallel we need to have each agent use its own seed generator, to ensure exact reproducibility.
  vector<Generator> agent_generators;
  for (int i = 0; i < num_models; ++i) {
    if (device.is_cuda()) {
      agent_generators.push_back(at::make_generator<CUDAGeneratorImpl>(config.seed + i));
    }
    else {
      agent_generators.push_back(at::make_generator<at::CPUGeneratorImpl>(config.seed + i));
    }
  }


  // Run until shutdown by eval_agent.py
  while (true) {
    // No grad
    NoGradGuard no_grad;

    zmq::message_t keepalive;
    const auto reply_keepalive = socket.recv(keepalive, zmq::recv_flags::none);
    if(!reply_keepalive) {
      cerr << "Connection to eval_agent.py interrupted." << endl;
      return 3;
    }
    if (keepalive.to_string() != ""s) {
      cout << "Finished route." << endl;
      break;
    }

    vector<zmq::message_t> recv_msgs;
    const zmq::recv_result_t result = zmq::recv_multipart(socket, back_inserter(recv_msgs));
    auto *p_bev_semantics = static_cast<uint8_t *>(recv_msgs[0].data());
    auto *p_measurements = static_cast<float *>(recv_msgs[1].data());
    auto *p_value_measurements = static_cast<float *>(recv_msgs[2].data());

    Tensor bev_semantics = torch::zeros({config.obs_num_channels, config.bev_semantics_height, config.bev_semantics_width}, kUInt8);
    Tensor measurements = torch::zeros({config.obs_num_measurements}, kCPU);
    Tensor value_measurements = torch::zeros({config.num_value_measurements}, kCPU);

    std::copy_n(p_bev_semantics, config.obs_num_channels * config.bev_semantics_height *  config.bev_semantics_width, bev_semantics.contiguous().data_ptr<uint8_t>());
    std::copy_n(p_measurements, config.obs_num_measurements, measurements.contiguous().data_ptr<float>());
    std::copy_n(p_value_measurements, config.num_value_measurements, value_measurements.contiguous().data_ptr<float>());

    bev_semantics = bev_semantics.unsqueeze(0).to(device, kFloat32);
    measurements = measurements.unsqueeze(0).to(device);
    value_measurements = value_measurements.unsqueeze(0).to(device);

    vector<Tensor> actions(num_models);
    vector<Tensor> values(num_models);
    vector<Tensor> mus(num_models);
    vector<Tensor> sigmas(num_models);
    for (int i = 0; i < num_models; ++i) { // TODO parallelize ensemble
      Tensor action, logprob, entropy, value, done, mu, sigma;
      tie(action, logprob, entropy, value, mu, sigma) = agents[i]->forward(bev_semantics, measurements, value_measurements, at::empty({0}), sample_type, agent_generators[i]);
      actions.at(i) = action;
      values.at(i) = value;
      mus.at(i) = mu;
      sigmas.at(i) = sigma;
    }

    Tensor action_ensemble = torch::mean(torch::stack(actions, 0), 0).squeeze(0).to(kCPU).contiguous();
    Tensor value_ensemble = torch::mean(torch::stack(values, 0), 0).squeeze(0).to(kCPU).contiguous();
    Tensor mu_ensemble = torch::mean(torch::stack(mus, 0), 0).squeeze(0).to(kCPU).contiguous();
    Tensor sigma_ensemble = torch::mean(torch::stack(sigmas, 0), 0).squeeze(0).to(kCPU).contiguous();

    // Send action
    auto action_accessor = action_ensemble.accessor<float,1>();
    auto value_accessor = value_ensemble.accessor<float,1>();
    auto mu_accessor = mu_ensemble.accessor<float,1>();
    auto sigma_accessor = sigma_ensemble.accessor<float,1>();

    zmq::message_t action_message(action_accessor.data(), CarlaEnv::action_space_ * sizeof(float));
    socket.send(action_message, zmq::send_flags::sndmore);
    zmq::message_t value_message(value_accessor.data(), 1 * sizeof(float));
    socket.send(value_message, zmq::send_flags::sndmore);
    zmq::message_t mu_message(mu_accessor.data(), CarlaEnv::action_space_ * sizeof(float));
    socket.send(mu_message, zmq::send_flags::sndmore);
    zmq::message_t sigma_message(sigma_accessor.data(), CarlaEnv::action_space_ * sizeof(float));
    socket.send(sigma_message, zmq::send_flags::none);

  }

  socket.close();
  context.close();
  return 0;
}
