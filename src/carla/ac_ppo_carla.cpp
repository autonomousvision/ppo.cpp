#include <iostream>
#include <iomanip>
#include <memory>
#include <tuple>
#include <numeric>
#include <random>
#include <chrono>
#include <filesystem>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <future>
#include <optional>
#include <unordered_map>
#include <regex>

#include <rl_utils.h>
#include <tictoc.h>
#include <gymcpp/gym.h>
#include <gymcpp/carla/carla_gym.h>
#include <gymcpp/wrappers/common.h>
// #include <gymcpp/wrappers/stateful_reward.h>
// #include <gymcpp/wrappers/vectorize_reward.h>
#include <distributed.h>
#include <tcp_store.h>
#include <carla/carla_model.h>
#include <carla/carla_config.h>

#include <mpi.h>
#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio.hpp>
#include <boost/format.hpp>
#include <torch/torch.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <tqdm/tqdm.hpp>
#include "tensorboard_logger.h"
#include <zmq.hpp>

using namespace std;
using namespace torch;
using namespace torch::indexing;
using namespace std::literals;

shared_ptr<EnvironmentWrapperCarla> make_env(const shared_ptr<EnvironmentCarla>& env_0, const GlobalConfig& config) {

  shared_ptr<EnvironmentWrapperCarla> env = make_shared<RecordEpisodeStatisticsCarla>(env_0);
  // TODO
  // if (config.normalize_rewards) {
  //   env = make_shared<NormalizeReward>(env, gamma);
  //   env = make_shared<TransformReward>(env, [](const float x){return std::clamp(x, -10.0f, 10.0f);});
  // }
  return env;
}

void save_state(const Agent& agent, const optim::Adam& optimizer, const filesystem::path& folder, const string& model_file, const string& optimizer_file, GlobalConfig global_config) {
  NoGradGuard no_grad;
  const filesystem::path model_path = folder / model_file;
  save(agent, model_path.string());

  const filesystem::path optimizer_path = folder / optimizer_file;
  save(optimizer, optimizer_path.string());

  auto config_file = folder / "config.json";
  std::ofstream out(config_file.string());
  out << global_config.to_json();
}

// main function
// For multi-device training start the program with: mpirun -n 2 --bind-to none gs_ppo_carla
int main(int argc, char** argv) {
  ios_base::sync_with_stdio(false);  // Faster print
  // TODO tune for CALRA Can be slightly faster to turn off multi-threading in libtorch. Might depend on model size.
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);

  // Initialize the MPI anc NCCL environment
  int rank = 0;
  int world_size = 1;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Get the number of processes
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // Get the rank of the process

  GlobalConfig config;
  // First update is just to get the load_file argument
  config.update_config_with_args(argc, argv, world_size);
  if (config.load_file != "None") {
    auto full_path_load_file = filesystem::path(config.load_file);
    auto load_file_name = full_path_load_file.parent_path() / "config.json";
    config.update_from_json(load_file_name.string());
  }
  // Arguments take priority over config, so we overwrite again.
  config.update_config_with_args(argc, argv, world_size);

  string config_json = config.to_json();

  auto local_rank = rank % config.num_devices;

  zmq::context_t context;
  vector<zmq::socket_t> sockets;
  // Send config to CARLA leaderboard processes:
  for (int i = 0; i < config.num_envs_per_proc; ++i) {
    sockets.push_back(zmq::socket_t(context, zmq::socket_type::pair));

    const filesystem::path comm_folder = filesystem::path(config.team_code_folder) / "comm_files";
    filesystem::create_directories(comm_folder);
    const filesystem::path file(to_string(config.ports[config.num_envs_per_proc * local_rank + i]) + ".conf_lock");
    const filesystem::path full_path = comm_folder / file;
    sockets[i].bind("ipc://" + full_path.string());
    zmq::message_t msg(config_json);
    sockets[i].send(msg, zmq::send_flags::none);
    cout << "Connecting to leaderboard gym, port: " << file.string() << endl;
    zmq::message_t answer;
    const auto config_result = sockets[i].recv(answer, zmq::recv_flags::none);
    if(!config_result) {
      throw runtime_error("Sending config failed.");
    }
    sockets[i].close();
  }
  context.close();



  cout << "world_size: " << world_size << "\n";
  cout << "rank: " << rank << "\n";
  cout << "local_rank: " << local_rank << endl;

  filesystem::path exp_folder(config.logdir);
  exp_folder = exp_folder / config.exp_name;
  if (rank == 0) {
    filesystem::create_directories(exp_folder);

    // Log config
    auto config_file = exp_folder / "config.json";
    std::ofstream out(config_file.string());
    out << config_json;
  }

  GOOGLE_PROTOBUF_VERIFY_VERSION;
  unique_ptr<TensorBoardLogger> logger;
  if (rank == 0) {
    // use current seconds as unique id.
    auto now = std::chrono::system_clock::now().time_since_epoch();
    auto now_id = std::chrono::duration_cast<std::chrono::seconds>(now).count();

    logger = make_unique<TensorBoardLogger>(exp_folder.string() + (boost::format("/events.out.tfevents.%d.pb") % now_id).str());
    logger->add_text("hyperparameters", 0, config_json.c_str());
  }

  // Seed libtorch
  manual_seed(static_cast<uint64_t>(config.seed));
  at::globalContext().setDeterministicCuDNN(config.torch_deterministic);
  at::globalContext().setDeterministicAlgorithms(config.torch_deterministic, true);

  // Debug entry for MPI debugging.
  // int i = 0;
  // while (!i)
  //   sleep(5);

  if (rank == 0) {
    cout << "Parallelization mehtod: " << get_parallel_info() << endl;
  }

  // Note: CPU is a lot faster than GPU in mujoco.
  auto collect_devices = Device(kCPU);
  auto train_devices = Device(kCPU);

  cout << "Rank :" << local_rank << "GPU id: " << config.gpu_ids.at(local_rank) << endl;
  if (config.collect_device == "cpu"s) {
    collect_devices = Device(kCPU);
  }
  else if (config.collect_device == "gpu"s) {
    collect_devices = Device(kCUDA, static_cast<DeviceIndex>(config.gpu_ids.at(local_rank)));
    cudaSetDevice(config.gpu_ids.at(local_rank));
  }
  else {
    cerr << "Unsupported Collect device selected. Options: cpu, gpu. Selected Device: " << config.collect_device << endl;
    return 2;
  }

  if (config.train_device == "cpu"s) {
    train_devices = Device(kCPU);
  }
  else if (config.train_device == "gpu"s) {
    train_devices = Device(kCUDA, static_cast<DeviceIndex>(config.gpu_ids.at(local_rank)));
    cudaSetDevice(config.gpu_ids.at(local_rank));
  }
  else {
    cerr << "Unsupported train device selected. Options: cpu, gpu. Selected device: " << config.train_device << endl;
    return 3;
  }

  // Note we might need something like this for multi-node training.
  // call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, local_comm, istat)
  // call MPI_Comm_rank(local_comm, local_rank, istat)
  shared_ptr<torchfort::Comm> comm;
  try {
    comm = make_shared<torchfort::Comm>(MPI_COMM_WORLD);
    comm->initialize((config.train_device == "gpu"s) or (config.collect_device == "gpu"s));
  }
  catch (const std::runtime_error& e) {
    std::cerr << e.what();
    return 6;
  }

  vector<shared_ptr<SeqVectorEnvCarla>> envs;

  for (int i = 0; i < config.num_envs_per_proc; ++i) {
    std::vector<shared_ptr<EnvironmentWrapperCarla>> env_array;

    auto env_0 = make_shared<CarlaEnv>(config, config.team_code_folder, config.ports[config.num_envs_per_proc * local_rank + i]);
    env_array.push_back(make_env(env_0, config));
    envs.push_back(make_shared<SeqVectorEnvCarla>(env_array, config.clip_actions));
  }

  auto obs_space = envs[0]->get_observation_space();
  auto agent = Agent(config, envs[0]->get_action_space(), envs[0]->get_action_space_max(), envs[0]->get_action_space_min());
  agent->to(collect_devices);

  int start_step = 0;
  if (config.load_file != "None") {
    auto full_path_load_file = filesystem::path(config.load_file);
    auto load_file_name = full_path_load_file.filename().string();
    regex number_regex("(\\d+)(?=\\.pth)");
    smatch match;
    if (regex_search(load_file_name, match, number_regex)) {
      start_step = stoi(match.str()) + 1; // The saved step was already completed
      cout << "Start training from step: " << start_step << endl;
    }
    torch::load(agent, full_path_load_file.string(), collect_devices);
  }

  // Having the same initialization ensures that distributing the model across gpus and aggregating the gradient
  // behaves the same as a bigger batch size on a single device.
  // Broadcast initial model parameters from rank 0
  for (auto& p : agent->parameters()) {
    comm->broadcast(p, 0);
  }

  auto optimizer = optim::Adam(agent->parameters(), optim::AdamOptions(config.learning_rate).eps(config.adam_eps).weight_decay(config.weight_decay).betas(std::make_tuple(config.beta_1, config.beta_2)));

  if (config.load_file != "None") {
    auto full_path_load_file = filesystem::path(config.load_file).string();
    auto optimizer_path = std::regex_replace(full_path_load_file, regex("\\model_"), "optimizer_");
    torch::load(optimizer, optimizer_path, collect_devices);
    if (rank == 0) {
      logger->add_scalar("charts/restart", config.global_step, 1.0f);
    }
  }

  if (rank == 0) {
    unsigned int num_params = 0;
    for (const auto& param: agent->parameters()) {
      if (param.requires_grad()) {
        num_params += param.numel();
      }
    }
    cout << "Number of parameters in model: " << num_params << endl;
  }

  // TCP store for preemption trick:
  auto tcp_server = make_shared<TCPStoreServer>(config.rdzv_addr, config.tcp_store_port, config.num_envs);
  if (rank == 0 and config.use_dd_ppo_preempt) {
    // Server store used for all threads and processes
      tcp_server->start();
  }

  boost::asio::thread_pool pool(config.num_envs_per_proc);
  vector<future<tuple<vector<env_info>, int>>> results;
  vector<shared_ptr<TCPStoreClient>> tcp_clients;
  for (int i = 0; i < config.num_envs_per_proc; ++i) {
    results.emplace_back();
    if (config.use_dd_ppo_preempt) {
      tcp_clients.push_back(make_shared<TCPStoreClient>(config.rdzv_addr, config.tcp_store_port));
    }
  }

  // Storage setup
  unordered_map<string, Tensor> obs;
  obs["bev_semantics"]      = torch::zeros({config.num_steps, config.num_envs_per_proc, obs_space[0], obs_space[1], obs_space[2]}, TensorOptions().dtype(kUInt8).device(collect_devices));
  obs["measurements"]       = torch::zeros({config.num_steps, config.num_envs_per_proc, obs_space[3]}, collect_devices);
  obs["value_measurements"] = torch::zeros({config.num_steps, config.num_envs_per_proc, obs_space[4]}, collect_devices);
  Tensor actions    = torch::zeros({config.num_steps, config.num_envs_per_proc, envs[0]->get_action_space()}, collect_devices);
  Tensor logprobs   = torch::zeros({config.num_steps, config.num_envs_per_proc}, collect_devices);
  Tensor rewards    = torch::zeros({config.num_steps, config.num_envs_per_proc}, collect_devices);
  Tensor dones      = torch::zeros({config.num_steps, config.num_envs_per_proc}, collect_devices);
  Tensor values     = torch::zeros({config.num_steps, config.num_envs_per_proc}, collect_devices);
  Tensor returns    = torch::zeros({config.num_steps, config.num_envs_per_proc}, collect_devices);
  Tensor advantages = torch::zeros({config.num_steps, config.num_envs_per_proc}, collect_devices);

  unordered_map<string, Tensor> next_obs;
  next_obs["bev_semantics"]      = torch::zeros({config.num_envs_per_proc, obs_space[0], obs_space[1], obs_space[2]}, TensorOptions().dtype(kUInt8).device(collect_devices));
  next_obs["measurements"]       = torch::zeros({config.num_envs_per_proc, obs_space[3]}, collect_devices);
  next_obs["value_measurements"] = torch::zeros({config.num_envs_per_proc, obs_space[4]}, collect_devices);
  Tensor next_done = torch::zeros({config.num_envs_per_proc}, collect_devices);

  std::deque<float> avg_returns;
  const int avg_returns_maxlen = 100;

  // When running the agent in parallel we need to have each agent use its own seed generator, to ensure exact reproducibility.
  vector<Generator> agent_generators;
  vector<at::cuda::CUDAStream> cuda_streams;
  for (int i = 0; i < config.num_envs_per_proc; ++i) {
    auto reset_obs = envs[i]->reset(config.seed+i);
    next_obs["bev_semantics"].index_put_({i}, reset_obs["bev_semantics"].squeeze(0).to(collect_devices, false, false));
    next_obs["measurements"].index_put_({i}, reset_obs["measurements"].squeeze(0).to(collect_devices, false, false));
    next_obs["value_measurements"].index_put_({i}, reset_obs["value_measurements"].squeeze(0).to(collect_devices, false, false));

    if (config.collect_device == "gpu"s) {
      agent_generators.push_back(at::make_generator<CUDAGeneratorImpl>(config.seed * (100 * rank) + i));
    }
    else {
      agent_generators.push_back(at::make_generator<at::CPUGeneratorImpl>(config.seed * (100 * rank) + i));
    }
    cuda_streams.push_back(at::cuda::getStreamFromPool(true, collect_devices.index()));
  }

  // Train indicies are typically on CPU.,
  Generator train_generator = at::make_generator<at::CPUGeneratorImpl>(config.seed * 1500 + rank);

  std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
  TicToc tt;

  long local_processed_samples = 0;
  float sychronize = 0.0f;
  for (auto iteration : tq::trange(start_step, config.num_iterations, (rank == 0))) {

    { // No grad
      NoGradGuard no_grad;
      c10::cuda::CUDACachingAllocator::emptyCache();
    }

    tt.tic();
    agent->eval();
    agent->to(collect_devices);

    if (rank == 0 and config.use_dd_ppo_preempt) {
      tcp_clients[0]->reset();
    }
    comm->allreduce(sychronize, false);

    if (config.lr_schedule == "linear"s) {
      float frac = 1.0f - static_cast<float>(iteration) / static_cast<float>(config.num_iterations);
      float lrnow = frac * config.learning_rate;
      auto &options = static_cast<optim::OptimizerOptions &>(optimizer.param_groups()[0].options());
      options.set_lr(lrnow);
    }

    for (int i = 0; i < config.num_envs_per_proc; ++i) {
      // We need to pass indices by value, so the parallel threads get the correct index.
      auto collect = [&, i] {
        vector<env_info> infos_vec;
        // We need to run the different threads on different cuda streams so that the GPU computation runs concurrently.
        at::cuda::CUDAStreamGuard guard(cuda_streams[i]);

        int step;
        for (step = 0; step < config.num_steps; ++step) { // No grad
          NoGradGuard no_grad;
          obs["bev_semantics"].index_put_({step, i}, next_obs["bev_semantics"].index({i}));
          obs["measurements"].index_put_({step, i}, next_obs["measurements"].index({i}));
          obs["value_measurements"].index_put_({step, i}, next_obs["value_measurements"].index({i}));
          dones.index_put_({step, i}, next_done.index({i}));

          Tensor action, logprob, entropy, value, done, mu, sigma;
          // 1.8ms
          tie(action, logprob, entropy, value, mu, sigma) = agent->forward(next_obs["bev_semantics"].index({i}).unsqueeze(0).to(collect_devices, false, false),
                                                                           next_obs["measurements"].index({i}).unsqueeze(0).to(collect_devices, false, false),
                                                                           next_obs["value_measurements"].index({i}).unsqueeze(0).to(collect_devices, false, false),
                                                                           at::empty({0}), "sample"s, agent_generators[i]);

          values.index_put_({step, i}, value.squeeze(0));
          actions.index_put_({step, i}, action.squeeze(0));
          logprobs.index_put_({step, i}, logprob.squeeze(0));
          // 20 ms
          auto [env_next_obs, reward, termination, truncation, infos] = envs[i]->step(action.to(kCPU, false, false));

          done = torch::logical_or(termination, truncation).to(kFloat32, false, false);

          rewards.index_put_({step, i}, reward.index({0}).to(collect_devices, false, false));
          next_obs["bev_semantics"].index_put_({i}, env_next_obs["bev_semantics"].index({0}).to(collect_devices, false, false));
          next_obs["measurements"].index_put_({i}, env_next_obs["measurements"].index({0}).to(collect_devices, false, false));
          next_obs["value_measurements"].index_put_({i}, env_next_obs["value_measurements"].index({0}).to(collect_devices, false, false));
          next_done.index_put_({i}, done.index({0}).to(collect_devices, false, false));

          for (const auto& info : infos)
          {
            if (info.has_value()) {
              auto x = info.value();
              infos_vec.push_back(x);
            }
          }

          if (config.use_dd_ppo_preempt) {
            const int num_done = tcp_clients[i]->get();
            const long min_steps = lround(config.dd_ppo_min_perc * static_cast<float>(config.num_steps));
            if (static_cast<float>(num_done) / static_cast<float>(config.num_envs) > config.dd_ppo_preempt_threshold
              and step > min_steps) {
              cout << "Rank: " << rank << ", Thread: " << i << ", Preempt at step: " << step << " Num done: " << num_done << '\n';
              break;
            }
          }
        }

        if (config.use_dd_ppo_preempt) {
          tcp_clients[i]->increment();
        }
        return make_tuple(infos_vec, step);
      };

      results.at(i) = boost::asio::post(pool, boost::asio::use_future(collect));
    }

    Tensor total_returns = torch::zeros({world_size}, TensorOptions().dtype(kFloat32).device(collect_devices).requires_grad(false));
    Tensor total_lengths = torch::zeros({world_size}, TensorOptions().dtype(kFloat32).device(collect_devices).requires_grad(false));
    Tensor num_total_returns = torch::zeros({world_size}, TensorOptions().dtype(kFloat32).device(collect_devices).requires_grad(false));
    int min_collected_steps = static_cast<int>(config.num_steps);
    // Collect returns for logging. Synchronizes threads.
    for (int i = 0; i < config.num_envs_per_proc; ++i) {
      const auto [infos, num_collected_steps] = results.at(i).get();
      if (num_collected_steps < min_collected_steps) {
        min_collected_steps = num_collected_steps;
      }
      for (const auto&[r, l, t]: infos) {
        total_returns.index_put_({rank}, total_returns.index({rank}) + r);
        total_lengths.index_put_({rank}, total_lengths.index({rank}) + static_cast<float>(l));
        num_total_returns.index_put_({rank}, num_total_returns.index({rank}) + 1.0f);
      }
    }

    int num_train_steps_per_env = min_collected_steps;

    // Average values across processes for logging.
    comm->allreduce(total_returns, false);
    comm->allreduce(total_lengths, false);
    comm->allreduce(num_total_returns, false);

    // NOTE: We count duplicated samples from dd_ppo_preempt as as collected samples
    config.global_step += config.num_envs * config.num_steps;
    local_processed_samples += config.num_envs * config.num_steps;

    // Log avg returns observed during data collection.
    if (rank == 0) {
      auto num_total_returns_all_processes = torch::sum(num_total_returns).cpu().item<float>();
      if (num_total_returns_all_processes > 0.0f) {
        auto total_returns_all_processes = torch::sum(total_returns).cpu().item<float>();
        auto total_lengths_all_processes = torch::sum(total_lengths).cpu().item<float>();
        float avg_return = total_returns_all_processes / num_total_returns_all_processes;
        float avg_length = total_lengths_all_processes / num_total_returns_all_processes;

        avg_returns.push_back(avg_return);
        if (avg_returns.size() > avg_returns_maxlen) {
          avg_returns.pop_front();
        }
        float windowed_avg_return = std::accumulate(avg_returns.begin(), avg_returns.end(), 0.0f) / static_cast<float>(avg_returns.size());

        logger->add_scalar("charts/episodic_return", static_cast<int>(config.global_step), avg_return);
        logger->add_scalar("charts/windowed_avg_return", static_cast<int>(config.global_step), windowed_avg_return);
        logger->add_scalar("charts/episodic_length", static_cast<int>(config.global_step), avg_length);

        cout << boost::format("global_step=%d, avg_episodic_return=%.2f \n") % config.global_step % avg_return;

        if (windowed_avg_return > config.max_training_score) {
          config.max_training_score = windowed_avg_return;
          if (config.best_iteration != iteration) {
            config.best_iteration = iteration;
            save_state(agent, optimizer, exp_folder, "model_best.pth"s, "optimizer_best.pth"s, config);
          }
        }
      }
    }


    tt.toc((boost::format("Rank: %d Time to collect data:") % rank).str());

    tt.tic();

    // Computes Generalized advantage estimation labels
    { // No grad
      NoGradGuard noGrad;
      const Tensor next_value = agent->get_value(next_obs["bev_semantics"], next_obs["measurements"], next_obs["value_measurements"]).flatten();
      Tensor lastgaelam = torch::zeros({config.num_envs_per_proc}, collect_devices);
      Tensor nextnonterminal = at::empty({0}, collect_devices);
      Tensor nextvalues = at::empty({0}, collect_devices);
      for (int t = num_train_steps_per_env-1; t >= 0; --t) {
        if(t == (config.num_steps-1)) {
          nextnonterminal = 1.0f - next_done;
          nextvalues = next_value;
        }
        else {
          nextnonterminal = 1.0 - dones.index({t + 1});
          nextvalues = values.index({t + 1});
        }
        Tensor delta = rewards.index({t}) + config.gamma * nextvalues * nextnonterminal - values.index({t});
        advantages.index_put_({t}, delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam);
        lastgaelam = advantages.index({t});
      }
      returns = advantages + values;
    }

    tt.toc((boost::format("Rank: %d Time to compute advantage:") % rank).str());

    tt.tic();
    unordered_map<string, Tensor> b_obs;
    b_obs["bev_semantics"]      = obs["bev_semantics"].index({Slice(None, num_train_steps_per_env)}).reshape({-1, obs_space[0], obs_space[1], obs_space[2]}).to(train_devices, false, false);
    b_obs["measurements"]       = obs["measurements"].index({Slice(None, num_train_steps_per_env)}).reshape({-1, obs_space[3]}).to(train_devices, false, false);
    b_obs["value_measurements"] = obs["value_measurements"].index({Slice(None, num_train_steps_per_env)}).reshape({-1, obs_space[4]}).to(train_devices, false, false);
    Tensor b_logprobs  = logprobs.index({Slice(None, num_train_steps_per_env)}).reshape({-1}).to(train_devices, false, false);
    Tensor b_actions = actions.index({Slice(None, num_train_steps_per_env)}).reshape({-1, envs[0]->get_action_space()}).to(train_devices, false, false);
    Tensor b_advantages = advantages.index({Slice(None, num_train_steps_per_env)}).reshape(-1).to(train_devices, false, false);
    Tensor b_returns = returns.index({Slice(None, num_train_steps_per_env)}).reshape(-1).to(train_devices, false, false);
    Tensor b_values = values.index({Slice(None, num_train_steps_per_env)}).reshape(-1).to(train_devices, false, false);
    long num_samples_collected = b_returns.sizes()[0];
    // Values defined outside the train loop, so that we can log the last value.
    Tensor v_loss;
    Tensor pg_loss;
    Tensor entropy_loss;
    Tensor old_approx_kl;
    Tensor approx_kl;
    vector<float> clipfracs;

    agent->train();
    agent->to(train_devices);

    // Optimize network
    for (int epoch = 0; epoch < config.update_epochs; ++epoch) {
      Tensor b_inds = torch::randperm(num_samples_collected, train_generator, TensorOptions().dtype(kLong).requires_grad(false));
      if (config.use_dd_ppo_preempt) {
        // For the missing sample we simply repeat existing once.
        long num_repeat = (config.batch_size_per_device + num_samples_collected - 1) / num_samples_collected; // Ceil
        b_inds = b_inds.repeat(num_repeat);
        b_inds = b_inds.index({Slice(None, config.batch_size_per_device)});
      }

      for (int start = 0; start < config.batch_size_per_device; start+=config.minibatch_per_device) {
        int end = start + config.minibatch_per_device;

        Tensor mb_inds = b_inds.index({Slice(start, end, 1)});
        auto [action, newlogprob, entropy, newvalue, mu, sigma] = agent->forward(b_obs["bev_semantics"].index({mb_inds}),
                                                                                              b_obs["measurements"].index({mb_inds}),
                                                                                              b_obs["value_measurements"].index({mb_inds}),
                                                                                              b_actions.index({mb_inds}));
        Tensor logratio = newlogprob - b_logprobs.index({mb_inds});
        Tensor ratio = logratio.exp();

        {  // No grad
          NoGradGuard no_grad;
          // calculate approx_kl http://joschu.net/blog/kl-approx.html
          old_approx_kl = (-logratio).mean();
          approx_kl = ((ratio - 1.0f) - logratio).mean();
          clipfracs.push_back(((ratio - 1.0f).abs() > config.clip_coef).to(kFloat, false, false).mean().item<float>());
        }

        Tensor mb_advantages = b_advantages.index({mb_inds});

        if (config.norm_adv) {
          NoGradGuard no_grad;

          // Distributed mean
          Tensor advantage_mean = mean(mb_advantages);
          vector<Tensor> means = {advantage_mean.contiguous()};
          comm->allreduce(means, true);
          advantage_mean = means[0];

          // Distributed standard deviation
          Tensor advantage_std = sum(square(mb_advantages - advantage_mean));
          vector<Tensor> stds = {advantage_std.contiguous()};
          comm->allreduce(stds, false);
          advantage_std = stds[0];
          // -1 is bessel's correction
          advantage_std = advantage_std / static_cast<float>(world_size * numel(mb_advantages) - 1);
          advantage_std = torch::sqrt(advantage_std);

          mb_advantages = (mb_advantages - advantage_mean) / (advantage_std + 1e-8);
        }

        // Policy loss
        Tensor pg_loss1 = -mb_advantages * ratio;
        Tensor pg_loss2 = -mb_advantages * torch::clamp(ratio, 1.0f - config.clip_coef, 1.0f + config.clip_coef);
        pg_loss = torch::max(pg_loss1, pg_loss2).mean();

        // Value loss
        newvalue = newvalue.view(-1);
        if(config.clip_vloss) {
          Tensor v_loss_unclipped = torch::pow(newvalue - b_returns.index({mb_inds}), 2);
          Tensor v_clipped = b_values.index({mb_inds}) + torch::clamp(newvalue - b_values.index({mb_inds}),
                                                                  -config.clip_coef,
                                                                  config.clip_coef);
          Tensor v_loss_clipped = torch::pow(v_clipped - b_returns.index({mb_inds}), 2);
          Tensor v_loss_max = torch::max(v_loss_unclipped, v_loss_clipped);
          v_loss = 0.5f * v_loss_max.mean();
        }
        else {
          v_loss = 0.5 * torch::pow(newvalue - b_returns.index({mb_inds}), 2).mean();
        }

        entropy_loss = entropy.mean();
        Tensor loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef;

        optimizer.zero_grad();
        loss.backward();

        // allreduce (average) gradients (if running distributed)
        std::vector<Tensor> grads;
        grads.reserve(agent->parameters().size());
        for (const auto& p : agent->parameters()) {
          if(p.requires_grad()) {
            grads.push_back(p.grad().contiguous());
          }
        }
        comm->allreduce(grads, true);

        nn::utils::clip_grad_norm_(agent->parameters(), config.max_grad_norm);
        optimizer.step();
      }
    }

    tt.toc((boost::format("Rank: %d Time to train") % rank).str());

    // Some of the logging tensors can have varying size across GPUs, need to bring them to a fixed size to allreduce.
    Tensor b_inds_original = torch::arange(num_train_steps_per_env * config.num_envs_per_proc);

    if (config.use_dd_ppo_preempt) {
      // For the missing sample we simply repeat existing once.
      long num_repeat = (config.batch_size_per_device + num_samples_collected - 1) / num_samples_collected; // Ceil
      b_inds_original = b_inds_original.repeat(num_repeat);
      b_inds_original = b_inds_original.index({Slice(None, config.batch_size_per_device)});
    }

    config.latest_iteration = iteration;
    // Average tensors across processes for logging.
    float avg_clipfrac = reduce(clipfracs.begin(), clipfracs.end()) / static_cast<float>(clipfracs.size());
    vector<Tensor> v_loss_v = {v_loss.contiguous()};
    vector<Tensor> pg_loss_v = {pg_loss.contiguous()};
    vector<Tensor> entropy_loss_v = {entropy_loss.contiguous()};
    vector<Tensor> old_approx_kl_v = {old_approx_kl.contiguous()};
    vector<Tensor> approx_kl_v = {approx_kl.contiguous()};
    vector<Tensor> b_returns_v = {b_returns.index({b_inds_original}).contiguous()};

    comm->allreduce(v_loss_v, true);
    comm->allreduce(pg_loss_v, true);
    comm->allreduce(entropy_loss_v, true);
    comm->allreduce(old_approx_kl_v, true);
    comm->allreduce(approx_kl_v, true);
    comm->allreduce(b_returns_v, true);
    comm->allreduce(avg_clipfrac, true);

    v_loss = v_loss_v[0];
    pg_loss = pg_loss_v[0];
    entropy_loss = entropy_loss_v[0];
    old_approx_kl = old_approx_kl_v[0];
    approx_kl = approx_kl_v[0];
    b_returns = b_returns_v[0];

    // Log agent and tensorboard values.
    if (rank == 0) {
      tt.tic();

      save_state(agent, optimizer, exp_folder,
        (boost::format("model_latest_%09d.pth") % iteration).str(),
        (boost::format("optimizer_latest_%09d.pth") % iteration).str(), config);

      // Cleanup files from past iterations
      for (const auto& dirEntry : filesystem::directory_iterator(exp_folder)) {
        const auto filename = dirEntry.path().filename().string();
        if (filename.starts_with("model_latest_") and filename.ends_with(".pth")) {
            if(filename != (boost::format("model_latest_%09d.pth") % iteration).str()) {
              filesystem::path old_model_file = exp_folder / filename;
              filesystem::remove(old_model_file);
            }
        }
        if (filename.starts_with("optimizer_latest_") and filename.ends_with(".pth")) {
          if(filename != (boost::format("optimizer_latest_%09d.pth") % iteration).str()) {
            filesystem::path old_model_file = exp_folder / filename;
            filesystem::remove(old_model_file);
          }
        }
      }
      chrono::high_resolution_clock::time_point current_time = chrono::high_resolution_clock::now();

      const float passed_seconds = std::chrono::duration<float>(current_time - start_time).count();
      float sps = 0.0f;
      if (passed_seconds > 0) { // If you divide by seconds and your code is too fast you get null division -_-
        sps = static_cast<float>(local_processed_samples) / passed_seconds;
        cout  << std::fixed << std::setprecision(0) << "SPS: " << sps << endl;
      }

      auto &options = static_cast<optim::OptimizerOptions &>(optimizer.param_groups()[0].options());
      logger->add_scalar("charts/learning_rate", static_cast<int>(config.global_step), options.get_lr());
      logger->add_scalar("losses/value_loss", static_cast<int>(config.global_step), v_loss.item<float>());
      logger->add_scalar("losses/policy_loss", static_cast<int>(config.global_step), pg_loss.item<float>());
      logger->add_scalar("losses/entropy", static_cast<int>(config.global_step), entropy_loss.item<float>());
      logger->add_scalar("losses/old_approx_kl", static_cast<int>(config.global_step), old_approx_kl.item<float>());
      logger->add_scalar("losses/approx_kl", static_cast<int>(config.global_step), approx_kl.item<float>());
      logger->add_scalar("losses/clipfrac", static_cast<int>(config.global_step), avg_clipfrac);
      logger->add_scalar("losses/discounted_returns", static_cast<int>(config.global_step), b_returns.mean().item<float>());
      logger->add_scalar("charts/SPS", static_cast<int>(config.global_step), sps);
      logger->add_scalar("charts/restart", config.global_step, 0.0f);

      tt.toc((boost::format("Rank: %d Time to log") % rank).str());
    }
    cout << std::flush;
  }
  if (rank == 0) {
    save_state(agent, optimizer, exp_folder, "model_final.pth"s, "optimizer_final.pth"s, config);
  }

  comm->allreduce(sychronize, false);
  pool.join();
  google::protobuf::ShutdownProtobufLibrary();
  comm->finalize();
  MPI_Finalize();
  return 0;
}
