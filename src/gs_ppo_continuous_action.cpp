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
#include <cassert>
#include <optional>

#include <rl_utils.h>
#include <tictoc.h>
#include <gymcpp/gym.h>
#include <gymcpp/mujoco/humanoid_v4.h>
#include <gymcpp/mujoco/half_cheetah_v5.h>
#include <gymcpp/wrappers/common.h>
#include <gymcpp/wrappers/stateful_observation.h>
#include <gymcpp/wrappers/transform_observation.h>
#include <gymcpp/wrappers/stateful_reward.h>
#include <gymcpp/wrappers/vectorize_reward.h>
#include <distributed.h>
#include <tcp_store.h>

#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio.hpp>
#include <boost/format.hpp>
#include <torch/torch.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <tqdm/tqdm.hpp>
#include "tensorboard_logger.h"
#include <args.hxx>
#include <mpi.h>

using namespace std;
using namespace torch;
using namespace torch::indexing;

shared_ptr<EnvironmentWrapper> make_env(const shared_ptr<Environment>& env_0, float gamma) {
  auto env_1 = make_shared<RecordEpisodeStatistics>(env_0);
  auto env_2 = make_shared<NormalizeObservation>(env_1, env_1->get_observation_space(), kFloat32);
  auto env_3 = make_shared<TransformObservation>(env_2, [](const Tensor& x){return torch::clamp(x, -10.0f, 10.0f);});
  auto env_4 = make_shared<NormalizeReward>(env_3, gamma);
  auto env_5 = make_shared<TransformReward>(env_4, [](const float x){return std::clamp(x, -10.0f, 10.0f);});

  return env_5;
}

class GlobalConfig {
public:
  int seed = 1;  // negative values mean no seeding will take place.
  int eval_seed = 2;  // negative values mean no seeding will take place.
  unsigned int total_timesteps = 1'000'000;
  float learning_rate = 3e-4;
  unsigned int num_envs = 1;
  unsigned int num_steps = 2048;
  float gamma = 0.99;
  float gae_lambda = 0.95;
  unsigned int num_minibatches = 32;
  unsigned int update_epochs = 10;
  bool norm_adv = true;
  float clip_coef = 0.2;
  bool clip_vloss = true;
  float ent_coef = 0.0;
  float vf_coef = 0.5;
  float max_grad_norm = 0.5;
  float adam_eps = 1e-5;
  bool anneal_lr = true;
  unsigned int num_eval_runs = 10;
  bool clip_actions = true;
  bool torch_deterministic = true;
  string exp_name_stem = "PPO_002"s;
  string env_id = "Humanoid-v4"s;
  vector<int> gpu_ids = vector<int>({0});
  string collect_device = "cpu"s;  // Options: cpu, gpu
  string train_device = "cpu"s;  // Options: cpu, gpu
  string rdzv_addr = "localhost"s;  // IP adress of master node.
  uint16_t tcp_store_port = 29500u;  // Port for the TCP store.
  int use_dd_ppo_preempt = 0;  // Flag to toggle the dd_ppo pre-emption trick.
  float dd_ppo_min_perc = 0.25;  // Minimum percentage of env steps that need to finish before preemtion.
  float dd_ppo_preempt_threshold = 0.6;  // Minimum number of envs that need to finish before preemtion.

  // These variables are dependent on the other variables and need to be recomputed if they change.
  unsigned int num_devices = gpu_ids.size();
  string exp_name = (boost::format("%s_%s") % exp_name_stem % seed).str();
  unsigned int batch_size {num_steps * num_envs};
  unsigned int minibatch_size {batch_size / num_minibatches};
  unsigned int num_iterations {total_timesteps / batch_size};
  unsigned int num_envs_per_device = num_envs / num_devices;
  unsigned int batch_size_per_device = batch_size / num_devices;
  unsigned int minibatch_per_device = minibatch_size / num_devices;


  [[nodiscard]] string to_string() const {
    return (boost::format("|param|value|\n"
                  "|-|-|\n"
                  "|seed|%s|\n"
                  "|eval_seed|%s|\n"
                  "|total_timesteps|%s|\n"
                  "|learning_rate|%s|\n"
                  "|num_envs|%s|\n"
                  "|num_steps|%s|\n"
                  "|gamma|%s|\n"
                  "|gae_lambda|%s|\n"
                  "|num_minibatches|%s|\n"
                  "|update_epochs|%s|\n"
                  "|norm_adv|%s|\n"
                  "|clip_coef|%s|\n"
                  "|clip_vloss|%s|\n"
                  "|ent_coef|%s|\n"
                  "|vf_coef|%s|\n"
                  "|max_grad_norm|%s|\n"
                  "|adam_eps|%s|\n"
                  "|anneal_lr|%s|\n"
                  "|num_eval_runs|%s|\n"
                  "|clip_actions|%s|\n"
                  "|exp_name|%s|\n"
                  "|batch_size|%s|\n"
                  "|minibatch_size|%s|\n"
                  "|num_iterations|%s|\n"
                  "|torch_deterministic|%s|\n"
                  "|exp_name_stem|%s|\n"
                  "|env_id|%s|\n"
                  "|collect_device|%s|\n"
                  "|train_device|%s|\n"
                  "|num_envs_per_device|%s|\n"
                  "|batch_size_per_device|%s|\n"
                  "|use_dd_ppo_preempt|%s|\n"
                  "|dd_ppo_min_perc|%s|\n"
                  "|dd_ppo_preempt_threshold|%s|\n"
                  "|minibatch_per_device|%s|\n")
                  % seed % eval_seed % total_timesteps % learning_rate % num_envs % num_steps % gamma % gae_lambda %
                  num_minibatches % update_epochs % norm_adv % clip_coef % clip_vloss % ent_coef % vf_coef % max_grad_norm %
                  adam_eps % anneal_lr % num_eval_runs % clip_actions % exp_name % batch_size % minibatch_size %
                  num_iterations % torch_deterministic % exp_name_stem % env_id % collect_device % train_device %
                  num_envs_per_device % batch_size_per_device % use_dd_ppo_preempt % dd_ppo_min_perc %
                  dd_ppo_preempt_threshold % minibatch_per_device).str();
  }
};

class AgentImpl final : public nn::Module {
public:
    explicit AgentImpl(int observation_space, int action_space)
    {
        critic = nn::Sequential(orthogonal_init(nn::Linear(observation_space, 64)),
            nn::Tanh(),
            orthogonal_init(nn::Linear(64, 64)),
            nn::Tanh(),
            orthogonal_init(nn::Linear(64, 1), 1.0));
        register_module("critic", critic);

        actor_mean = nn::Sequential(orthogonal_init(nn::Linear(observation_space, 64)),
            nn::Tanh(),
            orthogonal_init(nn::Linear(64, 64)),
            nn::Tanh(),
            orthogonal_init(nn::Linear(64, action_space), 0.01));
        register_module("actor_mean", actor_mean);
        actor_logstd = register_parameter("actor_logstd", torch::zeros({1, action_space}, torch::kFloat32));
    }
    Tensor get_value(Tensor x) {
        x = critic->forward(x);
        return x;
    }

    tuple<Tensor, Tensor, Tensor, Tensor> get_action_and_value(const Tensor& x, Tensor action=at::empty({0}), const optional<Generator>& generator=nullopt) {
        const Tensor action_mean = actor_mean->forward(x);
        const Tensor action_logstd = actor_logstd.expand_as(action_mean);
        const Tensor action_std = torch::exp(action_logstd);
        const Normal probs(action_mean, action_std);
        if (action.data_ptr() == nullptr) {
            action = probs.sample(generator);
        }
        Tensor logprob = probs.log_prob(action).sum(1);
        Tensor entropy = probs.entropy().sum(1);
        Tensor value = critic->forward(x);
        return make_tuple(action, logprob, entropy, value);
    }
protected:
    static nn::Linear orthogonal_init(nn::Linear linear, const double std=std::sqrt(2.0), const float bias_const=0.0) {
        NoGradGuard noGrad;
        nn::init::orthogonal_(linear->weight, std);
        nn::init::constant_(linear->bias, bias_const);
        return linear;
    }

    nn::Sequential critic{nullptr};
    nn::Sequential actor_mean{nullptr};
    Tensor actor_logstd;
};

TORCH_MODULE(Agent);

void save_state(const Agent& agent, const optim::Adam& optimizer, const filesystem::path& folder, const string& model_file, const string& optimizer_file) {
  NoGradGuard no_grad;
  const filesystem::path model_path = folder / model_file;
  save(agent, model_path.string());

  const filesystem::path optimizer_path = folder / optimizer_file;
  save(optimizer, optimizer_path.string());
}

// main function
// For multi-device training start the program with: mpirun -n 2 --bind-to none gs_ppo_carla
int main(const int argc, const char** argv) {
  auto program_location = filesystem::canonical("/proc/self/exe"); // Linux only
  std::string directory = std::filesystem::path(program_location).parent_path().string();
  cout << "Location of program: " << directory << endl;
  ios_base::sync_with_stdio(false);  // Faster print
  // Can be slightly faster to turn off multi-threading in libtorch. Might depend on model size.
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);

  // Initialize the MPI anc NCCL environment
  int rank;
  int world_size;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Get the number of processes
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // Get the rank of the process

  GlobalConfig config;

  args::ArgumentParser parser("parser");
  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
  args::ValueFlag seed(parser, "seed", "Training seed", {"seed"}, config.seed);
  args::ValueFlag eval_seed(parser, "eval_seed", "Seed of final evaluation run", {"eval_seed"}, config.eval_seed);
  args::ValueFlag total_timesteps(parser, "total_timesteps", "Number of environment steps", {"total_timesteps"}, config.total_timesteps);
  args::ValueFlag learning_rate(parser, "learning_rate", "Adam learning rate", {"learning_rate"}, config.learning_rate);
  args::ValueFlag num_steps(parser, "num_steps", "Num environment steps per iteration", {"num_steps"}, config.num_steps);
  args::ValueFlag gamma(parser, "gamma", "Discount factor", {"gamma"}, config.gamma);
  args::ValueFlag gae_lambda(parser, "gae_lambda", "Lambda of generalized advantage estimation", {"gae_lambda"}, config.gae_lambda);
  args::ValueFlag num_minibatches(parser, "num_minibatches", "Number of training iterations per epoch", {"num_minibatches"}, config.num_minibatches);
  args::ValueFlag update_epochs(parser, "update_epochs", "Number of training epochs per iteration", {"update_epochs"}, config.update_epochs);
  args::ValueFlag norm_adv(parser, "norm_adv", "Whether to normalize the advantage", {"norm_adv"}, config.norm_adv);
  args::ValueFlag clip_coef(parser, "clip_coef", "PPO clip coefficient", {"clip_coef"}, config.clip_coef);
  args::ValueFlag clip_vloss(parser, "clip_vloss", "Whether to apply clipping to the value loss", {"clip_vloss"}, config.clip_vloss);
  args::ValueFlag ent_coef(parser, "ent_coef", "Weigth of entropy loss.", {"ent_coef"}, config.ent_coef);
  args::ValueFlag vf_coef(parser, "vf_coef", "Weigth of value loss.", {"vf_coef"}, config.vf_coef);
  args::ValueFlag max_grad_norm(parser, "max_grad_norm", "Factor for gradient clipping.", {"max_grad_norm"}, config.max_grad_norm);
  args::ValueFlag adam_eps(parser, "adam_eps", "Epsilon of adam.", {"adam_eps"}, config.adam_eps);
  args::ValueFlag anneal_lr(parser, "anneal_lr", "Whether to anneal the learning rate linearly.", {"anneal_lr"}, config.anneal_lr);
  args::ValueFlag num_eval_runs(parser, "num_eval_runs", "How many environments to evaluate", {"num_eval_runs"}, config.num_eval_runs);
  args::ValueFlag clip_actions(parser, "clip_actions", "Whether to clip action into the valid range.", {"clip_actions"}, config.clip_actions);
  args::ValueFlag torch_deterministic(parser, "torch_deterministic", "Whether to use deterministic cuda algorithms", {"torch_deterministic"}, config.torch_deterministic);
  args::ValueFlag exp_name_stem(parser, "exp_name_stem", "Name of the experiment.", {"exp_name_stem"}, config.exp_name_stem);
  args::ValueFlag env_id(parser, "env_id", "Name of the mujoco env to be executed.", {"env_id"}, config.env_id);
  args::ValueFlag num_envs(parser, "num_envs", "Number of environments to be used.", {"num_envs"}, config.num_envs);
  args::ValueFlagList<int> gpu_ids(parser, "gpu_ids", "The ids of the GPUs used for training. Len of this list determines the number of devices."
                                                               "Usage: --gpu_ids 0 --gpu_ids 1 --gpu_ids 2 ...", {"gpu_ids"}, config.gpu_ids);
  args::ValueFlag collect_device(parser, "collect_device", "Whether to collect data on gpu or cpu. Options: cpu, gpu", {"collect_device"}, config.collect_device);
  args::ValueFlag train_device(parser, "train_device", "Whether to train on gpu or cpu. Options: cpu, gpu", {"train_device"}, config.train_device);
  args::ValueFlag rdzv_addr(parser, "rdzv_addr", "IP adress of master node. Default: localhost", {"rdzv_addr"}, config.rdzv_addr);
  args::ValueFlag tcp_store_port(parser, "tcp_store_port", "Port for the TCP store. Default: 29500", {"tcp_store_port"}, config.tcp_store_port);
  args::ValueFlag use_dd_ppo_preempt(parser, "use_dd_ppo_preempt", "Flag to toggle the dd_ppo pre-emption trick", {"use_dd_ppo_preempt"}, config.use_dd_ppo_preempt);
  args::ValueFlag dd_ppo_min_perc(parser, "dd_ppo_min_perc", "Percentage of envs that need to finish before preemtion.", {"dd_ppo_min_perc"}, config.dd_ppo_min_perc);
  args::ValueFlag dd_ppo_preempt_threshold(parser, "dd_ppo_preempt_threshold", "Percentage of envs that need to finish before preemtion.", {"dd_ppo_preempt_threshold"}, config.dd_ppo_preempt_threshold);


  try
  {
    parser.ParseCLI(argc, argv);
  }
  catch (const args::Help&)
  {
    cout << parser;
    return 0;
  }
  catch (const args::ParseError& e)
  {
    cerr << e.what() << std::endl;
    cerr << parser;
    return 1;
  }

  // Update config with argparse arguments.
  // Unfortunately we have to do this manually for every variable since c++ doesn't support reflections.
  config.seed = args::get(seed);
  config.eval_seed = args::get(eval_seed);
  config.total_timesteps = args::get(total_timesteps);
  config.learning_rate = args::get(learning_rate);
  config.num_steps = args::get(num_steps);
  config.gamma = args::get(gamma);
  config.gae_lambda = args::get(gae_lambda);
  config.num_minibatches = args::get(num_minibatches);
  config.update_epochs = args::get(update_epochs);
  config.norm_adv = args::get(norm_adv);
  config.clip_coef = args::get(clip_coef);
  config.clip_vloss = args::get(clip_vloss);
  config.ent_coef = args::get(ent_coef);
  config.vf_coef = args::get(vf_coef);
  config.max_grad_norm = args::get(max_grad_norm);
  config.adam_eps = args::get(adam_eps);
  config.anneal_lr = args::get(anneal_lr);
  config.num_eval_runs = args::get(num_eval_runs);
  config.clip_actions = args::get(clip_actions);
  config.torch_deterministic = args::get(torch_deterministic);
  config.exp_name_stem = args::get(exp_name_stem);
  config.env_id = args::get(env_id);
  config.num_envs = args::get(num_envs);
  config.gpu_ids = args::get(gpu_ids);
  config.collect_device = args::get(collect_device);
  config.train_device = args::get(train_device);
  config.rdzv_addr = args::get(rdzv_addr);
  config.tcp_store_port = args::get(tcp_store_port);
  config.use_dd_ppo_preempt = args::get(use_dd_ppo_preempt);
  config.dd_ppo_min_perc = args::get(dd_ppo_min_perc);
  config.dd_ppo_preempt_threshold = args::get(dd_ppo_preempt_threshold);

  // Need to recompute them as the value might have changed
  config.num_devices = world_size;  // TODO think about how to do this with multi-node
  config.num_envs_per_device = config.num_envs / config.num_devices;
  assert((config.num_envs % config.num_devices == 0) && "num_envs must be a multiple of num_devices.");

  config.exp_name = (boost::format("%s_%d") % config.exp_name_stem % config.seed).str();
  config.batch_size = config.num_steps * config.num_envs;
  config.minibatch_size = config.batch_size / config.num_minibatches;
  config.num_iterations = config.total_timesteps / config.batch_size;
  config.batch_size_per_device = config.batch_size / config.num_devices;
  config.minibatch_per_device = config.minibatch_size / config.num_devices;

  auto local_rank = rank % config.num_devices;

  cout << "world_size: " << world_size << "\n";
  cout << "rank: " << rank << "\n";
  cout << "local_rank: " << local_rank << endl;

  filesystem::path exp_folder(directory + "/../models"s);
  exp_folder = exp_folder / config.exp_name;
  filesystem::create_directories(exp_folder);

  GOOGLE_PROTOBUF_VERIFY_VERSION;
  TensorBoardLogger logger(exp_folder.string() + (boost::format("/tfevents_logs_%d.pb") % rank).str());
  if (rank == 0) {
    logger.add_text("hyperparameters", 0, config.to_string().c_str());
  }

  // Seed libtorch
  manual_seed(config.seed);
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

  if(config.collect_device == "cpu"s) {
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

  if(config.train_device == "cpu"s) {
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

  vector<shared_ptr<SeqVectorEnv>> envs;

  if (config.env_id == "Humanoid-v4") {
    for (int i = 0; i < config.num_envs_per_device; ++i) {
      std::vector<shared_ptr<EnvironmentWrapper>> env_array;

      auto env_0 = make_shared<HumanoidV4Env>(directory + "/../libs/gymcpp/mujoco/assests/humanoid.xml", "rgb_array");
      env_array.push_back(make_env(env_0, config.gamma));
      envs.push_back(make_shared<SeqVectorEnv>(env_array, config.clip_actions));
    }
  }
  else if (config.env_id == "HalfCheetah-v5") {
    for (int i = 0; i < config.num_envs_per_device; ++i) {
      std::vector<shared_ptr<EnvironmentWrapper>> env_array;
      auto env_0 = make_shared<HalfCheetahV5Env>(directory + "/../libs/gymcpp/mujoco/assests/half_cheetah.xml", "rgb_array");
      env_array.push_back(make_env(env_0, config.gamma));
      envs.push_back(make_shared<SeqVectorEnv>(env_array, config.clip_actions));
    }
  }
  else
  {
    cerr << boost::format("env_id: %d is not implemented.") % config.env_id << endl;
    return 1;
  }

  auto agent = Agent(envs[0]->get_observation_space(), envs[0]->get_action_space());

  const filesystem::path model_path = exp_folder / "model_intial.pth"s;

  // Having the same initialization ensures that distributing the model across gpus and aggregating the gradient
  // behaves the same as a bigger batch size on a single device.
  // Broadcast initial model parameters from rank 0
  for (auto& p : agent->parameters()) {
    comm->broadcast(p, 0);
  }

  agent->to(collect_devices);
  auto optimizer = optim::Adam(agent->parameters(), optim::AdamOptions(config.learning_rate).eps(config.adam_eps));

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
  shared_ptr<TCPStoreServer> tcp_server = make_shared<TCPStoreServer>(config.rdzv_addr, config.tcp_store_port, config.num_envs);
  if (rank == 0 and config.use_dd_ppo_preempt) {
    // Server store used for all threads and processes
      tcp_server->start();
  }

  boost::asio::thread_pool pool(config.num_envs_per_device);
  vector<future<tuple<vector<env_info>, int>>> results;
  vector<shared_ptr<TCPStoreClient>> tcp_clients;
  for (int i = 0; i < config.num_envs_per_device; ++i) {
    results.emplace_back();
    if (config.use_dd_ppo_preempt) {
      tcp_clients.push_back(make_shared<TCPStoreClient>(config.rdzv_addr, config.tcp_store_port));
    }
  }

  // Storage setup
  Tensor obs        = torch::zeros({config.num_steps, config.num_envs_per_device, envs[0]->get_observation_space()}, collect_devices);
  Tensor actions    = torch::zeros({config.num_steps, config.num_envs_per_device, envs[0]->get_action_space()}, collect_devices);
  Tensor logprobs   = torch::zeros({config.num_steps, config.num_envs_per_device}, collect_devices);
  Tensor rewards    = torch::zeros({config.num_steps, config.num_envs_per_device}, collect_devices);
  Tensor dones      = torch::zeros({config.num_steps, config.num_envs_per_device}, collect_devices);
  Tensor values     = torch::zeros({config.num_steps, config.num_envs_per_device}, collect_devices);
  Tensor returns    = torch::zeros({config.num_steps, config.num_envs_per_device}, collect_devices);
  Tensor advantages = torch::zeros({config.num_steps, config.num_envs_per_device}, collect_devices);

  Tensor next_obs  = torch::zeros({config.num_envs_per_device, envs[0]->get_observation_space()}, collect_devices);
  Tensor next_done = torch::zeros({config.num_envs_per_device}, collect_devices);

  // When running the agent in parallel we need to have each agent use its own seed generator, to ensure exact reproducibility.
  vector<Generator> agent_generators;
  Generator train_generator;
  vector<at::cuda::CUDAStream> cuda_streams;
  for (int i = 0; i < config.num_envs_per_device; ++i) {
    next_obs.index({i}) = envs[i]->reset(config.seed+i).squeeze(0).to(collect_devices);

    if (config.collect_device == "gpu"s) {
      agent_generators.push_back(at::make_generator<CUDAGeneratorImpl>(config.seed * (100 * rank) + i));
    }
    else {
      agent_generators.push_back(at::make_generator<at::CPUGeneratorImpl>(config.seed * (100 * rank) + i));
    }
    cuda_streams.push_back(at::cuda::getStreamFromPool(true, collect_devices.index()));
  }

  // Train indicies are typically on CPU.
  train_generator = at::make_generator<at::CPUGeneratorImpl>(config.seed * 1500 + rank);

  std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
  TicToc tt;
  long global_step = 0;
  float sychronize = 0.0f;
  for (auto iteration : tq::trange(config.num_iterations, (rank == 0))) {
    tt.tic();
    agent->eval();
    agent->to(collect_devices);

    if (rank == 0 and config.use_dd_ppo_preempt) {
        tcp_clients[0]->reset();
    }
    comm->allreduce(sychronize, false);

    if (config.anneal_lr) {
      float frac = 1.0f - static_cast<float>(iteration) / static_cast<float>(config.num_iterations);
      float lrnow = frac * config.learning_rate;
      auto &options = static_cast<optim::OptimizerOptions &>(optimizer.param_groups()[0].options());
      options.set_lr(lrnow);
    }

    for (int i = 0; i < config.num_envs_per_device; ++i) {
      // We need to pass indices by value, so the parallel threads get the correct index.
      auto collect = [&, i] {
        vector<env_info> infos_vec;
        // We need to run the different threads on different cuda streams so that the GPU computation runs concurrently.
        at::cuda::CUDAStreamGuard guard(cuda_streams[i]);

        int step;
        for (step = 0; step < config.num_steps; ++step) { // No grad
          NoGradGuard no_grad;
          obs.index({step, i}) = next_obs.index({i});
          dones.index({step, i}) = next_done.index({i});

          Tensor action, logprob, entropy, value, done;
          tie(action, logprob, entropy, value) = agent->get_action_and_value(next_obs.index({i}).unsqueeze(0).to(collect_devices), at::empty({0}), agent_generators[i]);

          values.index({step}).slice(0, i,i+1) = value.squeeze(0);
          actions.index({step, i}) = action.squeeze(0);
          logprobs.index({step, i}) = logprob.index({0});

          auto [env_next_obs, reward, termination, truncation, infos] = envs[i]->step(action.to(kCPU));

          done = torch::logical_or(termination, truncation).to(kFloat32);

          rewards.index({step, i}) = reward.index({0}).to(collect_devices);
          next_obs.index({i}) = env_next_obs.index({0}).to(collect_devices);
          next_done.index({i}) = done.index({0}).to(collect_devices);

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
              // cout << "Rank: " << rank << ", Thread: " << i << ", Preempt at step: " << step << " Num done: " << num_done << '\n';
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

    Tensor total_returns = torch::zeros({world_size}, TensorOptions().dtype(kFloat32).device(kCPU).requires_grad(false));
    Tensor total_lengths = torch::zeros({world_size}, TensorOptions().dtype(kFloat32).device(kCPU).requires_grad(false));
    Tensor num_total_returns = torch::zeros({world_size}, TensorOptions().dtype(kFloat32).device(kCPU).requires_grad(false));
    int min_collected_steps = static_cast<int>(config.num_steps);
    // Collect returns for logging. Synchronizes threads.
    for (int i = 0; i < config.num_envs_per_device; ++i) {
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
    global_step += config.num_envs * config.num_steps;

    // Log avg returns observed during data collection.
    if (rank == 0) {
      int num_total_returns_all_processes = torch::sum(num_total_returns).item<float>();
      if (num_total_returns_all_processes > 0.0f) {
        float total_returns_all_processes = torch::sum(total_returns).item<float>();
        float total_lengths_all_processes = torch::sum(total_lengths).item<float>();
        float avg_return = total_returns_all_processes / num_total_returns_all_processes;
        float avg_length = total_lengths_all_processes / num_total_returns_all_processes;

        logger.add_scalar("charts/episodic_return", static_cast<int>(global_step), avg_return);
        logger.add_scalar("charts/episodic_length", static_cast<int>(global_step), avg_length);

        cout << boost::format("global_step=%d, avg_episodic_return=%.2f \n") % global_step % avg_return;

        // Also log performance per wall clock time.
        chrono::high_resolution_clock::time_point current_time = chrono::high_resolution_clock::now();
        const long passed_seconds = lround(std::chrono::duration<float>(current_time - start_time).count());
        logger.add_scalar("charts/episodic_return_per_sec", static_cast<int>(passed_seconds), avg_return);
      }
    }


    tt.toc((boost::format("Rank: %d Time to collect data:") % rank).str());

    tt.tic();

    // Computes Generalized advantage estimation labels
    { // No grad
      NoGradGuard noGrad;
      const Tensor next_value = agent->get_value(next_obs).flatten();
      Tensor lastgaelam = torch::zeros({config.num_envs_per_device}, collect_devices);
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
        advantages.index({t}) = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam;
        lastgaelam = advantages.index({t});
      }
      returns = advantages + values;
    }

    tt.toc((boost::format("Rank: %d Time to compute advantage:") % rank).str());

    tt.tic();
    Tensor b_obs = obs.index({Slice(None, num_train_steps_per_env)}).reshape({-1, envs[0]->get_observation_space()}).to(train_devices);
    Tensor b_logprobs  = logprobs.index({Slice(None, num_train_steps_per_env)}).reshape({-1}).to(train_devices);
    Tensor b_actions = actions.index({Slice(None, num_train_steps_per_env)}).reshape({-1, envs[0]->get_action_space()}).to(train_devices);
    Tensor b_advantages = advantages.index({Slice(None, num_train_steps_per_env)}).reshape(-1).to(train_devices);
    Tensor b_returns = returns.index({Slice(None, num_train_steps_per_env)}).reshape(-1).to(train_devices);
    Tensor b_values = values.index({Slice(None, num_train_steps_per_env)}).reshape(-1).to(train_devices);
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
        auto [action, newlogprob, entropy, newvalue] = agent->get_action_and_value(b_obs.index({mb_inds}), b_actions.index({mb_inds}));
        Tensor logratio = newlogprob - b_logprobs.index({mb_inds});
        Tensor ratio = logratio.exp();

        {  // No grad
          NoGradGuard no_grad;
          // calculate approx_kl http://joschu.net/blog/kl-approx.html
          old_approx_kl = (-logratio).mean();
          approx_kl = ((ratio - 1.0f) - logratio).mean();
          clipfracs.push_back(((ratio - 1.0f).abs() > config.clip_coef).to(kFloat).mean().item<float>());
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
          grads.push_back(p.grad());
        }
        comm->allreduce(grads, true);

        nn::utils::clip_grad_norm_(agent->parameters(), config.max_grad_norm);
        optimizer.step();
      }
    }

    tt.toc((boost::format("Rank: %d Time to train") % rank).str());

    // Average tensors across processes for logging.
    float avg_clipfrac = reduce(clipfracs.begin(), clipfracs.end()) / static_cast<float>(clipfracs.size());
    comm->allreduce(v_loss, true);
    comm->allreduce(pg_loss, true);
    comm->allreduce(entropy_loss, true);
    comm->allreduce(old_approx_kl, true);
    comm->allreduce(approx_kl, true);
    comm->allreduce(avg_clipfrac, true);

    // Log agent and tensorboard values.
    if (rank == 0) {
      tt.tic();

      save_state(agent, optimizer, exp_folder,
        (boost::format("model_latest_%09d.pth") % iteration).str(),
        (boost::format("optimizer_latest_%09d.pth") % iteration).str());

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
        sps = static_cast<float>(global_step) / passed_seconds;
        cout  << std::fixed << std::setprecision(0) << "SPS: " << sps << endl;
      }

      auto &options = static_cast<optim::OptimizerOptions &>(optimizer.param_groups()[0].options());
      logger.add_scalar("charts/learning_rate", static_cast<int>(global_step), options.get_lr());
      logger.add_scalar("losses/value_loss", static_cast<int>(global_step), v_loss.item<float>());
      logger.add_scalar("losses/policy_loss", static_cast<int>(global_step), pg_loss.item<float>());
      logger.add_scalar("losses/entropy", static_cast<int>(global_step), entropy_loss.item<float>());
      logger.add_scalar("losses/old_approx_kl", static_cast<int>(global_step), old_approx_kl.item<float>());
      logger.add_scalar("losses/approx_kl", static_cast<int>(global_step), approx_kl.item<float>());
      logger.add_scalar("losses/clipfrac", static_cast<int>(global_step), avg_clipfrac);
      logger.add_scalar("charts/SPS", static_cast<int>(global_step), sps);

      tt.toc((boost::format("Rank: %d Time to log") % rank).str());
    }
    cout << std::flush;
  }
  if (rank == 0) {
    save_state(agent, optimizer, exp_folder, "model_final.pth"s, "optimizer_final.pth"s);
  }

  if (rank == 0)
  { // no grad
    NoGradGuard no_grad;
    // Evaluate final model

    agent->eval();
    agent->to(train_devices);
    // We are using the training envs to evaluate the model because the normalization wrappers have statistics
    // that get lost if you make a new env. If you want to actually evaluate a saved model with a new env you would need
    // to save an load those statistics as well.
    auto next_obs_eval = envs.at(0)->reset(config.eval_seed);
    vector<float> episodic_returns;
    while (episodic_returns.size() < config.num_eval_runs) {
      Tensor action;
      {
        Tensor entropy, done, logprob, value; // Don't need this variable. Will be deleted once left scope.
        tie(action, logprob, entropy, value) = agent->get_action_and_value(next_obs_eval.to(train_devices));
      }
      auto [env_next_obs, reward, termination, truncation, infos] = envs[0]->step(action.to(kCPU));

      next_obs_eval = env_next_obs;
      for (const auto info : infos)
      {
        if (info.has_value()) {
          auto [r, l, t] = info.value();
          cout << (boost::format("Evaluation result: episode=%d episodic_return=%.2f \n") % episodic_returns.size() % r).str();
          episodic_returns.push_back(r);
        }
      }
    }
    for (int i = 0; i < episodic_returns.size(); ++i) {
      logger.add_scalar("eval/episodic_return", i, episodic_returns.at(i));
    }
    const float avg_return = reduce(episodic_returns.begin(), episodic_returns.end()) / static_cast<float>(episodic_returns.size());
    logger.add_scalar("eval/avg_return", static_cast<int>(episodic_returns.size()), avg_return);
    cout << (boost::format("Average evaluation return=%.2f over %d episodes") % avg_return % episodic_returns.size()).str() << endl;
  }
  comm->allreduce(sychronize, false);
  pool.join();
  google::protobuf::ShutdownProtobufLibrary();
  comm->finalize();
  MPI_Finalize();
}
