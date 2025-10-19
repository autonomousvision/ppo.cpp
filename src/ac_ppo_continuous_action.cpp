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
#include <gymcpp/mujoco/hopper_v5.h>
#include <gymcpp/mujoco/ant_v5.h>
#include <gymcpp/wrappers/common.h>
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
#include <GLFW/glfw3.h>

using namespace std;
using namespace torch;
using namespace torch::indexing;

shared_ptr<EnvironmentWrapper> make_env(const shared_ptr<Environment>& env_0) {
  auto env_1 = make_shared<RecordEpisodeStatistics>(env_0);
  return env_1;
}

class GlobalConfig {
public:
  // Using Atari hyperparameters as default.
  int seed = 1;  // negative values mean no seeding will take place.
  int eval_seed = 2;  // negative values mean no seeding will take place.
  unsigned int total_timesteps = 10'000'000;
  float learning_rate = 2.5e-4;
  unsigned int num_envs = 8;
  unsigned int num_steps = 128;
  float gamma = 0.99;
  float gae_lambda = 0.95;
  unsigned int num_minibatches = 4;
  unsigned int update_epochs = 4;
  bool norm_adv = true;
  float clip_coef = 0.1;
  bool clip_vloss = true;
  float ent_coef = 0.01;
  float vf_coef = 0.5;
  float max_grad_norm = 0.5;
  float adam_eps = 1e-5;
  bool anneal_lr = true;
  unsigned int num_eval_runs = 128;
  bool clip_actions = true;
  bool torch_deterministic = true;
  string exp_name_stem = "Ant-v5_AC_PPO_Atari"s;
  string env_id = "Ant-v5"s;  // Options: Humanoid-v4, Ant-v5, HalfCheetah-v5
  string render = "rgb_array"s;  // Set to human for Visualizing the training with OpenGL, rgb_array for no visualization
  vector<int> gpu_ids = vector<int>({0});
  string collect_device = "cpu"s;  // Options: cpu, gpu
  string train_device = "cpu"s;  // Options: cpu, gpu
  string rdzv_addr = "localhost"s;  // IP adress of master node.
  uint16_t tcp_store_port = 29500u;  // Port for the TCP store.
  int use_dd_ppo_preempt = 0;  // Flag to toggle the dd_ppo pre-emption trick.
  float dd_ppo_min_perc = 0.25;  // Minimum percentage of env steps that need to finish before preemtion.
  float dd_ppo_preempt_threshold = 0.6;  // Minimum number of envs that need to finish before preemtion.
  bool estimate_mean_std = false;  // Estimates mean and std of the run for env 0 and prints at the end.

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
                  "|seed|%d|\n"
                  "|eval_seed|%d|\n"
                  "|total_timesteps|%d|\n"
                  "|learning_rate|%f|\n"
                  "|num_envs|%d|\n"
                  "|num_steps|%d|\n"
                  "|gamma|%f|\n"
                  "|gae_lambda|%f|\n"
                  "|num_minibatches|%u|\n"
                  "|update_epochs|%u|\n"
                  "|norm_adv|%d|\n"
                  "|clip_coef|%f|\n"
                  "|clip_vloss|%d|\n"
                  "|ent_coef|%f|\n"
                  "|vf_coef|%f|\n"
                  "|max_grad_norm|%f|\n"
                  "|adam_eps|%f|\n"
                  "|anneal_lr|%d|\n"
                  "|num_eval_runs|%u|\n"
                  "|clip_actions|%d|\n"
                  "|exp_name|%s|\n"
                  "|batch_size|%u|\n"
                  "|minibatch_size|%u|\n"
                  "|num_iterations|%u|\n"
                  "|torch_deterministic|%d|\n"
                  "|exp_name_stem|%s|\n"
                  "|env_id|%s|\n"
                  "|collect_device|%s|\n"
                  "|train_device|%s|\n"
                  "|num_envs_per_device|%u|\n"
                  "|batch_size_per_device|%u|\n"
                  "|use_dd_ppo_preempt|%d|\n"
                  "|dd_ppo_min_perc|%f|\n"
                  "|dd_ppo_preempt_threshold|%f|\n"
                  "|minibatch_per_device|%u|\n")
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
  explicit AgentImpl(int observation_space, int action_space, const float action_high, const float action_low, Tensor mean, Tensor std)
  {
    action_space_high = register_parameter("action_space_high"s, torch::tensor(action_high), false);
    action_space_low = register_parameter("action_space_low"s, torch::tensor(action_low), false);
    mean_ = register_parameter("mean_"s, mean.unsqueeze(0), false);
    std_ = register_parameter("std_"s, std.unsqueeze(0), false);

    critic = nn::Sequential(nn::Linear(observation_space, 256),
                            nn::LayerNorm(nn::LayerNormOptions({256})),
                            nn::ReLU(),
                            nn::Linear(256, 256),
                            nn::LayerNorm(nn::LayerNormOptions({256})),
                            nn::ReLU(),
                            nn::Linear(256, 1));
    register_module("critic", critic);

    actor_encoder = nn::Sequential(
        nn::Linear(observation_space, 256),
        nn::LayerNorm(nn::LayerNormOptions({256})),
        nn::ReLU(),
        nn::Linear(256, 256),
        nn::LayerNorm(nn::LayerNormOptions({256})),
        nn::ReLU()
    );
    register_module("actor_mean", actor_encoder);

    dist_alpha = nn::Sequential(
      nn::Linear(256, action_space)
    );
    register_module("dist_alpha", dist_alpha);

    dist_beta = nn::Sequential(
      nn::Linear(256, action_space)
    );
    register_module("dist_beta", dist_beta);
  }
  Tensor get_value(const Tensor& x) {
    Tensor normalized_x = (x - mean_) / std_;
    normalized_x = critic->forward(normalized_x);
    return normalized_x;
  }

  Tensor scale_action(const Tensor& action) const {
    // Values are specific to beta distribution
    constexpr float d_low = 0.0f;
    constexpr float d_high = 1.0f;
    constexpr float eps = 1e-7;

    Tensor scaled_action = (action - action_space_low) / (action_space_high - action_space_low) * (d_high - d_low) + d_low;
    scaled_action = torch::clamp(scaled_action, d_low + eps, d_high + eps);
    return scaled_action;
  }

  Tensor unscale_action(const Tensor& action) const {
    // Values are specific to beta distribution
    constexpr float d_low = 0.0f;
    constexpr float d_high = 1.0f;
    return (action - d_low) / (d_high - d_low) * (action_space_high - action_space_low) + action_space_low;
  }

  tuple<Tensor, Tensor, Tensor, Tensor> get_action_and_value(const Tensor& x, Tensor action=at::empty({0}),
    const string sample_type="sample"s, const optional<Generator>& generator=nullopt)
  {
    Tensor normalized_x = (x - mean_) / std_;
    Tensor actor_features = actor_encoder->forward(normalized_x);
    Tensor alpha = dist_alpha->forward(actor_features);
    Tensor beta = dist_beta->forward(actor_features);

    alpha = nn::functional::softplus(alpha) + 1.0f;
    beta = nn::functional::softplus(beta) + 1.0f;

    const Beta probs(alpha, beta);

    if (action.data_ptr() == nullptr) {
        if (sample_type == "sample"s) {
          action = probs.sample(generator);
        }
        else if (sample_type == "mean"s) {
          action = probs.mean();
        }
        else if (sample_type == "roach"s) {
          action = probs.roach_deterministic();
        }
        else {
          throw runtime_error("Unsupported sample type used. Sample type: "s + sample_type);
        }
      }
    else {
      action = scale_action(action);
    }

    Tensor logprob = probs.log_prob(action).sum(1);
    action = unscale_action(action);

    Tensor entropy = probs.entropy().sum(1);
    Tensor value = critic->forward(normalized_x);
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
  nn::Sequential actor_encoder{nullptr};
  nn::Sequential dist_alpha{nullptr};
  nn::Sequential dist_beta{nullptr};
  Tensor action_space_high;
  Tensor action_space_low;
  Tensor mean_;
  Tensor std_;
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
// For multi-device training start the program with: mpirun -n 2 --bind-to none ac_ppo_carla
int main(const int argc, const char** argv) {
  filesystem::path exe = filesystem::canonical(argv[0]);
  filesystem::path basedir = exe.parent_path();
  std::string directory = basedir.string();
  cout << "Location of program: " << directory << endl;
  ios_base::sync_with_stdio(false);  // Faster print
  // Can be slightly faster to turn off multi-threading in libtorch. Might depend on model size.
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
#ifdef _WIN32
  _putenv_s("OMP_NUM_THREADS", "1");
  _putenv_s("MKL_NUM_THREADS", "1");
#else
  setenv("OMP_NUM_THREADS", "1", 1);
  setenv("MKL_NUM_THREADS", "1", 1);
#endif

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
  args::ValueFlag render(parser, "render", "Set to human for Visualizing the training with OpenGL, rgb_array for no visualization", {"render"}, config.render);
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
  args::ValueFlag estimate_mean_std(parser, "estimate_mean_std", "Percentage of envs that need to finish before preemtion.", {"estimate_mean_std"}, config.estimate_mean_std);


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
  config.render = args::get(render);
  config.num_envs = args::get(num_envs);
  config.gpu_ids = args::get(gpu_ids);
  config.collect_device = args::get(collect_device);
  config.train_device = args::get(train_device);
  config.rdzv_addr = args::get(rdzv_addr);
  config.tcp_store_port = args::get(tcp_store_port);
  config.use_dd_ppo_preempt = args::get(use_dd_ppo_preempt);
  config.dd_ppo_min_perc = args::get(dd_ppo_min_perc);
  config.dd_ppo_preempt_threshold = args::get(dd_ppo_preempt_threshold);
  config.estimate_mean_std = args::get(estimate_mean_std);

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

  filesystem::path exp_folder(basedir / ".." / "models");
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

  Tensor observation_mean;
  Tensor observation_std;

  if (config.env_id == "Humanoid-v4") {
    filesystem::path mujoco_xml = basedir / "mujoco" / "assets" / "humanoid.xml";
    cout << "Loading file: " << mujoco_xml.string() << endl;
    for (int i = 0; i < config.num_envs_per_device; ++i) {
      std::vector<shared_ptr<EnvironmentWrapper>> env_array;

      auto env_0 = make_shared<HumanoidV4Env>(mujoco_xml.string(), config.render);
      env_array.push_back(make_env(env_0));
      envs.push_back(make_shared<SeqVectorEnv>(env_array, config.clip_actions));
    }
    observation_mean = torch::tensor({1.2067, 0.9201, -0.1168, -0.1367, 0.0888, -0.1257, 0.3312, 0.3262, 0.0628, 0.3241, 0.2047, -0.0534, 0.0014, 0.3436, -0.2829, -0.0991, 0.6670, -0.6024, -0.7289, -0.3202, 0.7680, -0.4821, 0.3625, 0.0331, -0.2775, -0.1132, -0.2253, -0.0627, -0.0480, 0.2474, 0.1203, 0.0065, 0.1339, 0.0812, -0.1305, -0.0386, 0.2401, -0.1599, -0.1570, 0.4250, -0.3941, -0.7164, -0.2651, 0.5239, -0.5930, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 2.0507, 2.0406, 0.2368, 0.0292, 0.1744, -0.1379, -0.3409, 0.2869, 4.0142, 8.9075, 0.1085, 0.1027, 0.0203, 0.0014, -0.0071, 0.0113, 0.0361, -0.0522, 0.4558, 2.2619, 0.0841, 0.0564, 0.0785, 0.0005, 0.0013, 0.0064, -0.0332, -0.1863, 0.3131, 6.6162, 0.2483, 0.2109, 0.0874, -0.0054, -0.0251, -0.0668, -0.0873, -0.4644, -0.7548, 4.7518, 0.7999, 0.8191, 0.1109, -0.0123, -0.1377, -0.0971, -0.2662, -0.1974, -1.4102, 2.7557, 0.9059, 0.9410, 0.1227, -0.0090, -0.1667, -0.0735, -0.2495, -0.1086, -1.2281, 1.7671, 0.1803, 0.1852, 0.0773, -0.0096, 0.0345, 0.0400, 0.2252, 0.2988, -0.6201, 4.7518, 0.7009, 0.7702, 0.1624, -0.0119, 0.1354, 0.0968, 0.3330, 0.1942, -1.2986, 2.7557, 0.8167, 0.8956, 0.1777, -0.0078, 0.1468, 0.0965, 0.2681, 0.1417, -1.1489, 1.7671, 0.3601, 0.3185, 0.0839, 0.0048, -0.0022, 0.1013, 0.0089, -0.2631, 0.6819, 1.6611, 0.1679, 0.1529, 0.1146, 0.0274, -0.0451, 0.0539, 0.1662, -0.2379, 0.2808, 1.2295, 0.2650, 0.2283, 0.1157, 0.0157, 0.0265, -0.0974, -0.1027, 0.3202, 0.5252, 1.6611, 0.1486, 0.1315, 0.1269, -0.0074, -0.0305, -0.0405, 0.0424, 0.2681, 0.1897, 1.2295, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.2181, -0.1105, -0.1053, 0.4110, -0.0397, -0.2029, -0.1152, 0.0918, -0.1766, 0.3631, -0.0147, -0.2006, 0.0214, 0.0575, -0.1637, 0.3679, 0.0034, -0.1822, -0.0591, 0.1322, 0.0098, 0.3472, 0.0138, -0.1843, -0.1156, 0.2339, 0.0411, 0.3836, 0.0381, -0.1935, -0.1156, 0.2339, 0.0411, 0.3836, 0.0381, -0.1935, 0.0466, -0.0779, -0.4843, 0.3511, -0.0037, -0.1815, 0.0758, 0.0612, -0.4836, 0.4064, -0.0132, -0.1862, 0.0758, 0.0612, -0.4836, 0.4064, -0.0132, -0.1862, 0.2557, 0.5031, -0.2566, 0.0992, 0.1956, -0.2359, -0.0884, 0.9220, -0.5763, 0.0649, 0.0985, -0.3221, -0.4464, 0.4929, -0.1611, 0.1485, -0.1382, -0.2348, -0.5712, 0.9898, -0.1302, 0.0953, -0.1208, -0.3205, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -1.7708, 8.0737, 1.9869, 6.1523, 13.2124, 28.5278, 37.2543, 3.3135, 6.6229, 4.4404, 26.5631, -0.0497, -0.4320, -0.0597, -0.1960, 0.0591, 0.7946, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0015, -0.0808, 0.0028, -0.1954, 0.0011, -0.1048, -0.0994, -0.1408, -0.0357, -0.6195, 0.5659, -0.7131, 0.0025, -0.0035, -0.0054, -0.0974, -0.0165, -0.1257, -0.0998, 0.0306, -0.0076, -0.1546, -0.3981, -0.0144, -0.0059, 0.0008, -0.0002, -0.0016, -0.0133, -0.0014, -3.7099, 13.0694, -0.4196, 18.7501, 6.8694, 239.8089, 0.1006, -0.0288, 0.0039, 0.1301, 0.3542, -0.0258, 0.0092, 0.0020, -0.0001, -0.0018, 0.0168, 0.0080, 4.1156, -11.8467, -0.6581, -6.3820, -6.1693, 154.1199, 0.0032, 0.0185, 0.0066, 0.0721, -0.0660, 0.1900, 0.1546, 0.1605, 0.0459, 0.6995, -0.8975, 0.5440, -0.0002, 0.0063, -0.0032, 0.0268, 0.0231, 0.0600, -0.0599, 0.0381, -0.0072, 0.1407, 0.4348, 0.1967}, dtype(kFloat).requires_grad(false));
    observation_std = torch::tensor({0.0838, 0.0786, 0.1742, 0.1637, 0.2235, 0.2485, 0.3006, 0.3639, 0.0661, 0.2727, 0.2045, 0.1338, 0.1081, 0.2391, 0.4400, 0.2078, 0.4106, 0.4370, 0.7245, 0.3611, 0.5052, 0.7217, 0.5590, 0.5931, 0.4081, 1.2603, 1.5436, 2.0705, 2.5298, 2.3992, 1.7899, 1.2342, 2.8975, 2.6135, 3.0877, 2.1655, 3.2730, 3.6635, 4.9855, 2.0595, 2.4454, 2.8886, 2.0891, 2.7278, 2.8454, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.2180, 0.2182, 0.2212, 0.0987, 0.3376, 0.3503, 0.7600, 0.8071, 0.3076, 0.0000, 0.0178, 0.0186, 0.0139, 0.0063, 0.0190, 0.0220, 0.0959, 0.1110, 0.0441, 0.0000, 0.0280, 0.0254, 0.0266, 0.0167, 0.0144, 0.0177, 0.2501, 0.2898, 0.1319, 0.0000, 0.0431, 0.0412, 0.0456, 0.0254, 0.0405, 0.0328, 0.2523, 0.2119, 0.1462, 0.0000, 0.0985, 0.1024, 0.1160, 0.0542, 0.1478, 0.1315, 0.3092, 0.2812, 0.1266, 0.0000, 0.1108, 0.1067, 0.1332, 0.0617, 0.1697, 0.1537, 0.2713, 0.2494, 0.1080, 0.0000, 0.0577, 0.0339, 0.0491, 0.0241, 0.0412, 0.0339, 0.2796, 0.2201, 0.1582, 0.0000, 0.1759, 0.0934, 0.1608, 0.0641, 0.1777, 0.1296, 0.4397, 0.2862, 0.2151, 0.0000, 0.1931, 0.1023, 0.1839, 0.0715, 0.2108, 0.1514, 0.3890, 0.2525, 0.1804, 0.0000, 0.0765, 0.0713, 0.0474, 0.0294, 0.0640, 0.0649, 0.1749, 0.1646, 0.1178, 0.0000, 0.1063, 0.1146, 0.0549, 0.0373, 0.0596, 0.0503, 0.1657, 0.1387, 0.2017, 0.0000, 0.0884, 0.0760, 0.0486, 0.0412, 0.0614, 0.0495, 0.2229, 0.1427, 0.1390, 0.0000, 0.0963, 0.1078, 0.0595, 0.0527, 0.0601, 0.0524, 0.2439, 0.1162, 0.2245, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.3468, 1.6557, 1.9246, 0.6424, 0.6251, 0.3845, 1.3868, 1.8024, 2.1527, 0.4451, 0.5707, 0.3692, 1.3040, 1.7927, 2.1730, 0.4268, 0.4694, 0.3656, 1.3557, 1.6341, 3.2462, 0.5427, 0.4880, 0.3626, 1.4846, 1.8916, 3.2423, 1.0402, 0.7554, 0.4889, 1.4846, 1.8916, 3.2423, 1.0402, 0.7554, 0.4889, 1.8360, 2.4584, 3.4402, 0.4857, 0.4831, 0.3985, 1.9953, 2.9820, 3.5171, 1.4613, 0.9335, 0.8695, 1.9953, 2.9820, 3.5171, 1.4613, 0.9335, 0.8695, 1.8673, 2.3263, 1.9583, 1.2115, 0.9559, 0.4748, 2.0948, 2.7284, 2.1195, 1.2116, 0.9518, 0.5800, 1.8253, 2.2696, 2.3032, 1.0659, 0.8993, 0.5105, 2.0049, 2.9582, 2.4394, 1.0604, 0.9027, 0.6122, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.6812, 0.1127, 1.3318, 1.4458, 9.2671, 3.6582, 1.1704, 1.0806, 0.5137, 0.7694, 6.0397, 5.4231, 5.4939, 5.3830, 5.4752, 5.3715, 5.4023, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.1086, 1.3705, 0.3843, 3.2683, 2.8190, 2.3140, 1.6833, 1.4983, 0.6813, 6.6424, 8.5245, 7.0875, 0.2368, 0.2138, 0.3039, 3.1042, 3.0361, 3.0391, 3.1869, 1.7751, 0.9081, 7.0531, 13.2572, 4.6236, 0.7359, 0.3876, 0.1618, 0.9006, 1.7000, 0.6539, 44.9869, 57.8573, 12.3572, 76.7473, 60.8978, 279.5875, 3.1967, 1.7859, 0.9243,7.2009, 13.4145, 4.7449, 1.0385, 0.6452, 0.2391, 1.0946, 1.8960, 1.4966, 35.4017, 51.5872, 11.9319, 76.3289, 59.8354, 299.4579, 0.3642, 0.5474, 0.2621, 2.3356, 2.2154, 4.2920, 1.6720, 1.7124, 0.6763, 6.7303, 7.9297, 5.3757, 0.2708, 0.3514, 0.1830, 1.4793, 1.4129, 2.3110, 1.0881, 0.9243, 0.4474, 4.0738, 5.1913, 3.3184}, dtype(kFloat).requires_grad(false));
  }
  else if (config.env_id == "HalfCheetah-v5") {
    filesystem::path mujoco_xml = basedir / "mujoco" / "assets" / "half_cheetah.xml";
    cout << "Loading file: " << mujoco_xml.string() << endl;
    for (int i = 0; i < config.num_envs_per_device; ++i) {
      std::vector<shared_ptr<EnvironmentWrapper>> env_array;
      auto env_0 = make_shared<HalfCheetahV5Env>(mujoco_xml.string(), config.render);
      env_array.push_back(make_env(env_0));
      envs.push_back(make_shared<SeqVectorEnv>(env_array, config.clip_actions));
    }
    // TODO estimate
    observation_mean = torch::zeros({envs[0]->get_observation_space()}, dtype(kFloat).requires_grad(false));
    observation_std = torch::ones({envs[0]->get_observation_space()}, dtype(kFloat).requires_grad(false));
  }
  else if (config.env_id == "Ant-v5") {
    filesystem::path mujoco_xml = basedir / "mujoco" / "assets" / "ant.xml";
    cout << "Loading file: " << mujoco_xml.string() << endl;
    for (int i = 0; i < config.num_envs_per_device; ++i) {
      std::vector<shared_ptr<EnvironmentWrapper>> env_array;
      auto env_0 = make_shared<AntV5Env>(mujoco_xml.string(), config.render);
      env_array.push_back(make_env(env_0));
      envs.push_back(make_shared<SeqVectorEnv>(env_array, config.clip_actions));
    }
    observation_std = torch::tensor({0.1457, 0.3284, 0.2356, 0.2533, 0.2465, 0.4133, 0.2276, 0.3176, 0.1543, 0.4031, 0.1743, 0.3922, 0.2557, 1.7925, 0.8815, 1.0024, 1.1423, 1.2915, 1.1746, 5.1470, 2.7205, 2.5965, 1.3809, 5.1394, 1.5694, 3.1544, 3.2665, 0.1580, 0.1582, 0.0336, 0.2323, 0.2323, 0.2680, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.0473, 0.0478, 0.0466, 0.0521, 0.0522, 0.0564, 0.2816, 0.2806, 0.2776, 0.2871, 0.2872, 0.2835, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.0495, 0.0491, 0.0474, 0.0540, 0.0540, 0.0586, 0.2308, 0.2317, 0.2256, 0.2332, 0.2323, 0.2355, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.0478, 0.0479, 0.0465, 0.0523, 0.0522, 0.0568, 0.2985, 0.2952, 0.2994, 0.3057, 0.3040, 0.2995, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.0475, 0.0475, 0.0463, 0.0521, 0.0521, 0.0568, 0.1716, 0.1743, 0.1647, 0.1770, 0.1756, 0.1800}, dtype(kFloat).requires_grad(false));
    observation_mean = torch::tensor({ 0.5450, 0.8056, 0.0048, -0.0222, 0.2493, -0.0541, 0.7024, -0.3042, -0.5781, 0.0793, -0.5972, 0.0423, 0.7128, 2.6974, -0.0774, 0.0041, 0.0080, -0.0116, 0.0058, -0.0149, 0.0318, 0.0159, -0.0370, 0.0117, -0.0387, 0.0010, 0.0324, -0.0006, 0.0005, -0.0001, 0.0003, -0.0002, 0.0779, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0003, -0.0000, 0.0001, -0.0000, -0.0001, 0.0033, 0.0778, -0.0441, -0.0437, 0.0337, 0.0069, 0.0885, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, -0.0002, 0.0001, -0.0001, 0.0000, 0.0035, 0.0318, 0.0467, -0.0031, -0.0335, 0.0115, 0.0594, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0003, -0.0001, -0.0001, -0.0000, 0.0001, 0.0033, -0.0889, 0.0038, 0.0318, 0.0173, -0.0123, 0.1002, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0000, 0.0001, 0.0001, 0.0033, -0.0066, -0.0232, -0.0003, -0.0066, 0.0009, 0.0337}, dtype(kFloat).requires_grad(false));
  }
  else if (config.env_id == "Hopper-v5") {
    filesystem::path mujoco_xml = basedir / "mujoco" / "assets" / "hopper.xml";
    cout << "Loading file: " << mujoco_xml.string() << endl;
    for (int i = 0; i < config.num_envs_per_device; ++i) {
      std::vector<shared_ptr<EnvironmentWrapper>> env_array;
      auto env_0 = make_shared<HopperV5Env>(mujoco_xml.string(), config.render);
      env_array.push_back(make_env(env_0));
      envs.push_back(make_shared<SeqVectorEnv>(env_array, config.clip_actions));
    }
    observation_mean = torch::tensor({1.2739, 0.0315, -0.4293, -0.1627, 0.1548, 2.3837, -0.1428, 0.0132, -0.3067, -0.1605, 0.0042}, dtype(kFloat).requires_grad(false));
    observation_std = torch::tensor({0.1744, 0.0657, 0.2552, 0.2478, 0.5991, 0.9265, 1.3565, 0.8800, 1.9136, 2.6953, 5.9598}, dtype(kFloat).requires_grad(false));
  }
  else
  {
    cerr << boost::format("env_id: %d is not implemented.") % config.env_id << endl;
    return 1;
  }

  auto agent = Agent(envs[0]->get_observation_space(), envs[0]->get_action_space(),
                     envs[0]->get_action_space_max(), envs[0]->get_action_space_min(),
                     observation_mean, observation_std);

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

  // Stores obs of the run.
  std::vector<Tensor> obs_mean_std;

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
          tie(action, logprob, entropy, value) = agent->get_action_and_value(next_obs.index({i}).unsqueeze(0).to(collect_devices), at::empty({0}), "sample", agent_generators[i]);

          values.index({step}).slice(0, i,i+1) = value.squeeze(0);
          actions.index({step, i}) = action.squeeze(0);
          logprobs.index({step, i}) = logprob.index({0});

          auto [env_next_obs, reward, termination, truncation, infos] = envs[i]->step(action.to(kCPU));
          // Store states for estimating mean and std at the end.
          if (i == 0 && config.estimate_mean_std) {
            obs_mean_std.push_back(env_next_obs.index({0}).clone());
          }
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

    if (config.render == "human"s) {
      // The main thread need to occasionally call this, otherwise the OS thinks the rendering windows are unresponsive.
      // Not threadsafe, so we cannot call it from the data collection threads.
      glfwPollEvents();
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
        auto [action, newlogprob, entropy, newvalue] = agent->get_action_and_value(b_obs.index({mb_inds}), b_actions.index({mb_inds}), "sample"s);
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

  // Uncomment when estimating mean and std.
  if (config.estimate_mean_std) {
    auto stacked = torch::stack(obs_mean_std, 0);
    auto mean_builtin = torch::mean(stacked, 0);
    auto std_builtin  = torch::std(stacked, 0, /*unbiased=*/false); // population std

    std::cout << "Mean obs:\n" << mean_builtin << "\n\n";
    std::cout << "Std obs:\n" << std_builtin << "\n" << std::endl;
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
        tie(action, logprob, entropy, value) = agent->get_action_and_value(next_obs_eval.to(train_devices),  at::empty({0}), "mean"s);
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
