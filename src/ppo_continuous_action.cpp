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

#include <torch/torch.h>
#include <tqdm/tqdm.hpp>
#include "tensorboard_logger.h"
#include <args.hxx>
#include <boost/format.hpp>

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
  int seed = 1;
  int eval_seed = 2;
  int total_timesteps = 1'000'000;
  float learning_rate = 3e-4;
  int num_envs = 1;
  int num_steps = 2048;
  float gamma = 0.99;
  float gae_lambda = 0.95;
  int num_minibatches = 32;
  int update_epochs = 10;
  bool norm_adv = true;
  float clip_coef = 0.2;
  bool clip_vloss = true;
  float ent_coef = 0.0;
  float vf_coef = 0.5;
  float max_grad_norm = 0.5;
  float adam_eps = 1e-5;
  bool anneal_lr = true;
  int num_eval_runs = 10;
  bool clip_actions = true;
  bool torch_deterministic = true;
  string exp_name_stem = "PPO_002"s;
  string env_id = "Humanoid-v4"s;

  string exp_name = (boost::format("%s_%d") % exp_name_stem % seed).str();
  int batch_size {num_steps * num_envs};
  int minibatch_size {batch_size / num_minibatches};
  int num_iterations {total_timesteps / batch_size};

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
                  "|num_minibatches|%d|\n"
                  "|update_epochs|%d|\n"
                  "|norm_adv|%d|\n"
                  "|clip_coef|%f|\n"
                  "|clip_vloss|%d|\n"
                  "|ent_coef|%f|\n"
                  "|vf_coef|%f|\n"
                  "|max_grad_norm|%f|\n"
                  "|adam_eps|%f|\n"
                  "|anneal_lr|%d|\n"
                  "|num_eval_runs|%d|\n"
                  "|clip_actions|%d|\n"
                  "|exp_name|%s|\n"
                  "|batch_size|%d|\n"
                  "|minibatch_size|%d|\n"
                  "|num_iterations|%d|\n"
                  "|torch_deterministic|%d|\n"
                  "|exp_name_stem|%s|\n"
                  "|env_id|%s|\n")
                  % seed % eval_seed % total_timesteps % learning_rate % num_envs % num_steps % gamma % gae_lambda %
                  num_minibatches % update_epochs % norm_adv % clip_coef % clip_vloss % ent_coef % vf_coef % max_grad_norm %
                  adam_eps % anneal_lr % num_eval_runs % clip_actions % exp_name % batch_size % minibatch_size %
                  num_iterations % torch_deterministic % exp_name_stem % env_id).str();
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

    tuple<Tensor, Tensor, Tensor, Tensor> get_action_and_value(const Tensor& x, Tensor action=at::empty({0})) {
        const Tensor action_mean = actor_mean->forward(x);
        const Tensor action_logstd = actor_logstd.expand_as(action_mean);
        const Tensor action_std = torch::exp(action_logstd);
        const Normal probs(action_mean, action_std);
        if (action.data_ptr() == nullptr) {
            action = probs.sample(std::nullopt);
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
int main(const int argc, const char** argv) {
  ios_base::sync_with_stdio(false);  // Faster print
  // Can be slightly faster to turn off multi-threading in libtorch.
  // Somehow affects the result of computation though :/
   torch::set_num_threads(1);
   torch::set_num_interop_threads(1);

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

  // Update config with argparse arguments. Unfortunately we have to do this manually for every variable.
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

  // Need to recompute them as the value might have changed
  config.exp_name = (boost::format("%s_%d") % config.exp_name_stem % config.seed).str();
  config.batch_size = config.num_steps * config.num_envs;
  config.minibatch_size = config.batch_size / config.num_minibatches;
  config.num_iterations = config.total_timesteps / config.batch_size;

  filesystem::path exe = filesystem::canonical(argv[0]);
  filesystem::path basedir = exe.parent_path();
  filesystem::path exp_folder(basedir / ".." / "models");
  exp_folder = exp_folder / config.exp_name;
  filesystem::create_directories(exp_folder);

  GOOGLE_PROTOBUF_VERIFY_VERSION;
  TensorBoardLogger logger(exp_folder.string() + "/tfevents_logs.pb"s);
  logger.add_text("hyperparameters", 0, config.to_string().c_str());

  // Seed libtorch
  manual_seed(config.seed);
  at::globalContext().setDeterministicCuDNN(config.torch_deterministic);
  at::globalContext().setDeterministicAlgorithms(config.torch_deterministic, true);

  cout << "Parallelization mehtod: " << get_parallel_info() << endl;

  // CPU is a lot faster than GPU in ppo_continous_action.
  // Device collect_device(kCUDA, 0);
  Device collect_device(kCPU);
  // Device train_device(kCUDA, 0);
  Device train_device(kCPU);


  std::vector<shared_ptr<EnvironmentWrapper>> env_array;

  if (config.env_id == "Humanoid-v4") {
    filesystem::path mujoco_xml = basedir / "mujoco" / "assets" / "humanoid.xml";
    cout << "Loading file: " << mujoco_xml.string() << endl;
    for (int i = 0; i < config.num_envs; ++i) {
      auto env_0 = make_shared<HumanoidV4Env>(mujoco_xml.string(), "rgb_array");
      env_array.push_back(make_env(env_0, config.gamma));
    }
  }
  else if (config.env_id == "HalfCheetah-v5") {
    filesystem::path mujoco_xml = basedir / "mujoco" / "assets" / "half_cheetah.xml";
    cout << "Loading file: " << mujoco_xml.string() << endl;
    for (int i = 0; i < config.num_envs; ++i) {
      auto env_0 = make_shared<HalfCheetahV5Env>(mujoco_xml.string(), "rgb_array");
      env_array.push_back(make_env(env_0, config.gamma));
    }
  }
  else
  {
    cerr << (boost::format("env_id: %s is not implemented.") % config.env_id).str() << endl;
    return 1;
  }

  Agent agent(env_array[0]->get_observation_space(), env_array[0]->get_action_space());
  agent->to(collect_device);

  optim::Adam optimizer(agent->parameters(),
                        optim::AdamOptions(config.learning_rate).eps(config.adam_eps));

  auto x = agent->parameters();
  unsigned int num_params = 0;
  for (const auto& param: agent->parameters()) {
    if (param.requires_grad()) {
      num_params += param.numel();
    }
  }
  cout << "Number of parameters in model: " << num_params << endl;


  const auto envs = make_shared<ParVectorEnv>(env_array, config.clip_actions);

  // Storage setup
  Tensor obs        = torch::zeros({config.num_steps, config.num_envs, env_array[0]->get_observation_space()}, collect_device);
  Tensor actions    = torch::zeros({config.num_steps, config.num_envs, env_array[0]->get_action_space()}, collect_device);
  Tensor logprobs   = torch::zeros({config.num_steps, config.num_envs}, collect_device);
  Tensor rewards    = torch::zeros({config.num_steps, config.num_envs}, collect_device);
  Tensor dones      = torch::zeros({config.num_steps, config.num_envs}, collect_device);
  Tensor values     = torch::zeros({config.num_steps, config.num_envs}, collect_device);
  Tensor returns    = torch::zeros({config.num_steps, config.num_envs}, collect_device);
  Tensor advantages = torch::zeros({config.num_steps, config.num_envs}, collect_device);


  long global_step = 0;

  auto next_obs = envs->reset(config.seed);
  next_obs = next_obs.to(collect_device);
  Tensor next_done = torch::zeros({config.num_envs}, collect_device);

  std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
  TicToc tt;
  for (auto iteration : tq::trange(config.num_iterations)) {
    agent->to(collect_device);
    agent->eval();
    tt.tic();
    if (config.anneal_lr) {
      float frac = 1.0f - static_cast<float>(iteration) / static_cast<float>(config.num_iterations);
      float lrnow = frac * config.learning_rate;
      auto &options = static_cast<optim::OptimizerOptions &>(optimizer.param_groups()[0].options());
      options.set_lr(lrnow);
    }
    TicToc tl;
    vector<double> avg_env_time;
    for (int step = 0; step < config.num_steps; ++step) { // No grad
      NoGradGuard no_grad;
      global_step += config.num_envs;

      obs.index({step}) = next_obs;
      dones.index({step}) = next_done;

      Tensor action, logprob, entropy, value, done;
      tie(action, logprob, entropy, value) = agent->get_action_and_value(next_obs.to(collect_device));
      values.index({step}) = value.flatten();

      actions.index({step}) = action;
      logprobs.index({step}) = logprob;

      tl.tic();
      auto [env_next_obs, reward, termination, truncation, infos] = envs->step(action.to(kCPU));
      avg_env_time.push_back(tl.tocvalue());

      done = torch::logical_or(termination, truncation).to(kFloat32);
      rewards.index({step}) = reward.to(collect_device);
      next_obs = env_next_obs.to(collect_device);
      next_done = done.to(collect_device);

      float total_reward = 0.0f;
      int total_length = 0;
      int finished_envs = 0;
      for (const auto info : infos)
      {
        if (info.has_value()) {
          auto [r, l, t] = info.value();
          cout << (boost::format("global_step=%d, episodic_return=%.2f \n") % global_step % r).str();
          total_reward += r;
          total_length += l;
          finished_envs++;
        }
      }
      if (finished_envs > 0) {
        float avg_reward = total_reward / static_cast<float>(finished_envs);
        float avg_length = static_cast<float>(total_length) / static_cast<float>(finished_envs);
        logger.add_scalar("charts/episodic_return", static_cast<int>(global_step), avg_reward);
        logger.add_scalar("charts/episodic_length", static_cast<int>(global_step), avg_length);

        // Also log performance per wall clock time.
        chrono::high_resolution_clock::time_point current_time = chrono::high_resolution_clock::now();
        const long passed_seconds = lround(std::chrono::duration<float>(current_time - start_time).count());
        logger.add_scalar("charts/episodic_return_per_sec", static_cast<int>(passed_seconds), avg_reward);
      }
    }
    std::cout << std::fixed << std::setprecision(6) << "Total env step time" << " " << reduce(avg_env_time.begin(), avg_env_time.end()) << " seconds \n";
    tt.toc("Time to collect data:");

    tt.tic();
    // Computes Generalized advantage estimation labels
    { // No grad
      NoGradGuard noGrad;
      Tensor next_value = agent->get_value(next_obs).flatten();
      Tensor lastgaelam = torch::zeros({config.num_envs}, collect_device);
      Tensor nextnonterminal = at::empty({0}, collect_device);
      Tensor nextvalues = at::empty({0}, collect_device);
      for (int t = (config.num_steps-1); t >= 0; --t) {
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

    tt.toc("Time to compute advantage:");

    tt.tic();
    Tensor b_obs        = obs.reshape({-1, env_array[0]->get_observation_space()}).to(train_device);
    Tensor b_logprobs   = logprobs.reshape({-1}).to(train_device);
    Tensor b_actions    = actions.reshape({-1, env_array[0]->get_action_space()}).to(train_device);
    Tensor b_advantages = advantages.reshape(-1).to(train_device);
    Tensor b_returns    = returns.reshape(-1).to(train_device);
    Tensor b_values     = values.reshape(-1).to(train_device);

    // Values defined outside, so that we can log the last value.
    Tensor v_loss;
    Tensor pg_loss;
    Tensor entropy_loss;
    Tensor old_approx_kl;
    Tensor approx_kl;
    vector<float> clipfracs;
    agent->train();
    agent->to(train_device);
    // Optimize network
    for (int epoch = 0; epoch < config.update_epochs; ++epoch) {
      Tensor b_inds = torch::randperm(config.batch_size, TensorOptions().dtype(kLong).requires_grad(false));

      for (int start = 0; start < config.batch_size; start+=config.minibatch_size) {
        int end = start + config.minibatch_size;

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
          mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8);
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
        nn::utils::clip_grad_norm_(agent->parameters(), config.max_grad_norm);
        optimizer.step();
      }
    }
    tt.toc("Time to train");

    tt.tic();
    save_state(agent, optimizer, exp_folder, (boost::format("model_latest_%09d.pth") %  iteration).str(), (boost::format("optimizer_latest_%09d.pth") %  iteration).str());

    // Cleanup files from past iterations
    for (const auto& dirEntry : filesystem::directory_iterator(exp_folder)) {
      const auto filename = dirEntry.path().filename().string();
      if (filename.starts_with("model_latest_") and filename.ends_with(".pth")) {
          if(filename != (boost::format("model_latest_%09d.pth") %  iteration).str()) {
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
    float sps = 0;
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
    const float avg_clipfrac = reduce(clipfracs.begin(), clipfracs.end()) / static_cast<float>(clipfracs.size());
    logger.add_scalar("losses/clipfrac", static_cast<int>(global_step), avg_clipfrac);
    logger.add_scalar("charts/SPS", static_cast<int>(global_step), sps);

    tt.toc("Time to log");
  }
  save_state(agent, optimizer, exp_folder, "model_final.pth"s, "optimizer_final.pth"s);

  { // no grad
    NoGradGuard no_grad;
    // Evaluate final model

    agent->eval();
    // We are using the training envs to evaluate the model because the normalization wrappers have statistics
    // that get lost if you make a new env. If you want to actually evaluate a saved model with a new env you would need
    // to save an load those statistics as well.
    auto next_obs_eval = envs->reset(config.eval_seed);
    vector<float> episodic_returns;
    while (episodic_returns.size() < config.num_eval_runs) {
      Tensor action;
      {
        Tensor entropy, done, logprob, value; // Don't need this variable. Will be deleted once left scope.
        tie(action, logprob, entropy, value) = agent->get_action_and_value(next_obs_eval.to(collect_device));
      }
      auto [env_next_obs, reward, termination, truncation, infos] = envs->step(action.to(kCPU));

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
    cout << (boost::format("Average evaluation return=%.2f over %d episodes") % avg_return % episodic_returns.size()).str();
  }
  google::protobuf::ShutdownProtobufLibrary();
}
