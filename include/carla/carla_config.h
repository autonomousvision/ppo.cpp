//
// Config management for CARLA PPO.
// This type of config management system was designed for python where it is quite elegant due to reflections.
// C++ does not support reflections which makes this file a bit ugly.
// We keep it nevertheless to keep compatibility with the python system.

#ifndef CARLA_CONFIG_H
#define CARLA_CONFIG_H
#include <string>
#include <vector>
#include <limits>

#include <boost/format.hpp>
#include <args.hxx>
#include "json_spirit.h"

using namespace std;
using namespace json_spirit;

// When adding a new variable, also add it in to_json() and optionally update_config_with_args() if you want it to be
// configurable.
class GlobalConfig {
public:
  int seed = 1;  // negative values mean no seeding will take place.
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
  string lr_schedule = "linear";
  int num_eval_runs = 10;
  bool clip_actions = true;
  bool torch_deterministic = true;
  vector<int> gpu_ids = vector<int>({0});
  string collect_device = "gpu"s;  // Options: cpu, gpu
  string train_device = "gpu"s;  // Options: cpu, gpu
  string rdzv_addr = "localhost"s;  // IP adress of master node.
  uint16_t tcp_store_port = 29500u;  // Port for the TCP store.
  int use_dd_ppo_preempt = 0;  // Flag to toggle the dd_ppo pre-emption trick.
  float dd_ppo_min_perc = 0.25;  // Minimum percentage of env steps that need to finish before preemtion.
  float dd_ppo_preempt_threshold = 0.6;  // Minimum number of envs that need to finish before preemtion.
  vector<int> ports = vector<int>({5555});
  bool use_exploration_suggest = false; // Whether to use the exploration loss from roach. NOTE not implemented.
  bool use_speed_limit_as_max_speed = false; // If true rr_maximum_speed will be overwritten to the current speed limit affecting the ego vehicle.
  float beta_min_a_b_value = 1.0f; // Minimum value for a, b of the beta distribution that the model can predict. Gets added to the softplus output.
  bool use_new_bev_obs = false; // Whether to use bev_observation.py instead of chauffeurnet.py to render
  int obs_num_channels = 15; // Number of channels in the BEV image.
  string map_folder = "maps_low_res"; // Map folder for the preprocessed map data
  float pixels_per_meter = 5.0f; // 1 / pixels_per_meter = size of pixel in meters
  int route_width = 16; // Width of the rendered route in pixel.
  string reward_type = "roach"; // Reward function to be used during training. Options: roach, simple_reward
  bool consider_tl = true; // If set to false traffic light infractions are turned off. Used in simple reward
  float eval_time = 1200.0f; // Seconds. After this time a timeout is triggered in the reward which counts as truncation.
  float terminal_reward = 0.0f; // Reward at the end of the episode
  bool normalize_rewards = false; // Whether to use gymnasiums reward normalization. // TODO not implemented yet.
  bool speeding_infraction = false; // Whether to terminate the route if the agent drives too fast.
  float min_thresh_lat_dist = 3.5f; // Meters. If the agent is father away from the centerline (laterally) it counts as route deviation in the reward
  int num_route_points_rendered = 80; // Number of route points rendered into the BEV seg observation.
  bool use_green_wave = false; // If true in some routes all TL that the agent encounters are set to green.
  string image_encoder = "roach"; // Which image cnn encoder to use. Either roach, roach_ln
  bool use_comfort_infraction = false; // Whether to apply a soft penalty if comfort limits are exceeded
  float comfort_penalty_factor = 0.5f;  // Max comfort penalty if all comfort metrics are violated.
  bool use_layer_norm = false; // Whether to use LayerNorm before ReLU in MLPs.
  bool use_vehicle_close_penalty = false; //  Whether to use a penalty for being too close to the front vehicle.
  bool render_green_tl = true; // Whether to render green traffic lights into the observation.
  string distribution = "beta"; // Distribution used for the action space. Options beta // TODO implement gaus / beta_uni_mix
  float weight_decay = 0.0f; // Weight decay applied to the Adam optimizer.
  bool use_termination_hint = false; // Whether to give a penalty depending on vehicle speed when crashing or running red light.
  bool use_perc_progress = false; // Whether to multiply RC reward by percentage away from lane center.
  float lane_distance_violation_threshold = 0.0;  // Grace distance in m at which no lane perc penalty is applied
  float lane_dist_penalty_softener = 1.0;  //  If smaller than 1 reduces lane distance penalty.
  bool use_min_speed_infraction = false; // Whether to penalize the agent for driving slower than other agents on avg.
  bool use_leave_route_done = true; // Whether to terminate the route when leaving the precomputed path.
  int obs_num_measurements = 8; // Number of scalar measurements in observation.
  bool use_extra_control_inputs = false; // Whether to use extra control inputs such as integral of past steering.
  bool condition_outside_junction = true; // Whether to render the route outside junctions.
  // Applicable if use_layer_norm=True, whether to also apply layernorm to the policy head.
  bool use_layer_norm_policy_head = true; // Can be useful to remove to allow the policy to predict large values (for a, b of Beta).
  bool use_outside_route_lanes = false; // Whether to terminate the route when invading opposing lanes or sidewalks.
  bool use_max_change_penalty = false; // Whether to apply a soft penalty when the action changes too fast.
  float terminal_hint = 10.0f; // Reward at the end of the episode when colliding, the number will be subtracted.
  bool penalize_yellow_light = true; // Whether to penalize running a yellow light.
  bool use_target_point = false; // Whether to input a target point in the measurements.
  float speeding_multiplier = 0.0f; // Penalty for driving too fast
  bool use_value_measurements = true; // Whether to use value measurements (otherwise all are set to 0)
  int bev_semantics_width = 192; //  Numer of pixels the bev_semantics is wide
  int bev_semantics_height = 192; //  Numer of pixels the bev_semantics is high
  int num_value_measurements = 3; //  Number of measurements exclusive to the value head.
  int pixels_ev_to_bottom = 40; //  Numer of pixels from the vehicle to the bottom.
  bool use_history = false; //  Whether to use the history in bev_observation
  string team_code_folder = ""; // Path to the team_code folder where env_agent.py is. Required to be set by args.
  string load_file = "None"; // Path to an agent model to resume training from. If "None" no file will be loaded.
  bool debug = false;  // Whether to render debugging information.
  string debug_type = "render"; // Whether to render or save debug images. Options: render, save
  string logdir = ""; // Whether to render or save debug images. Options: render, save
  long global_step = 0;  // Number of environment samples that were already collected.
  float max_training_score = std::numeric_limits<float>::lowest();  // Highest training score achieved so far
  long best_iteration = 0;  // Iteration of the best model
  long latest_iteration = 0;  // Iteration of the latest model
  bool use_off_road_term = false;  // Whether to terminate when he agent drives off the drivable area
  float off_road_term_perc = 0.5;  // Percentage of agent overlap with off-road, that triggers the termination
  float beta_1 = 0.9;  // Beta 1 parameter of Adam
  float beta_2 = 0.999;  // Beta 2 parameter of Adam
  bool render_speed_lines = false; // Whether to render the speed lines for moving objects
  // Whether to use a different stop sign detector that prevents the policy from cheating by changing lanes.
  bool use_new_stop_sign_detector = false;
  bool use_positional_encoding = false;  // Whether to add positional encoding to the image
  bool use_ttc = false;  // Whether to use TTC in the reward.
  int ttc_resolution = 2;  // Interval of frame_rate time steps at which TTC is evaluated
  int ttc_penalty_ticks = 100;  // Number of simulator steps that a TTC penalty is applied for
  bool render_yellow_time = false;  // Whether to indicate the remaining time to red in yellow light rendering
  // Whether to only use RC als reward source in simple reward, else adds TTC, comfort and speed like in nuPlan
  bool use_single_reward = true;
  bool use_rl_termination_hint = false; // Whether to include red light infraction for termination hints
  bool render_shoulder = true;  // Whether to render shoulder lanes as roads.
  bool use_shoulder_channel = true;  // Whether to use an extra channel for shoulder lanes
  bool use_survival_reward = false;  // Whether to add a constant reward every frame
  float survival_reward_magnitude = 0.0001f;  // How large the survival reward is.

  // These variables are dependent on the other variables and need to be recomputed if they change.
  int num_devices = static_cast<int>(gpu_ids.size());
  string exp_name = (boost::format("%s_%d") % "PPO_002"s % seed).str();
  int batch_size {num_steps * num_envs};
  int minibatch_size {batch_size / num_minibatches};
  int num_iterations {total_timesteps / batch_size};
  int num_envs_per_proc = num_envs / num_devices;
  int batch_size_per_device = batch_size / num_devices;
  int minibatch_per_device = minibatch_size / num_devices;

  void update_config_with_args(int argc, char** argv, int world_size) {
    args::ArgumentParser parser("parser");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag seed(parser, "seed", "Training seed", {"seed"}, this->seed);
    args::ValueFlag<string> team_code_folder(parser, "team_code_folder", "Path to the team_code folder where env_agent.py is.", {"team_code_folder"}, args::Options::Required);
    args::ValueFlag total_timesteps(parser, "total_timesteps", "Number of environment steps", {"total_timesteps"}, this->total_timesteps);
    args::ValueFlag learning_rate(parser, "learning_rate", "Adam learning rate", {"learning_rate"}, this->learning_rate);
    args::ValueFlag num_steps(parser, "num_steps", "Num environment steps per iteration", {"num_steps"}, this->num_steps);
    args::ValueFlag gamma(parser, "gamma", "Discount factor", {"gamma"}, this->gamma);
    args::ValueFlag gae_lambda(parser, "gae_lambda", "Lambda of generalized advantage estimation", {"gae_lambda"}, this->gae_lambda);
    args::ValueFlag num_minibatches(parser, "num_minibatches", "Number of training iterations per epoch", {"num_minibatches"}, this->num_minibatches);
    args::ValueFlag update_epochs(parser, "update_epochs", "Number of training epochs per iteration", {"update_epochs"}, this->update_epochs);
    args::ValueFlag norm_adv(parser, "norm_adv", "Whether to normalize the advantage", {"norm_adv"}, this->norm_adv);
    args::ValueFlag clip_coef(parser, "clip_coef", "PPO clip coefficient", {"clip_coef"}, this->clip_coef);
    args::ValueFlag clip_vloss(parser, "clip_vloss", "Whether to apply clipping to the value loss", {"clip_vloss"}, this->clip_vloss);
    args::ValueFlag ent_coef(parser, "ent_coef", "Weigth of entropy loss.", {"ent_coef"}, this->ent_coef);
    args::ValueFlag vf_coef(parser, "vf_coef", "Weigth of value loss.", {"vf_coef"}, this->vf_coef);
    args::ValueFlag max_grad_norm(parser, "max_grad_norm", "Factor for gradient clipping.", {"max_grad_norm"}, this->max_grad_norm);
    args::ValueFlag adam_eps(parser, "adam_eps", "Epsilon of adam.", {"adam_eps"}, this->adam_eps);
    args::ValueFlag lr_schedule(parser, "lr_schedule", "Whether to anneal the learning rate linearly.", {"lr_schedule"}, this->lr_schedule);
    args::ValueFlag num_eval_runs(parser, "num_eval_runs", "How many environments to evaluate", {"num_eval_runs"}, this->num_eval_runs);
    args::ValueFlag clip_actions(parser, "clip_actions", "Whether to clip action into the valid range.", {"clip_actions"}, this->clip_actions);
    args::ValueFlag torch_deterministic(parser, "torch_deterministic", "Whether to use deterministic cuda algorithms", {"torch_deterministic"}, this->torch_deterministic);
    args::ValueFlag exp_name(parser, "exp_name", "Name of the experiment.", {"exp_name"}, this->exp_name);
    args::ValueFlag num_envs(parser, "num_envs", "Number of environments to be used.", {"num_envs"}, this->num_envs);
    args::ValueFlagList<int> gpu_ids(parser, "gpu_ids", "The ids of the GPUs used for training. Len of this list determines the number of devices."
                                                                 "Usage: --gpu_ids 0 --gpu_ids 1 --gpu_ids 2 ...", {"gpu_ids"}, this->gpu_ids);
    args::ValueFlag collect_device(parser, "collect_device", "Whether to collect data on gpu or cpu. Options: cpu, gpu", {"collect_device"}, this->collect_device);
    args::ValueFlag train_device(parser, "train_device", "Whether to train on gpu or cpu. Options: cpu, gpu", {"train_device"}, this->train_device);
    args::ValueFlag rdzv_addr(parser, "rdzv_addr", "IP adress of master node. Default: localhost", {"rdzv_addr"}, this->rdzv_addr);
    args::ValueFlag tcp_store_port(parser, "tcp_store_port", "Port for the TCP store. Default: 29500", {"tcp_store_port"}, this->tcp_store_port);
    args::ValueFlag use_dd_ppo_preempt(parser, "use_dd_ppo_preempt", "Flag to toggle the dd_ppo pre-emption trick", {"use_dd_ppo_preempt"}, this->use_dd_ppo_preempt);
    args::ValueFlag dd_ppo_min_perc(parser, "dd_ppo_min_perc", "Percentage of envs that need to finish before preemtion.", {"dd_ppo_min_perc"}, this->dd_ppo_min_perc);
    args::ValueFlag dd_ppo_preempt_threshold(parser, "dd_ppo_preempt_threshold", "Percentage of envs that need to finish before preemtion.", {"dd_ppo_preempt_threshold"}, this->dd_ppo_preempt_threshold);
    args::ValueFlagList<int> ports(parser, "ports", "Ports of the carla_gym wrapper to connect to. It requires to submit a port for every envs. #ports == --nproc_per_node"
                                                             "Usage: --ports 0 --ports 1 --ports 2 ...", {"ports"}, this->ports);
    args::ValueFlag use_exploration_suggest(parser, "use_exploration_suggest", "Whether to use the exploration loss from roach. NOTE not implemented.", {"use_exploration_suggest"}, this->use_exploration_suggest);
    args::ValueFlag use_speed_limit_as_max_speed(parser, "use_speed_limit_as_max_speed", "If true rr_maximum_speed will be overwritten to the current speed limit affecting the ego vehicle.", {"use_speed_limit_as_max_speed"}, this->use_speed_limit_as_max_speed);
    args::ValueFlag beta_min_a_b_value(parser, "beta_min_a_b_value", "Minimum value for a, b of the beta distribution that the model can predict. Gets added to the softplus output.", {"beta_min_a_b_value"}, this->beta_min_a_b_value);
    args::ValueFlag use_new_bev_obs(parser, "use_new_bev_obs", "Whether to use bev_observation.py instead of chauffeurnet.py to render", {"use_new_bev_obs"}, this->use_new_bev_obs);
    args::ValueFlag obs_num_channels(parser, "obs_num_channels", "Number of channels in the BEV image.", {"obs_num_channels"}, this->obs_num_channels);
    args::ValueFlag map_folder(parser, "map_folder", "Map folder for the preprocessed map data.", {"map_folder"}, this->map_folder);
    args::ValueFlag pixels_per_meter(parser, "pixels_per_meter", "1 / pixels_per_meter = size of pixel in meters.", {"pixels_per_meter"}, this->pixels_per_meter);
    args::ValueFlag route_width(parser, "route_width", "1 / pixels_per_meter = size of pixel in meters.", {"route_width"}, this->route_width);
    args::ValueFlag reward_type(parser, "reward_type", "Reward function to be used during training. Options: roach, simple_reward.", {"reward_type"}, this->reward_type);
    args::ValueFlag consider_tl(parser, "consider_tl", "If set to false traffic light infractions are turned off. Used in simple reward", {"consider_tl"}, this->consider_tl);
    args::ValueFlag eval_time(parser, "eval_time", "Seconds. After this time a timeout is triggered in the reward which counts as truncation.", {"eval_time"}, this->eval_time);
    args::ValueFlag terminal_reward(parser, "terminal_reward", "Reward at the end of the episode", {"terminal_reward"}, this->terminal_reward);
    args::ValueFlag normalize_rewards(parser, "normalize_rewards", "Whether to use gymnasiums reward normalization.", {"normalize_rewards"}, this->normalize_rewards);
    args::ValueFlag speeding_infraction(parser, "speeding_infraction", "Whether to terminate the route if the agent drives too fast.", {"speeding_infraction"}, this->speeding_infraction);
    args::ValueFlag min_thresh_lat_dist(parser, "min_thresh_lat_dist", "Meters. If the agent is father away from the centerline (laterally) it counts as route deviation in the reward", {"min_thresh_lat_dist"}, this->min_thresh_lat_dist);
    args::ValueFlag num_route_points_rendered(parser, "num_route_points_rendered", "Number of route points rendered into the BEV seg observation.", {"num_route_points_rendered"}, this->num_route_points_rendered);
    args::ValueFlag use_green_wave(parser, "use_green_wave", "If true in some routes all TL that the agent encounters are set to green.", {"use_green_wave"}, this->use_green_wave);
    args::ValueFlag image_encoder(parser, "image_encoder", "Which image cnn encoder to use. Either roach, roach_ln.", {"image_encoder"}, this->image_encoder);
    args::ValueFlag use_comfort_infraction(parser, "use_comfort_infraction", "Whether to apply a soft penalty if comfort limits are exceeded", {"use_comfort_infraction"}, this->use_comfort_infraction);
    args::ValueFlag comfort_penalty_factor(parser, "comfort_penalty_factor", "Max comfort penalty if all comfort metrics are violated.", {"comfort_penalty_factor"}, this->comfort_penalty_factor);
    args::ValueFlag use_layer_norm(parser, "use_layer_norm", "Whether to use LayerNorm before ReLU in MLPs.", {"use_layer_norm"}, this->use_layer_norm);
    args::ValueFlag use_vehicle_close_penalty(parser, "use_vehicle_close_penalty", " Whether to use a penalty for being too close to the front vehicle.", {"use_vehicle_close_penalty"}, this->use_vehicle_close_penalty);
    args::ValueFlag render_green_tl(parser, "render_green_tl", " Whether to render green traffic lights into the observation.", {"render_green_tl"}, this->render_green_tl);
    args::ValueFlag distribution(parser, "distribution", "Distribution used for the action space. Options beta.", {"distribution"}, this->distribution);
    args::ValueFlag weight_decay(parser, "weight_decay", "Weight decay applied to optimizer.", {"weight_decay"}, this->weight_decay);
    args::ValueFlag use_termination_hint(parser, "use_termination_hint", "Whether to give a penalty depending on vehicle speed when crashing or running red light.", {"use_termination_hint"}, this->use_termination_hint);
    args::ValueFlag use_perc_progress(parser, "use_perc_progress", "Whether to multiply RC reward by percentage away from lane center.", {"use_perc_progress"}, this->use_perc_progress);
    args::ValueFlag lane_distance_violation_threshold(parser, "lane_distance_violation_threshold", "Grace distance in m at which no lane perc penalty is applied", {"lane_distance_violation_threshold"}, this->lane_distance_violation_threshold);
    args::ValueFlag lane_dist_penalty_softener(parser, "lane_dist_penalty_softener", "If smaller than 1 reduces lane distance penalty.", {"lane_dist_penalty_softener"}, this->lane_dist_penalty_softener);
    args::ValueFlag use_min_speed_infraction(parser, "use_min_speed_infraction", "Whether to penalize the agent for driving slower than other agents on avg.", {"use_min_speed_infraction"}, this->use_min_speed_infraction);
    args::ValueFlag use_leave_route_done(parser, "use_leave_route_done", "Whether to terminate the route when leaving the precomputed path.", {"use_leave_route_done"}, this->use_leave_route_done);
    args::ValueFlag obs_num_measurements(parser, "obs_num_measurements", "Number of scalar measurements in observation.", {"obs_num_measurements"}, this->obs_num_measurements);
    args::ValueFlag use_extra_control_inputs(parser, "use_extra_control_inputs", "Whether to use extra control inputs such as integral of past steering.", {"use_extra_control_inputs"}, this->use_extra_control_inputs);
    args::ValueFlag condition_outside_junction(parser, "condition_outside_junction", "Whether to render the route outside junctions.", {"condition_outside_junction"}, this->condition_outside_junction);
    args::ValueFlag use_layer_norm_policy_head(parser, "use_layer_norm_policy_head", "Applicable if use_layer_norm=True, whether to also apply layernorm to the policy head. Can be useful to remove to allow the policy to predict large values (for a, b of Beta).", {"use_layer_norm_policy_head"}, this->use_layer_norm_policy_head);
    args::ValueFlag use_outside_route_lanes(parser, "use_outside_route_lanes", "Whether to terminate the route when invading opposing lanes or sidewalks.", {"use_outside_route_lanes"}, this->use_outside_route_lanes);
    args::ValueFlag use_max_change_penalty(parser, "use_max_change_penalty", "Whether to apply a soft penalty when the action changes too fast.", {"use_max_change_penalty"}, this->use_max_change_penalty);
    args::ValueFlag terminal_hint(parser, "terminal_hint", "Reward at the end of the episode when colliding, the number will be subtracted.", {"terminal_hint"}, this->terminal_hint);
    args::ValueFlag penalize_yellow_light(parser, "penalize_yellow_light", "Whether to penalize running a yellow light.", {"penalize_yellow_light"}, this->penalize_yellow_light);
    args::ValueFlag use_target_point(parser, "use_target_point", "Whether to input a target point in the measurements.", {"use_target_point"}, this->use_target_point);
    args::ValueFlag speeding_multiplier(parser, "speeding_multiplier", "Penalty for driving too fast.", {"speeding_multiplier"}, this->speeding_multiplier);
    args::ValueFlag use_value_measurements(parser, "use_value_measurements", "Whether to use value measurements (otherwise all are set to 0)", {"use_value_measurements"}, this->use_value_measurements);
    args::ValueFlag bev_semantics_width(parser, "bev_semantics_width", " Numer of pixels the bev_semantics is wide", {"bev_semantics_width"}, this->bev_semantics_width);
    args::ValueFlag bev_semantics_height(parser, "bev_semantics_height", " Numer of pixels the bev_semantics is high", {"bev_semantics_height"}, this->bev_semantics_height);
    args::ValueFlag pixels_ev_to_bottom(parser, "pixels_ev_to_bottom", "Numer of pixels from the vehicle to the bottom.", {"pixels_ev_to_bottom"}, this->pixels_ev_to_bottom);
    args::ValueFlag num_value_measurements(parser, "num_value_measurements", "Number of measurements exclusive to the value head.", {"num_value_measurements"}, this->num_value_measurements);
    args::ValueFlag use_history(parser, "use_history", "Whether to use the history in bev_observation", {"use_history"}, this->use_history);
    args::ValueFlag load_file(parser, "load_file", "Path to an agent model to resume training from. If None no file will be loaded.", {"load_file"}, this->load_file);
    args::ValueFlag debug(parser, "debug", "Whether to render debugging information.", {"debug"}, this->debug);
    args::ValueFlag debug_type(parser, "debug_type", "Whether to render or save debug images. Options: render, save", {"debug_type"}, this->debug_type);
    args::ValueFlag logdir(parser, "logdir", "Path to directory where model will be saved", {"logdir"}, this->logdir);
    args::ValueFlag use_off_road_term(parser, "use_off_road_term", "Whether to terminate when he agent drives off the drivable area", {"use_off_road_term"}, this->use_off_road_term);
    args::ValueFlag off_road_term_perc(parser, "off_road_term_perc", "Percentage of agent overlap with off-road, that triggers the termination", {"off_road_term_perc"}, this->off_road_term_perc);
    args::ValueFlag beta_1(parser, "beta_1", " Beta 1 parameter of Adam", {"beta_1"}, this->beta_1);
    args::ValueFlag beta_2(parser, "beta_2", " Beta 2 parameter of Adam", {"beta_2"}, this->beta_2);
    args::ValueFlag render_speed_lines(parser, "render_speed_lines", "Whether to render the speed lines for moving objects", {"render_speed_lines"}, this->render_speed_lines);
    args::ValueFlag use_new_stop_sign_detector(parser, "use_new_stop_sign_detector", "Whether to use a different stop sign detector that prevents the policy from cheating by changing lanes.", {"use_new_stop_sign_detector"}, this->use_new_stop_sign_detector);
    args::ValueFlag use_positional_encoding(parser, "use_positional_encoding", "Whether to add positional encoding to the image", {"use_positional_encoding"}, this->use_positional_encoding);
    args::ValueFlag use_ttc(parser, "use_ttc", "Whether to use TTC in the reward.", {"use_ttc"}, this->use_ttc);
    args::ValueFlag render_yellow_time(parser, "render_yellow_time", "Whether to indicate the remaining time to red in yellow light rendering", {"render_yellow_time"}, this->render_yellow_time);
    args::ValueFlag use_single_reward(parser, "use_single_reward", "Whether to only use RC als reward source in simple reward, else adds TTC, comfort and speed like in nuPlan", {"use_single_reward"}, this->use_single_reward);
    args::ValueFlag use_rl_termination_hint(parser, "use_rl_termination_hint", "Whether to include red light infraction for termination hints", {"use_rl_termination_hint"}, this->use_rl_termination_hint);
    args::ValueFlag render_shoulder(parser, "render_shoulder", " Whether to render shoulder lanes as roads.", {"render_shoulder"}, this->render_shoulder);
    args::ValueFlag use_shoulder_channel(parser, "use_shoulder_channel", "Whether to use an extra channel for shoulder lanes", {"use_shoulder_channel"}, this->use_shoulder_channel);
    args::ValueFlag use_survival_reward(parser, "use_survival_reward", "Whether to add a constant reward every frame", {"use_survival_reward"}, this->use_survival_reward);

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
    this->seed = args::get(seed);
    this->total_timesteps = args::get(total_timesteps);
    this->learning_rate = args::get(learning_rate);
    this->num_steps = args::get(num_steps);
    this->gamma = args::get(gamma);
    this->gae_lambda = args::get(gae_lambda);
    this->num_minibatches = args::get(num_minibatches);
    this->update_epochs = args::get(update_epochs);
    this->norm_adv = args::get(norm_adv);
    this->clip_coef = args::get(clip_coef);
    this->clip_vloss = args::get(clip_vloss);
    this->ent_coef = args::get(ent_coef);
    this->vf_coef = args::get(vf_coef);
    this->max_grad_norm = args::get(max_grad_norm);
    this->adam_eps = args::get(adam_eps);
    this->lr_schedule = args::get(lr_schedule);
    this->num_eval_runs = args::get(num_eval_runs);
    this->clip_actions = args::get(clip_actions);
    this->torch_deterministic = args::get(torch_deterministic);
    this->exp_name = args::get(exp_name);
    this->num_envs = args::get(num_envs);
    this->gpu_ids = args::get(gpu_ids);
    this->collect_device = args::get(collect_device);
    this->train_device = args::get(train_device);
    this->rdzv_addr = args::get(rdzv_addr);
    this->tcp_store_port = args::get(tcp_store_port);
    this->use_dd_ppo_preempt = args::get(use_dd_ppo_preempt);
    this->dd_ppo_min_perc = args::get(dd_ppo_min_perc);
    this->dd_ppo_preempt_threshold = args::get(dd_ppo_preempt_threshold);
    this->ports = args::get(ports);
    this->use_speed_limit_as_max_speed = args::get(use_speed_limit_as_max_speed);
    this->beta_min_a_b_value = args::get(beta_min_a_b_value);
    this->use_new_bev_obs = args::get(use_new_bev_obs);
    this->obs_num_channels = args::get(obs_num_channels);
    this->map_folder = args::get(map_folder);
    this->pixels_per_meter = args::get(pixels_per_meter);
    this->route_width = args::get(route_width);
    this->reward_type = args::get(reward_type);
    this->consider_tl = args::get(consider_tl);
    this->eval_time = args::get(eval_time);
    this->terminal_reward = args::get(terminal_reward);
    this->normalize_rewards = args::get(normalize_rewards);
    this->speeding_infraction = args::get(speeding_infraction);
    this->min_thresh_lat_dist = args::get(min_thresh_lat_dist);
    this->num_route_points_rendered = args::get(num_route_points_rendered);
    this->use_green_wave = args::get(use_green_wave);
    this->image_encoder = args::get(image_encoder);
    this->use_comfort_infraction = args::get(use_comfort_infraction);
    this->comfort_penalty_factor = args::get(comfort_penalty_factor);
    this->use_layer_norm = args::get(use_layer_norm);
    this->use_vehicle_close_penalty = args::get(use_vehicle_close_penalty);
    this->render_green_tl = args::get(render_green_tl);
    this->distribution = args::get(distribution);
    this->weight_decay = args::get(weight_decay);
    this->use_termination_hint = args::get(use_termination_hint);
    this->use_perc_progress = args::get(use_perc_progress);
    this->lane_distance_violation_threshold = args::get(lane_distance_violation_threshold);
    this->lane_dist_penalty_softener = args::get(lane_dist_penalty_softener);
    this->use_min_speed_infraction = args::get(use_min_speed_infraction);
    this->use_leave_route_done = args::get(use_leave_route_done);
    this->obs_num_measurements = args::get(obs_num_measurements);
    this->use_extra_control_inputs = args::get(use_extra_control_inputs);
    this->condition_outside_junction = args::get(condition_outside_junction);
    this->use_layer_norm_policy_head = args::get(use_layer_norm_policy_head);
    this->use_outside_route_lanes = args::get(use_outside_route_lanes);
    this->use_max_change_penalty = args::get(use_max_change_penalty);
    this->terminal_hint = args::get(terminal_hint);
    this->penalize_yellow_light = args::get(penalize_yellow_light);
    this->use_target_point = args::get(use_target_point);
    this->speeding_multiplier = args::get(speeding_multiplier);
    this->use_value_measurements = args::get(use_value_measurements);
    this->bev_semantics_width = args::get(bev_semantics_width);
    this->bev_semantics_height = args::get(bev_semantics_height);
    this->pixels_ev_to_bottom = args::get(pixels_ev_to_bottom);
    this->num_value_measurements = args::get(num_value_measurements);
    this->use_history = args::get(use_history);
    this->team_code_folder = args::get(team_code_folder);
    this->load_file = args::get(load_file);
    this->debug = args::get(debug);
    this->debug_type = args::get(debug_type);
    this->logdir = args::get(logdir);
    this->use_off_road_term = args::get(use_off_road_term);
    this->off_road_term_perc = args::get(off_road_term_perc);
    this->beta_1 = args::get(beta_1);
    this->beta_2 = args::get(beta_2);
    this->render_speed_lines = args::get(render_speed_lines);
    this->use_new_stop_sign_detector = args::get(use_new_stop_sign_detector);
    this->use_positional_encoding = args::get(use_positional_encoding);
    this->use_ttc = args::get(use_ttc);
    this->render_yellow_time = args::get(render_yellow_time);
    this->use_single_reward = args::get(use_single_reward);
    this->use_rl_termination_hint = args::get(use_rl_termination_hint);
    this->render_shoulder = args::get(render_shoulder);
    this->use_shoulder_channel = args::get(use_shoulder_channel);
    this->use_survival_reward = args::get(use_survival_reward);

    // Need to recompute them as the value might have changed
    this->num_devices = world_size;  // TODO think about how to do this with multi-node
    this->num_envs_per_proc = this->num_envs / this->num_devices;
    if(this->num_envs % this->num_devices != 0) {
      throw std::runtime_error("num_envs must be a multiple of num_devices.");
    }
    if(this->batch_size % this->num_minibatches != 0) {
      throw std::runtime_error("The batch size must be divisible by the minibatch size.");
    }
    if(this->batch_size % this->num_devices != 0) {
      throw std::runtime_error("The batch size must be divisible by the number of devices.");
    }
    if(this->minibatch_size % this->num_devices != 0) {
      throw std::runtime_error("The minibatch size must be divisible by the number of devices.");
    }

    this->batch_size = this->num_steps * this->num_envs;
    this->minibatch_size = this->batch_size / this->num_minibatches;
    this->num_iterations = this->total_timesteps / this->batch_size;
    this->batch_size_per_device = this->batch_size / this->num_devices;
    this->minibatch_per_device = this->minibatch_size / this->num_devices;
  }

  [[nodiscard]] string to_json() {
    Object config_json;
    config_json.emplace_back("py/object", "rl_config.GlobalConfig");
    config_json.emplace_back("seed", this->seed);
    config_json.emplace_back("total_timesteps", this->total_timesteps);
    config_json.emplace_back("learning_rate", this->learning_rate);
    config_json.emplace_back("num_envs", this->num_envs);
    config_json.emplace_back("num_steps", this->num_steps);
    config_json.emplace_back("gamma", this->gamma);
    config_json.emplace_back("gae_lambda", this->gae_lambda);
    config_json.emplace_back("num_minibatches", this->num_minibatches);
    config_json.emplace_back("update_epochs", this->update_epochs);
    config_json.emplace_back("norm_adv", this->norm_adv);
    config_json.emplace_back("clip_coef", this->clip_coef);
    config_json.emplace_back("clip_vloss", this->clip_vloss);
    config_json.emplace_back("ent_coef", this->ent_coef);
    config_json.emplace_back("vf_coef", this->vf_coef);
    config_json.emplace_back("max_grad_norm", this->max_grad_norm);
    config_json.emplace_back("adam_eps", this->adam_eps);
    config_json.emplace_back("lr_schedule", this->lr_schedule);
    config_json.emplace_back("num_eval_runs", this->num_eval_runs);
    config_json.emplace_back("clip_actions", this->clip_actions);
    config_json.emplace_back("torch_deterministic", this->torch_deterministic);
    // Would need to define vector to python tuple but this variable is not really needed to be logged.
    // config_json.emplace_back("gpu_ids", this->gpu_ids);
    config_json.emplace_back("collect_device", this->collect_device);
    config_json.emplace_back("train_device", this->train_device);
    config_json.emplace_back("rdzv_addr", this->rdzv_addr);
    config_json.emplace_back("tcp_store_port", this->tcp_store_port);
    config_json.emplace_back("use_dd_ppo_preempt", this->use_dd_ppo_preempt);
    config_json.emplace_back("dd_ppo_min_perc", this->dd_ppo_min_perc);
    config_json.emplace_back("dd_ppo_preempt_threshold", this->dd_ppo_preempt_threshold);
    config_json.emplace_back("num_devices", this->num_devices);
    config_json.emplace_back("exp_name", this->exp_name);
    config_json.emplace_back("batch_size", this->batch_size);
    config_json.emplace_back("minibatch_size", this->minibatch_size);
    config_json.emplace_back("num_iterations", this->num_iterations);
    config_json.emplace_back("num_envs_per_proc", this->num_envs_per_proc);
    config_json.emplace_back("batch_size_per_device", this->batch_size_per_device);
    config_json.emplace_back("minibatch_per_device", this->minibatch_per_device);
    // config_json.emplace_back("ports", this->ports);
    config_json.emplace_back("use_exploration_suggest", this->use_exploration_suggest);
    config_json.emplace_back("use_speed_limit_as_max_speed", this->use_speed_limit_as_max_speed);
    config_json.emplace_back("beta_min_a_b_value", this->beta_min_a_b_value);
    config_json.emplace_back("use_new_bev_obs", this->use_new_bev_obs);
    config_json.emplace_back("obs_num_channels", this->obs_num_channels);
    config_json.emplace_back("map_folder", this->map_folder);
    config_json.emplace_back("pixels_per_meter", this->pixels_per_meter);
    config_json.emplace_back("route_width", this->route_width);
    config_json.emplace_back("reward_type", this->reward_type);
    config_json.emplace_back("consider_tl", this->consider_tl);
    config_json.emplace_back("eval_time", this->eval_time);
    config_json.emplace_back("terminal_reward", this->terminal_reward);
    config_json.emplace_back("normalize_rewards", this->normalize_rewards);
    config_json.emplace_back("speeding_infraction", this->speeding_infraction);
    config_json.emplace_back("min_thresh_lat_dist", this->min_thresh_lat_dist);
    config_json.emplace_back("num_route_points_rendered", this->num_route_points_rendered);
    config_json.emplace_back("use_green_wave", this->use_green_wave);
    config_json.emplace_back("image_encoder", this->image_encoder);
    config_json.emplace_back("use_comfort_infraction", this->use_comfort_infraction);
    config_json.emplace_back("comfort_penalty_factor", this->comfort_penalty_factor);
    config_json.emplace_back("use_layer_norm", this->use_layer_norm);
    config_json.emplace_back("use_vehicle_close_penalty", this->use_vehicle_close_penalty);
    config_json.emplace_back("render_green_tl", this->render_green_tl);
    config_json.emplace_back("distribution", this->distribution);
    config_json.emplace_back("weight_decay", this->weight_decay);
    config_json.emplace_back("use_termination_hint", this->use_termination_hint);
    config_json.emplace_back("use_perc_progress", this->use_perc_progress);
    config_json.emplace_back("lane_distance_violation_threshold", this->lane_distance_violation_threshold);
    config_json.emplace_back("lane_dist_penalty_softener", this->lane_dist_penalty_softener);
    config_json.emplace_back("use_min_speed_infraction", this->use_min_speed_infraction);
    config_json.emplace_back("use_leave_route_done", this->use_leave_route_done);
    config_json.emplace_back("obs_num_measurements", this->obs_num_measurements);
    config_json.emplace_back("use_extra_control_inputs", this->use_extra_control_inputs);
    config_json.emplace_back("condition_outside_junction", this->condition_outside_junction);
    config_json.emplace_back("use_layer_norm_policy_head", this->use_layer_norm_policy_head);
    config_json.emplace_back("use_outside_route_lanes", this->use_outside_route_lanes);
    config_json.emplace_back("use_max_change_penalty", this->use_max_change_penalty);
    config_json.emplace_back("terminal_hint", this->terminal_hint);
    config_json.emplace_back("penalize_yellow_light", this->penalize_yellow_light);
    config_json.emplace_back("use_target_point", this->use_target_point);
    config_json.emplace_back("speeding_multiplier", this->speeding_multiplier);
    config_json.emplace_back("use_value_measurements", this->use_value_measurements);
    config_json.emplace_back("bev_semantics_width", this->bev_semantics_width);
    config_json.emplace_back("bev_semantics_height", this->bev_semantics_height);
    config_json.emplace_back("num_value_measurements", this->num_value_measurements);
    config_json.emplace_back("pixels_ev_to_bottom", this->pixels_ev_to_bottom);
    config_json.emplace_back("use_history", this->use_history);
    config_json.emplace_back("team_code_folder", this->team_code_folder);
    config_json.emplace_back("load_file", this->load_file);
    config_json.emplace_back("debug", this->debug);
    config_json.emplace_back("debug_type", this->debug_type);
    config_json.emplace_back("logdir", this->logdir);
    config_json.emplace_back("global_step", this->global_step);
    config_json.emplace_back("max_training_score", this->max_training_score);
    config_json.emplace_back("best_iteration", this->best_iteration);
    config_json.emplace_back("latest_iteration", this->latest_iteration);
    config_json.emplace_back("use_off_road_term", this->use_off_road_term);
    config_json.emplace_back("off_road_term_perc", this->off_road_term_perc);
    config_json.emplace_back("beta_1", this->beta_1);
    config_json.emplace_back("beta_2", this->beta_2);
    config_json.emplace_back("render_speed_lines", this->render_speed_lines);
    config_json.emplace_back("use_new_stop_sign_detector", this->use_new_stop_sign_detector);
    config_json.emplace_back("use_positional_encoding", this->use_positional_encoding);
    config_json.emplace_back("use_ttc", this->use_ttc);
    config_json.emplace_back("ttc_resolution", this->ttc_resolution);
    config_json.emplace_back("ttc_penalty_ticks", this->ttc_penalty_ticks);
    config_json.emplace_back("render_yellow_time", this->render_yellow_time);
    config_json.emplace_back("use_single_reward", this->use_single_reward);
    config_json.emplace_back("use_rl_termination_hint", this->use_rl_termination_hint);
    config_json.emplace_back("render_shoulder", this->render_shoulder);
    config_json.emplace_back("use_shoulder_channel", this->use_shoulder_channel);
    config_json.emplace_back("use_survival_reward", this->use_survival_reward);
    config_json.emplace_back("survival_reward_magnitude", this->survival_reward_magnitude);

    string output = json_spirit::write(config_json, pretty_print);
    return output;
  }

  void update_from_json(const string& path_to_config) {
    std::ifstream file(path_to_config);
    if (!file.is_open()) {
      std::cerr << "Could not open the file: " << path_to_config << std::endl;
      return;
    }

    json_spirit::Value config_json;
    json_spirit::read(file, config_json);
    json_spirit::Object config_object = config_json.get_obj();
    for (const auto& pair : config_object) {
      if (pair.name_ == "py/object") {
        // Itentionally left blank
      }
      else if (pair.name_ == "seed") {
        this->seed = pair.value_.get_int();
      }
      else if (pair.name_ == "total_timesteps") {
        this->total_timesteps = pair.value_.get_int();
      }
      else if (pair.name_ == "learning_rate") {
        this->learning_rate = static_cast<float>(pair.value_.get_real());
      }
      else if (pair.name_ == "num_envs") {
        this->num_envs = pair.value_.get_int();
      }
      else if (pair.name_ == "num_steps") {
        this->num_steps = pair.value_.get_int();
      }
      else if (pair.name_ == "gamma") {
        this->gamma = static_cast<float>(pair.value_.get_real());
      }
      else if (pair.name_ == "gae_lambda") {
        this->gae_lambda = static_cast<float>(pair.value_.get_real());
      }
      else if (pair.name_ == "num_minibatches") {
        this->num_minibatches = pair.value_.get_int();
      }
      else if (pair.name_ == "update_epochs") {
        this->update_epochs = pair.value_.get_int();
      }
      else if (pair.name_ == "norm_adv") {
        this->norm_adv = pair.value_.get_bool();
      }
      else if (pair.name_ == "clip_coef") {
        this->clip_coef = static_cast<float>(pair.value_.get_real());
      }
      else if (pair.name_ == "clip_vloss") {
        this->clip_vloss = pair.value_.get_bool();
      }
      else if (pair.name_ == "ent_coef") {
        this->ent_coef = static_cast<float>(pair.value_.get_real());
      }
      else if (pair.name_ == "vf_coef") {
        this->vf_coef = static_cast<float>(pair.value_.get_real());
      }
      else if (pair.name_ == "max_grad_norm") {
        this->max_grad_norm = static_cast<float>(pair.value_.get_real());
      }
      else if (pair.name_ == "adam_eps") {
        this->adam_eps = static_cast<float>(pair.value_.get_real());
      }
      else if (pair.name_ == "lr_schedule") {
        this->lr_schedule = pair.value_.get_str();
      }
      else if (pair.name_ == "num_eval_runs") {
        this->num_eval_runs = pair.value_.get_int();
      }
      else if (pair.name_ == "clip_actions") {
        this->clip_actions = pair.value_.get_bool();
      }
      else if (pair.name_ == "torch_deterministic") {
        this->torch_deterministic = pair.value_.get_bool();
      }
      else if (pair.name_ == "collect_device") {
        this->collect_device = pair.value_.get_str();
      }
      else if (pair.name_ == "train_device") {
        this->train_device = pair.value_.get_str();
      }
      else if (pair.name_ == "rdzv_addr") {
        this->rdzv_addr = pair.value_.get_str();
      }
      else if (pair.name_ == "tcp_store_port") {
        this->tcp_store_port = pair.value_.get_int();
      }
      else if (pair.name_ == "use_dd_ppo_preempt") {
        this->use_dd_ppo_preempt = pair.value_.get_int();
      }
      else if (pair.name_ == "dd_ppo_min_perc") {
        this->dd_ppo_min_perc = static_cast<float>(pair.value_.get_real());
      }
      else if (pair.name_ == "dd_ppo_preempt_threshold") {
        this->dd_ppo_preempt_threshold = static_cast<float>(pair.value_.get_real());
      }
      else if (pair.name_ == "num_devices") {
        this->num_devices = pair.value_.get_int();
      }
      else if (pair.name_ == "exp_name") {
        this->exp_name = pair.value_.get_str();
      }
      else if (pair.name_ == "batch_size") {
        this->batch_size = pair.value_.get_int();
      }
      else if (pair.name_ == "minibatch_size") {
        this->minibatch_size = pair.value_.get_int();
      }
      else if (pair.name_ == "num_iterations") {
        this->num_iterations = pair.value_.get_int();
      }
      else if (pair.name_ == "num_envs_per_proc") {
        this->num_envs_per_proc = pair.value_.get_int();
      }
      else if (pair.name_ == "batch_size_per_device") {
        this->batch_size_per_device = pair.value_.get_int();
      }
      else if (pair.name_ == "minibatch_per_device") {
        this->minibatch_per_device = pair.value_.get_int();
      }
      else if (pair.name_ == "use_exploration_suggest") {
        this->use_exploration_suggest = pair.value_.get_bool();
      }
      else if (pair.name_ == "use_speed_limit_as_max_speed") {
        this->use_speed_limit_as_max_speed = pair.value_.get_bool();
      }
      else if (pair.name_ == "beta_min_a_b_value") {
        this->beta_min_a_b_value = static_cast<float>(pair.value_.get_real());
      }
      else if (pair.name_ == "use_new_bev_obs") {
        this->use_new_bev_obs = pair.value_.get_bool();
      }
      else if (pair.name_ == "obs_num_channels") {
        this->obs_num_channels = pair.value_.get_int();
      }
      else if (pair.name_ == "map_folder") {
        this->map_folder = pair.value_.get_str();
      }
      else if (pair.name_ == "pixels_per_meter") {
        this->pixels_per_meter = static_cast<float>(pair.value_.get_real());
      }
      else if (pair.name_ == "route_width") {
        this->route_width = pair.value_.get_int();
      }
      else if (pair.name_ == "reward_type") {
        this->reward_type = pair.value_.get_str();
      }
      else if (pair.name_ == "consider_tl") {
        this->consider_tl = pair.value_.get_bool();
      }
      else if (pair.name_ == "eval_time") {
        this->eval_time = static_cast<float>(pair.value_.get_real());
      }
      else if (pair.name_ == "terminal_reward") {
        this->terminal_reward = static_cast<float>(pair.value_.get_real());
      }
      else if (pair.name_ == "normalize_rewards") {
        this->normalize_rewards = pair.value_.get_bool();
      }
      else if (pair.name_ == "speeding_infraction") {
        this->speeding_infraction = pair.value_.get_bool();
      }
      else if (pair.name_ == "min_thresh_lat_dist") {
        this->min_thresh_lat_dist = static_cast<float>(pair.value_.get_real());
      }
      else if (pair.name_ == "num_route_points_rendered") {
        this->num_route_points_rendered = pair.value_.get_int();
      }
      else if (pair.name_ == "use_green_wave") {
        this->use_green_wave = pair.value_.get_bool();
      }
      else if (pair.name_ == "image_encoder") {
        this->image_encoder = pair.value_.get_str();
      }
      else if (pair.name_ == "use_comfort_infraction") {
        this->use_comfort_infraction = pair.value_.get_bool();
      }
      else if (pair.name_ == "comfort_penalty_factor") {
        this->comfort_penalty_factor = static_cast<float>(pair.value_.get_real());
      }
      else if (pair.name_ == "use_layer_norm") {
        this->use_layer_norm = pair.value_.get_bool();
      }
      else if (pair.name_ == "use_vehicle_close_penalty") {
        this->use_vehicle_close_penalty = pair.value_.get_bool();
      }
      else if (pair.name_ == "render_green_tl") {
        this->render_green_tl = pair.value_.get_bool();
      }
      else if (pair.name_ == "distribution") {
        this->distribution = pair.value_.get_str();
      }
      else if (pair.name_ == "weight_decay") {
        this->weight_decay = static_cast<float>(pair.value_.get_real());
      }
      else if (pair.name_ == "use_termination_hint") {
        this->use_termination_hint = pair.value_.get_bool();
      }
      else if (pair.name_ == "use_perc_progress") {
        this->use_perc_progress = pair.value_.get_bool();
      }
      else if (pair.name_ == "lane_distance_violation_threshold") {
        this->lane_distance_violation_threshold = static_cast<float>(pair.value_.get_real());
      }
      else if (pair.name_ == "lane_dist_penalty_softener") {
        this->lane_dist_penalty_softener = static_cast<float>(pair.value_.get_real());
      }
      else if (pair.name_ == "use_min_speed_infraction") {
        this->use_min_speed_infraction = pair.value_.get_bool();
      }
      else if (pair.name_ == "use_leave_route_done") {
        this->use_leave_route_done = pair.value_.get_bool();
      }
      else if (pair.name_ == "obs_num_measurements") {
        this->obs_num_measurements = pair.value_.get_int();
      }
      else if (pair.name_ == "use_extra_control_inputs") {
        this->use_extra_control_inputs = pair.value_.get_bool();
      }
      else if (pair.name_ == "condition_outside_junction") {
        this->condition_outside_junction = pair.value_.get_bool();
      }
      else if (pair.name_ == "use_layer_norm_policy_head") {
        this->use_layer_norm_policy_head = pair.value_.get_bool();
      }
      else if (pair.name_ == "use_outside_route_lanes") {
        this->use_outside_route_lanes = pair.value_.get_bool();
      }
      else if (pair.name_ == "use_max_change_penalty") {
        this->use_max_change_penalty = pair.value_.get_bool();
      }
      else if (pair.name_ == "terminal_hint") {
        this->terminal_hint = static_cast<float>(pair.value_.get_real());
      }
      else if (pair.name_ == "penalize_yellow_light") {
        this->penalize_yellow_light = pair.value_.get_bool();
      }
      else if (pair.name_ == "use_target_point") {
        this->use_target_point = pair.value_.get_bool();
      }
      else if (pair.name_ == "speeding_multiplier") {
        this->speeding_multiplier = static_cast<float>(pair.value_.get_real());
      }
      else if (pair.name_ == "use_value_measurements") {
        this->use_value_measurements = pair.value_.get_bool();
      }
      else if (pair.name_ == "bev_semantics_width") {
        this->bev_semantics_width = pair.value_.get_int();
      }
      else if (pair.name_ == "bev_semantics_height") {
        this->bev_semantics_height = pair.value_.get_int();
      }
      else if (pair.name_ == "num_value_measurements") {
        this->num_value_measurements = pair.value_.get_int();
      }
      else if (pair.name_ == "pixels_ev_to_bottom") {
        this->pixels_ev_to_bottom = pair.value_.get_int();
      }
      else if (pair.name_ == "use_history") {
        this->use_history = pair.value_.get_bool();
      }
      else if (pair.name_ == "team_code_folder") {
        this->team_code_folder = pair.value_.get_str();
      }
      else if (pair.name_ == "load_file") {
        this->load_file = pair.value_.get_str();
      }
      else if (pair.name_ == "debug") {
        this->debug = pair.value_.get_bool();
      }
      else if (pair.name_ == "debug_type") {
        this->debug_type = pair.value_.get_str();
      }
      else if (pair.name_ == "logdir") {
        this->logdir = pair.value_.get_str();
      }
      else if (pair.name_ == "global_step") {
        this->global_step = pair.value_.get_int64();
      }
      else if (pair.name_ == "max_training_score") {
        this->max_training_score = static_cast<float>(pair.value_.get_real());
      }
      else if (pair.name_ == "best_iteration") {
        this->best_iteration = pair.value_.get_int64();
      }
      else if (pair.name_ == "latest_iteration") {
        this->latest_iteration = pair.value_.get_int64();
      }
      else if (pair.name_ == "use_off_road_term") {
        this->use_off_road_term = pair.value_.get_bool();
      }
      else if (pair.name_ == "off_road_term_perc") {
        this->off_road_term_perc = static_cast<float>(pair.value_.get_real());
      }
      else if (pair.name_ == "beta_1") {
        this->beta_1 = static_cast<float>(pair.value_.get_real());
      }
      else if (pair.name_ == "beta_2") {
        this->beta_2 = static_cast<float>(pair.value_.get_real());
      }
      else if (pair.name_ == "render_speed_lines") {
        this->render_speed_lines = pair.value_.get_bool();
      }
      else if (pair.name_ == "use_new_stop_sign_detector") {
        this->use_new_stop_sign_detector = pair.value_.get_bool();
      }
      else if (pair.name_ == "use_positional_encoding") {
        this->use_positional_encoding = pair.value_.get_bool();
      }
      else if (pair.name_ == "use_ttc") {
        this->use_ttc = pair.value_.get_bool();
      }
      else if (pair.name_ == "ttc_resolution") {
        this->ttc_resolution = pair.value_.get_int();
      }
      else if (pair.name_ == "ttc_penalty_ticks") {
        this->ttc_penalty_ticks = pair.value_.get_int();
      }
      else if (pair.name_ == "render_yellow_time") {
        this->render_yellow_time = pair.value_.get_bool();
      }
      else if (pair.name_ == "use_single_reward") {
        this->use_single_reward = pair.value_.get_bool();
      }
      else if (pair.name_ == "use_rl_termination_hint") {
        this->use_rl_termination_hint = pair.value_.get_bool();
      }
      else if (pair.name_ == "render_shoulder") {
        this->render_shoulder = pair.value_.get_bool();
      }
      else if (pair.name_ == "use_shoulder_channel") {
        this->use_shoulder_channel = pair.value_.get_bool();
      }
      else if (pair.name_ == "use_survival_reward") {
        this->use_survival_reward = pair.value_.get_bool();
      }
      else if (pair.name_ == "survival_reward_magnitude") {
        this->survival_reward_magnitude = static_cast<float>(pair.value_.get_real());
      }
      else {
        std::cerr << "Unkown config parameter: " << pair.name_ << "\n";
      }
    }
  }
};

#endif //CARLA_CONFIG_H
