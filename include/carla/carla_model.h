#ifndef CARLA_MODEL_H
#define CARLA_MODEL_H

#include <filesystem>
#include <string>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include <opencv2/core/hal/interface.h>
#include <boost/format.hpp>

#include <carla/carla_config.h>
#include <rl_utils.h>

using namespace std;
using namespace torch;
using namespace torch::indexing;
using namespace std::literals;

class AgentImpl final : public nn::Module {
  nn::Sequential cnn{nullptr};
  nn::Sequential linear{nullptr};
  nn::Sequential state_linear{nullptr};
  nn::Sequential value_head{nullptr};
  nn::Sequential policy_head{nullptr};
  nn::Sequential dist_mu{nullptr};  // Alpha of Beta distribution
  nn::Sequential dist_sigma{nullptr};  // Beta of Beta distribution
  Tensor action_space_high;
  Tensor action_space_low;
  GlobalConfig config_;
  int visualization_counter = 0;

public:
  explicit AgentImpl(const GlobalConfig& config, const int action_dim, const float action_high, const float action_low):
  config_(config)
  {
    int num_input_channels = config_.obs_num_channels;
    if (config_.use_positional_encoding)
    {
      num_input_channels += 2;
    }

    if (config_.image_encoder == "roach_ln") {
      cnn = nn::Sequential(_weights_init(nn::Conv2d(nn::Conv2dOptions(num_input_channels, 8, 5).stride(2))),  // -> [B, 8, 94, 94]
                           nn::LayerNorm(nn::LayerNormOptions({8, 94, 94})),
                           nn::ReLU(),
                           _weights_init(nn::Conv2d(nn::Conv2dOptions(8, 16, 5).stride(2))),  // -> [B, 16, 45, 45]
                           nn::LayerNorm(nn::LayerNormOptions({16, 45, 45})),
                           nn::ReLU(),
                           _weights_init(nn::Conv2d(nn::Conv2dOptions(16, 32, 5).stride(2))),  // -> [B, 32, 21, 21]
                           nn::LayerNorm(nn::LayerNormOptions({32, 21, 21})),
                           nn::ReLU(),
                           _weights_init(nn::Conv2d(nn::Conv2dOptions(32, 64, 3).stride(2))),  // -> [B, 64, 10, 10]
                           nn::LayerNorm(nn::LayerNormOptions({64, 10, 10})),
                           nn::ReLU(),
                           _weights_init(nn::Conv2d(nn::Conv2dOptions(64, 128, 3).stride(2))),  // -> [B, 128, 4, 4]
                           nn::LayerNorm(nn::LayerNormOptions({128, 4, 4})),
                           nn::ReLU(),
                           _weights_init(nn::Conv2d(nn::Conv2dOptions(128, 256, 3).stride(1))),  // -> [B, 256, 2, 2]
                           nn::LayerNorm(nn::LayerNormOptions({256, 2, 2})),
                           nn::ReLU()
                          );
    }
    else if (config_.image_encoder == "roach") {
      cnn = nn::Sequential(_weights_init(nn::Conv2d(nn::Conv2dOptions(num_input_channels, 8, 5).stride(2))),  // -> [B, 8, 94, 94]
                           nn::ReLU(),
                           _weights_init(nn::Conv2d(nn::Conv2dOptions(8, 16, 5).stride(2))),  // -> [B, 16, 45, 45]
                           nn::ReLU(),
                           _weights_init(nn::Conv2d(nn::Conv2dOptions(16, 32, 5).stride(2))),  // -> [B, 32, 21, 21]
                           nn::ReLU(),
                           _weights_init(nn::Conv2d(nn::Conv2dOptions(32, 64, 3).stride(2))),  // -> [B, 64, 10, 10]
                           nn::ReLU(),
                           _weights_init(nn::Conv2d(nn::Conv2dOptions(64, 128, 3).stride(2))),  // -> [B, 128, 4, 4]
                           nn::ReLU(),
                           _weights_init(nn::Conv2d(nn::Conv2dOptions(128, 256, 3).stride(1))),  // -> [B, 256, 2, 2]
                           nn::ReLU()
                          );
    }
    else if (config_.image_encoder == "roach_ln2") {
      cnn = nn::Sequential(_weights_init(nn::Conv2d(nn::Conv2dOptions(num_input_channels, 8, 5).stride(2))),  // -> [B, 8, 126, 126]
                            nn::LayerNorm(nn::LayerNormOptions({8, 126, 126})),
                            nn::ReLU(),
                            _weights_init(nn::Conv2d(nn::Conv2dOptions(8, 16, 5).stride(2))),  // -> [B, 16, 61, 61]
                            nn::LayerNorm(nn::LayerNormOptions({16, 61, 61})),
                            nn::ReLU(),
                            _weights_init(nn::Conv2d(nn::Conv2dOptions(16, 24, 5).stride(2))),  // -> [B, 16, 29, 29]
                            nn::LayerNorm(nn::LayerNormOptions({24, 29, 29})),
                            nn::ReLU(),
                            _weights_init(nn::Conv2d(nn::Conv2dOptions(24, 32, 5).stride(2))),  // -> [B, 32, 13, 13]
                            nn::LayerNorm(nn::LayerNormOptions({32, 13, 13})),
                            nn::ReLU(),
                            _weights_init(nn::Conv2d(nn::Conv2dOptions(32, 64, 3).stride(2))),  // -> [B, 64, 6, 6]
                            nn::LayerNorm(nn::LayerNormOptions({64, 6, 6})),
                            nn::ReLU(),
                            _weights_init(nn::Conv2d(nn::Conv2dOptions(64, 128, 3).stride(1))),  // -> [B, 128, 4, 4]
                            nn::LayerNorm(nn::LayerNormOptions({128, 4, 4})),
                            nn::ReLU(),
                            _weights_init(nn::Conv2d(nn::Conv2dOptions(128, 256, 3).stride(1))),  // -> [B, 256, 2, 2]
                            nn::LayerNorm(nn::LayerNormOptions({256, 2, 2})),
                            nn::ReLU()
                           );
    }
    else {
      throw runtime_error("Unsupported image_encoder chosen. Options roach, roach_ln. Chosen option: " + config_.image_encoder);
    }

    register_module("cnn", cnn);

    constexpr int n_flatten = 256 * 2 * 2;  // Number of channels output from the CNN
    constexpr int n_features = 256;  // Number of features after merging cnn features with measurement features.

    if (config_.use_layer_norm) {
      linear = nn::Sequential(nn::Linear(n_flatten + 256, 512),
                              nn::LayerNorm(nn::LayerNormOptions({512})),
                              nn::ReLU(),
                              nn::Linear(512, 256),
                              nn::LayerNorm(nn::LayerNormOptions({256})),
                              nn::ReLU());
    }
    else {
      linear = nn::Sequential(nn::Linear(n_flatten + 256, 512),
                        nn::ReLU(),
                        nn::Linear(512, 256),
                        nn::ReLU());
    }

    register_module("linear", linear);

    if (config_.use_layer_norm) {
      state_linear = nn::Sequential(
          nn::Linear(config_.obs_num_measurements, 256),
          nn::LayerNorm(nn::LayerNormOptions({256})),
          nn::ReLU(),
          nn::Linear(256, 256),
          nn::LayerNorm(nn::LayerNormOptions({256})),
          nn::ReLU()
      );
    }
    else {
      state_linear = nn::Sequential(nn::Linear(config_.obs_num_measurements, 256),
                                    nn::ReLU(),
                                    nn::Linear(256, 256),
                                    nn::ReLU());
    }
    register_module("state_linear", state_linear);

    if (config_.use_layer_norm) {
      value_head = nn::Sequential(nn::Linear(n_features + config_.num_value_measurements, 256),
                                  nn::LayerNorm(nn::LayerNormOptions({256})),
                                  nn::ReLU(),
                                  nn::Linear(256, 256),
                                  nn::LayerNorm(nn::LayerNormOptions({256})),
                                  nn::ReLU(),
                                  nn::Linear(256, 1));
    }
    else {
      value_head = nn::Sequential(nn::Linear(n_features + config_.num_value_measurements, 256),
                            nn::ReLU(),
                            nn::Linear(256, 256),
                            nn::ReLU(),
                            nn::Linear(256, 1));
    }
    register_module("value_head", value_head);

    if (config_.use_layer_norm and config_.use_layer_norm_policy_head) {
      policy_head = nn::Sequential(
          nn::Linear(n_features, 256),
          nn::LayerNorm(nn::LayerNormOptions({256})),
          nn::ReLU(),
          nn::Linear(256, n_features),
          nn::LayerNorm(nn::LayerNormOptions({256})),
          nn::ReLU()
      );
    }
    else {
      policy_head = nn::Sequential(nn::Linear(n_features, 256),
                                   nn::ReLU(),
                                   nn::Linear(256, n_features),
                                   nn::ReLU());
    }
    register_module("policy_head", policy_head);

    dist_mu = nn::Sequential(
      nn::Linear(n_features, action_dim)
    );
    register_module("dist_mu", dist_mu);

    dist_sigma = nn::Sequential(
      nn::Linear(n_features, action_dim)
    );
    register_module("dist_sigma", dist_sigma);

    action_space_high = register_parameter("action_space_high"s, torch::tensor(action_high), false);
    action_space_low = register_parameter("action_space_low"s, torch::tensor(action_low), false);

    if (config_.debug) {
      if (config_.debug_type == "render") {
        cv::startWindowThread();
        cv::namedWindow("Observation");
      }
      else {
        filesystem::create_directories("./visu");
      }
    }
  }

  ~AgentImpl() override {
    if (config_.debug) {
      cv::destroyAllWindows();
    }
  }

  Tensor normalize_bev(const Tensor& bev_semantics) {
    return bev_semantics / 255.0f;
  }

  Tensor unnormalize_bev(const Tensor& bev_semantics) {
    return bev_semantics * 255.0f;
  }

  Tensor forward_cnn_encoder(const Tensor& bev_semantics, const Tensor& measurements) {
    Tensor birdview = bev_semantics.to(torch::kFloat32);
    birdview = normalize_bev(bev_semantics);

    if (config_.use_positional_encoding) {
      Tensor x = torch::linspace(-1.0f, 1.0f, config_.bev_semantics_height, bev_semantics.device());
      Tensor y = torch::linspace(-1.0f, 1.0f, config_.bev_semantics_width, bev_semantics.device());
      vector<Tensor> grids = torch::meshgrid({x, y}, "ij");
      grids[0] = grids[0].unsqueeze(0).unsqueeze(0).expand({bev_semantics.sizes()[0], -1, -1, -1});
      grids[1] = grids[1].unsqueeze(0).unsqueeze(0).expand({bev_semantics.sizes()[0], -1, -1, -1});

      birdview = torch::concatenate({bev_semantics, grids[0], grids[1]}, 1);
    }

    Tensor x = cnn->forward(birdview);
    x = flatten(x, 1);
    Tensor latent_state = state_linear->forward(measurements);
    x = cat({x, latent_state}, 1);
    x = linear->forward(x);
    return x;
  }

  Tensor get_value(const Tensor& bev_semantics, const Tensor& measurements, const Tensor& value_measurements) {
    Tensor features = forward_cnn_encoder(bev_semantics, measurements);
    Tensor value_features = cat({features, value_measurements}, 1);
    Tensor values = value_head->forward(value_features);
    return values;
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

  tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> forward(const Tensor& bev_semantics, const Tensor& measurements,
    const Tensor& value_measurements, Tensor actions=at::empty({0}), const string sample_type="sample"s,
    const optional<Generator>& generator=nullopt)
  {
    Tensor features = forward_cnn_encoder(bev_semantics, measurements);

    Tensor value_features = cat({features, value_measurements}, 1);
    Tensor values = value_head->forward(value_features);

    Tensor latent_pi = policy_head->forward(features);
    Tensor mu = dist_mu->forward(latent_pi);
    Tensor sigma = dist_sigma->forward(latent_pi);

    mu = nn::functional::softplus(mu) + config_.beta_min_a_b_value;
    sigma = nn::functional::softplus(sigma) + config_.beta_min_a_b_value;
    const auto proba_distribution = Beta(mu, sigma);

    if (actions.data_ptr() == nullptr) {
      if (sample_type == "sample"s) {
        actions = proba_distribution.sample(generator);
      }
      else if (sample_type == "mean"s) {
        actions = proba_distribution.mean();
      }
      else if (sample_type == "roach"s) {
        actions = proba_distribution.roach_deterministic();
      }
      else {
        throw runtime_error("Unsupported sample type used. Sample type: "s + sample_type);
      }
    }
    else {
      actions = scale_action(actions);
    }

    Tensor log_prob = proba_distribution.log_prob(actions).sum(1);
    actions = unscale_action(actions);

    Tensor entropy = proba_distribution.entropy().sum(1);

    if (config_.debug)
    {
      // We only visualize the first element in the batch when visualizing during training.
      const auto proba_distribution_visu = Beta(mu.slice(0, 0, 1), sigma.slice(0, 0, 1));
      visualize_model(proba_distribution_visu, bev_semantics, measurements, value_measurements, actions, values);
    }

    return make_tuple(actions, log_prob, entropy, values, mu, sigma);
  }

  Tensor convert_action_to_control(const Tensor& action) {
    // Convert acceleration to brake / throttle. Acc in [-1,1]. Negative acceleration -> brake
    Tensor control = torch::zeros(3);
    control.index({0}) = action.index({0});

    if (action.accessor<float, 1>()[1] > 0.0f) {
      control.index({1}) = action.index({1});
      control.accessor<float, 1>()[2] = 0.0f;
    }
    else {
      control.accessor<float, 1>()[1] = 0.0f;
      control.index({2}) = -action.index({1});
    }
    return control;
  }

  void visualize_model(const Distribution& distribution, const Tensor& observation, const Tensor& measurement,
    const Tensor& value_measurement, const Tensor& actions, const Tensor& values) {
    NoGradGuard no_grad;
    const auto height = static_cast<int>(observation.sizes()[2]);
    const auto width = static_cast<int>(observation.sizes()[3]);
    const Tensor local_measurement = measurement.index({0}).to(kCPU, false, true);
    const Tensor local_value_measurement = value_measurement.index({0}).to(kCPU, false, true);
    const Tensor local_actions_scaled = scale_action(actions).index({0}).to(kCPU, false, true);
    const Tensor local_value = values.index({0}).to(kCPU, false, true);

    // Render action distributions
    Device device = observation.device();
    Tensor granularity, granularity_cpu;
    if (config_.distribution == "beta") {
      granularity = torch::arange(0.0f, 1.0f, 0.001f).unsqueeze(1);
      granularity = torch::ones({granularity.sizes()[0], 2}) * granularity;
      granularity = granularity.to(device, false, false);
      granularity_cpu = granularity.to(kCPU, false, true);
    }
    else {
      throw runtime_error("Unsupported distribution in visualization.");
    }

    Tensor local_distribution = distribution.log_prob(granularity);
    local_distribution = torch::exp(local_distribution).cpu();

    array<string, 2> action_type {"steering", "acceleration"};
    constexpr int upscale_factor = 4;
    const int plot_height = static_cast<int>(lround(height / (2 + 1)));
    vector<cv::Mat> action_plots_v;
    for (int i = 0; i < 2; ++i) {
      cv::Mat action_plot = cv::Mat::zeros(plot_height, width, CV_8UC3);
      cv::line(action_plot, cv::Point(width / 2, 0), cv::Point(width / 2, (plot_height - 1)), cv::Scalar(0, 255, 0), 2);
      cv::line(action_plot, cv::Point(0, 0), cv::Point(0, (plot_height - 1)), cv::Scalar(0, 255, 0), 2);
      cv::line(action_plot, cv::Point(width - 1, 0), cv::Point(width - 1, (plot_height - 1)), cv::Scalar(0, 255, 0), 2);

      // plot chosen action.
      const int control_pixel = static_cast<int>(local_actions_scaled.accessor<float,1>()[i] * (width - 1.0f));
      cv::line(action_plot, cv::Point(control_pixel, 0), cv::Point(control_pixel, (plot_height - 1)), cv::Scalar(255, 255, 0), 2);

      auto granularity_cpu_accessor = granularity_cpu.accessor<float, 2>();
      auto local_distribution_accessor = local_distribution.accessor<float, 2>();
      const int num_points = granularity_cpu.sizes()[0];
      for (int j = 0; j < num_points; ++j) {
        int x =  static_cast<int>(granularity_cpu_accessor[j][i] * static_cast<float>(width));
        constexpr float y_max = 25.0f; // Continuous PDFs can be arbitrary high. We clipp after 25.
        int y_pixel = static_cast<int>(local_distribution_accessor[j][i] / y_max * (plot_height - 1.0f));
        int clipped_pixel = std::min(plot_height - 1, y_pixel);
        int y = (plot_height - 1) - clipped_pixel;  // Mirror
        cv::circle(action_plot, cv::Point(x, y), 1, cv::Scalar(255, 255, 0), -1, cv::LINE_AA);
      }
      cv::putText(action_plot, action_type[i], cv::Point(0, 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
      action_plots_v.push_back(action_plot);
    }
    cv::Mat action_plots;
    cv::vconcat(action_plots_v.at(0), action_plots_v.at(1), action_plots);

    // Render measrements
    int measure_plot_height = height - action_plots.rows;
    cv::Mat measurement_plot = cv::Mat::zeros(measure_plot_height, width, CV_8UC3);

    cv::putText(measurement_plot, (boost::format("Last steer: %.2f") % local_measurement.accessor<float, 1>()[0]).str(), cv::Point(0, 10), cv::FONT_HERSHEY_SIMPLEX, 0.33, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(measurement_plot, (boost::format("Last throt: %.2f") % local_measurement.accessor<float, 1>()[1]).str(), cv::Point(0, 25), cv::FONT_HERSHEY_SIMPLEX, 0.33, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(measurement_plot, (boost::format("Last break: %.2f") % local_measurement.accessor<float, 1>()[2]).str(), cv::Point(0, 40), cv::FONT_HERSHEY_SIMPLEX, 0.33, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    if (config_.use_target_point) {
      cv::putText(measurement_plot, (boost::format("TP: %.1f %.1f") % local_measurement.accessor<float, 1>()[8] % local_measurement.accessor<float, 1>()[9]).str(), cv::Point(0, 55), cv::FONT_HERSHEY_SIMPLEX, 0.33, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    cv::putText(measurement_plot, (boost::format("Gear: %.2f") % local_measurement.accessor<float, 1>()[3]).str(), cv::Point(width / 2, 10), cv::FONT_HERSHEY_SIMPLEX, 0.33, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(measurement_plot, (boost::format("Speed: %.1f %.1f") % local_measurement.accessor<float, 1>()[4] % local_measurement.accessor<float, 1>()[5]).str(), cv::Point(width / 2, 25), cv::FONT_HERSHEY_SIMPLEX, 0.33, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(measurement_plot, (boost::format("F. speed: %.2f") % local_measurement.accessor<float, 1>()[6]).str(), cv::Point(width / 2, 40), cv::FONT_HERSHEY_SIMPLEX, 0.33, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    cv::putText(measurement_plot, (boost::format("Speed lim.: %.2f") % local_measurement.accessor<float, 1>()[7]).str(), cv::Point(width / 2, 55), cv::FONT_HERSHEY_SIMPLEX, 0.33, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    cv::vconcat(measurement_plot, action_plots, action_plots);

    // Render bev observation.
    const Tensor COLOR_BLACK       = torch::tensor({  0,   0,   0}, kUInt8);
    const Tensor COLOR_RED         = torch::tensor({255,   0,   0}, kUInt8);
    const Tensor COLOR_GREEN       = torch::tensor({  0, 255,   0}, kUInt8);
    const Tensor COLOR_BLUE        = torch::tensor({{  0,   0, 255}}, kUInt8);
    const Tensor COLOR_CYAN        = torch::tensor({{  0, 255, 255}}, kUInt8);
    const Tensor COLOR_MAGENTA     = torch::tensor({{255,   0, 255}}, kUInt8);
    const Tensor COLOR_MAGENTA_2   = torch::tensor({{255, 140, 255}}, kUInt8);
    const Tensor COLOR_YELLOW      = torch::tensor({{255, 255,   0}}, kUInt8);
    const Tensor COLOR_YELLOW_2    = torch::tensor({{160, 160,   0}}, kUInt8);
    const Tensor COLOR_WHITE       = torch::tensor({{255, 255, 255}}, kUInt8);
    const Tensor COLOR_GREY        = torch::tensor({{128, 128, 128}}, kUInt8);
    const Tensor COLOR_ALUMINIUM_0 = torch::tensor({{238, 238, 236}}, kUInt8);
    const Tensor COLOR_ALUMINIUM_3 = torch::tensor({{136, 138, 133}}, kUInt8);
    const Tensor COLOR_ALUMINIUM_5 = torch::tensor({{ 46,  52,  54}}, kUInt8);

    Tensor local_observation = observation.index({0}).to(kCPU, false, true);

    Tensor rendered_bev = torch::zeros({width, height, 3}, kUInt8);
    Tensor road_mask = local_observation.index({0}).to(kBool, false, false);
    Tensor route_mask = local_observation.index({1}).to(kBool, false, false);
    Tensor lane_mask = local_observation.index({2}).to(kUInt8, false, false) == 255;
    Tensor lane_mask_broken = local_observation.index({2}).to(kUInt8, false, false) == 127;
    Tensor vehicle_mask = local_observation.index({3}).to(kBool, false, false);
    Tensor walker_mask = local_observation.index({4}).to(kBool, false, false);
    Tensor tl_mask_green = local_observation.index({5}).to(kUInt8, false, false) == 80;
    Tensor tl_mask_yellow = local_observation.index({5}).to(kUInt8, false, false) == 170;
    Tensor tl_mask_red = local_observation.index({5}).to(kUInt8, false, false) == 255;
    Tensor stop_mask = local_observation.index({6}).to(kBool, false, false);

    Tensor speed_mask;
    if (config_.obs_num_channels > 7) {
      speed_mask = local_observation.index({7}).to(kBool, false, false);
    }
    Tensor static_mask;
    if (config_.obs_num_channels > 8) {
      static_mask = local_observation.index({8}).to(kBool, false, false);
    }

    vector<Tensor> past_vehicle_mask;
    vector<Tensor> past_vehicle;
    vector<Tensor> past_walker_mask;
    vector<Tensor> past_walker;
    if (config_.obs_num_channels > 14 and config_.use_history) {
      past_vehicle_mask.push_back(local_observation.index({9}).to(kBool, false, false));
      past_vehicle.push_back(local_observation.index({9}));
      past_vehicle_mask.push_back(local_observation.index({10}).to(kBool, false, false));
      past_vehicle.push_back(local_observation.index({10}));
      past_vehicle_mask.push_back(local_observation.index({11}).to(kBool, false, false));
      past_vehicle.push_back(local_observation.index({11}));

      past_walker_mask.push_back(local_observation.index({12}).to(kBool, false, false));
      past_walker.push_back(local_observation.index({12}));
      past_walker_mask.push_back(local_observation.index({13}).to(kBool, false, false));
      past_walker.push_back(local_observation.index({13}));
      past_walker_mask.push_back(local_observation.index({14}).to(kBool, false, false));
      past_walker.push_back(local_observation.index({14}));
    }

    rendered_bev.index_put_({road_mask}, COLOR_ALUMINIUM_5);
    rendered_bev.index_put_({route_mask}, COLOR_ALUMINIUM_3);
    rendered_bev.index_put_({lane_mask}, COLOR_MAGENTA);
    rendered_bev.index_put_({lane_mask_broken}, COLOR_MAGENTA_2);

    rendered_bev.index_put_({stop_mask}, COLOR_YELLOW_2);
    if (config_.render_green_tl) {
      rendered_bev.index_put_({tl_mask_green}, COLOR_GREEN);
    }
    rendered_bev.index_put_({tl_mask_yellow}, COLOR_YELLOW);
    rendered_bev.index_put_({tl_mask_red}, COLOR_RED);

    if (config_.obs_num_channels > 8) {
      rendered_bev.index_put_({static_mask}, COLOR_ALUMINIUM_0);
    }

    for (int i = 0; i < past_vehicle_mask.size(); ++i) {
      Tensor vehicle_float = (past_vehicle.at(i) / 255.0f).unsqueeze(2);
      float factor = (past_vehicle_mask.size() + 1 - i) * 0.2f;
      Tensor past_color = torch::clamp(COLOR_BLUE + (255 - COLOR_BLUE) * factor, 0, 255);
      Tensor vehicle_colors = vehicle_float.index({past_vehicle_mask.at(i)}) * past_color;
      rendered_bev.index_put_({past_vehicle_mask.at(i)}, vehicle_colors.to(kUInt8, false, false));

      Tensor walker_float = (past_walker.at(i) / 255.0f).unsqueeze(2);
      factor = (past_walker_mask.size() + 1 - i) * 0.2f;
      past_color = torch::clamp(COLOR_CYAN + (255 - COLOR_CYAN) * factor, 0, 255);
      Tensor walker_colors = walker_float.index({past_walker_mask.at(i)}) * past_color;
      rendered_bev.index_put_({past_walker_mask.at(i)}, walker_colors.to(kUInt8, false, false));

    }

    Tensor vehicle_float = (local_observation.index({3}) / 255.0f).unsqueeze(2);
    Tensor vehicle_colors = vehicle_float.index({vehicle_mask}) * COLOR_BLUE;
    rendered_bev.index_put_({vehicle_mask}, vehicle_colors.to(kUInt8, false, false));

    Tensor walker_float = (local_observation.index({4}) / 255.0f).unsqueeze(2);
    Tensor walker_colors = walker_float.index({walker_mask}) * COLOR_CYAN;
    rendered_bev.index_put_({walker_mask}, walker_colors.to(kUInt8, false, false));

    if (config_.obs_num_channels > 7) {
      Tensor speed_float = (local_observation.index({7}) / 255.0f).unsqueeze(2);
      Tensor speed_colors = speed_float.index({speed_mask}) * COLOR_GREY;
      rendered_bev.index_put_({speed_mask}, speed_colors.to(kUInt8, false, false));
    }
    rendered_bev = rendered_bev.contiguous();
    cv::Mat obs_rendered = cv::Mat(cv::Size(width, height), CV_8UC3, rendered_bev.data_ptr<uchar>());

    Tensor control = convert_action_to_control(actions.index({0}).to(kCPU, false, true));
    cv::putText(obs_rendered, (boost::format("Steer: %.2f") % control.accessor<float, 1>()[0]).str(), cv::Point(5, 10), cv::FONT_HERSHEY_SIMPLEX, 0.33, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(obs_rendered, (boost::format("Throt: %.2f") % control.accessor<float, 1>()[1]).str(), cv::Point(5, 25), cv::FONT_HERSHEY_SIMPLEX, 0.33, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(obs_rendered, (boost::format("Brake: %.2f") % control.accessor<float, 1>()[2]).str(), cv::Point(5, 40), cv::FONT_HERSHEY_SIMPLEX, 0.33, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(obs_rendered, (boost::format("Value: %.2f") % local_value.accessor<float, 1>()[0]).str(), cv::Point(5, 55), cv::FONT_HERSHEY_SIMPLEX, 0.33, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    cv::putText(obs_rendered, (boost::format("timeout: %.2f") % local_value_measurement.accessor<float, 1>()[0]).str(), cv::Point(110, 10), cv::FONT_HERSHEY_SIMPLEX, 0.33, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(obs_rendered, (boost::format("blocked: %.2f") % local_value_measurement.accessor<float, 1>()[1]).str(), cv::Point(110, 25), cv::FONT_HERSHEY_SIMPLEX, 0.33, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(obs_rendered, (boost::format("route: %.2f") % local_value_measurement.accessor<float, 1>()[2]).str(), cv::Point(110, 40), cv::FONT_HERSHEY_SIMPLEX, 0.33, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    if (config_.use_extra_control_inputs) {
      cv::putText(obs_rendered, (boost::format("wheel: %.2f") % local_measurement.accessor<float, 1>()[8]).str(), cv::Point(110, 140), cv::FONT_HERSHEY_SIMPLEX, 0.33, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
      cv::putText(obs_rendered, (boost::format("error: %.2f") % local_measurement.accessor<float, 1>()[9]).str(), cv::Point(110, 155), cv::FONT_HERSHEY_SIMPLEX, 0.33, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
      cv::putText(obs_rendered, (boost::format("deriv: %.2f") % local_measurement.accessor<float, 1>()[10]).str(), cv::Point(110, 170), cv::FONT_HERSHEY_SIMPLEX, 0.33, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
      cv::putText(obs_rendered, (boost::format("integ: %.2f") % local_measurement.accessor<float, 1>()[11]).str(), cv::Point(110, 185), cv::FONT_HERSHEY_SIMPLEX, 0.33, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    cv::Mat final_plot;
    cv::hconcat(action_plots, obs_rendered, final_plot);
    cv::cvtColor(final_plot, final_plot, cv::COLOR_BGR2RGB);

    if (config_.debug_type == "render") {
      cv::Mat upscaled;
      double scale_factor = 4.0;

      // Method 1: Using direct scale factor
      cv::resize(final_plot, upscaled, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);
      cv::imshow("Observation", upscaled);
    }
    else {
      cv::imwrite((boost::format("./visu/image%07d.png") % visualization_counter).str(), final_plot);
    }
    visualization_counter++;
  }

protected:
  static nn::Conv2d _weights_init(nn::Conv2d m, const float bias_const=0.1) {
    NoGradGuard noGrad;
    nn::init::xavier_uniform_(m->weight, nn::init::calculate_gain(kReLU));
    nn::init::constant_(m->bias, bias_const);
    return m;
  }
};


TORCH_MODULE(Agent);

#endif //CARLA_MODEL_H
