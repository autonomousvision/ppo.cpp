//
// Created by jaeger on 10/25/25.
//

#ifndef PPO_CPP_DRIVE_ENV_H
#define PPO_CPP_DRIVE_ENV_H

#include <string>
#include <filesystem>
#include <cstdlib>
#include <cstring>
#include <chrono>

#include <gymcpp/gym.h>
#include <gymcpp/pufferdrive/drive.h>
#include <torch/torch.h>
#include <boost/format.hpp>

using namespace std;
using namespace torch;

// TODO clip actions

class PufferDriveEnv final {
protected:
    Tensor obs_;
    Tensor rewards_;
    Tensor terminations_;
    Tensor truncations_;
    const bool clip_actions_{true};

    int max_episode_steps_{1000};
    int elapsed_step_{max_episode_steps_ + 1};

    int report_interval_{1};
    int width_{1280};
    int height_{1024};
    int human_agent_idx_{0};
    float reward_vehicle_collision_{-0.1f};
    float reward_offroad_collision_{-0.1f};
    float reward_goal_{1.0f};
    float reward_goal_post_respawn_{0.5f};
    float reward_ade_{0.0f};
    float goal_radius_{2.0f};
    int resample_frequency_{91};
    int num_maps_{1}; // 100 normally. Only have 1 map for now.
    int available_maps_{1};
    int max_num_agents_{64}; // TODO decreased was 512 before for all envs
    string action_type_ = "continuous"s; // TODO Maybe include discrete
    int init_steps_{0};
    int _action_type_flag{1}; // 1 continuous, 0 discrete

    vector<int> map_ids_{0};
    int num_envs_{1};

    // To be initialized after envs are created
    vector<int> resized_agent_offsets_;
    vector<int> resized_map_ids_;
    int final_env_count_;

    Drive* env_;
    string render_mode_;

    // Set variables
    int max_controlled_agents_{-1};
    int init_mode_{0};  // 0: create_all_valid, 1: create_only_controlled
    int control_mode_{0};  // 0: control_vehicles, 1: control_agents, 2: control_tracks_to_predict, 3: control_sdc_only
    int goal_behavior_{0};

    // TODO load from config file instead of constant
    // Config variable
    int dynamics_model_ = CLASSIC;
    int scenario_length_{91};
    int collision_behavior_{0}; // Options: 0 - Ignore, 1 - Stop, 2 - Remove
    int offroad_behavior_{0}; // Options: 0 - Ignore, 1 - Stop, 2 - Remove
    float dt_{0.1};

    const filesystem::path map_folder_;
    std::mt19937 gen_;
    int seed_{1};

    // Logging
    vector<chrono::steady_clock::time_point> episode_start_times_;
    vector<float> episode_returns_;
    vector<int> episode_lengths_;

    // Note the 1 in the first dimension will be overwritten.
    vector<int> observation_space_ = vector<int>{1, 7 + (MAX_AGENTS - 1)*7 + MAX_ROAD_SEGMENT_OBSERVATIONS*7};
    static constexpr int action_space_{2};
    static constexpr float action_space_min_{-1.0};
    static constexpr float action_space_max_{1.0};
    static constexpr float observation_space_min_{-1.0};
    static constexpr float observation_space_max_{1.0};

public:

    explicit PufferDriveEnv(const filesystem::path& map_folder, const string& render_mode="rgb_array"s,
                            const int seed=1):
    map_folder_(map_folder), render_mode_(render_mode), seed_(seed)
    {
        gen_.seed(seed_);// Will be used to obtain a seed for the random number engine
        srand(seed_); // if pufferdrive internally uses rng I need to set this too

        filesystem::path binary_path = map_folder / "map_000.bin";
        if (!filesystem::exists(binary_path)) {
            throw std::runtime_error(
                "Required directory " + binary_path.string() +
                " not found. Please ensure the Drive maps are downloaded and installed correctly per docs."
            );
        }

        int available_maps = 0;

        // Count .bin files in the directory
        for (const auto& entry : filesystem::directory_iterator(map_folder)) {
            if (entry.is_regular_file() && entry.path().extension() == ".bin") {
                ++available_maps;
            }
        }

        // Check availability
        if (num_maps_ > available_maps) {
            throw std::invalid_argument(
                "num_maps (" + std::to_string(num_maps_) +
                ") exceeds available maps in directory (" + std::to_string(available_maps) +
                "). Please reduce num_maps or add more maps to resources/drive/binaries."
            );
        }
        available_maps_ = available_maps;

        env_init();
        obs_ = torch::zeros({observation_space_.at(0), observation_space_.at(1)});
        rewards_ = torch::zeros({observation_space_.at(0)});
        terminations_ = torch::zeros({observation_space_.at(0)});
        truncations_ = torch::zeros({observation_space_.at(0)});
    }

    ~PufferDriveEnv()
    {
        if (render_mode_ == "human" and env_->client != NULL) {
            close_client(env_->client);
        }
        free_allocated(env_);  // Also calls c_close()
        delete env_;
    }

    // TODO log statistics about return etc. and send info

    [[nodiscard]] vector<int> get_observation_space() const {
        return observation_space_;
    }
    [[nodiscard]] int get_action_space() const {
        return action_space_;
    }
    [[nodiscard]] float get_action_space_min() const {
        return action_space_min_;
    }
    [[nodiscard]] float get_action_space_max() const {
        return action_space_max_;
    }

    Tensor reset(const int seed) {
        // TODO autoreset of individual agents
        // Negative seed indicates we do not want to set the seed.
        if (seed > 0) {
            srand(seed);
            gen_.seed(seed);
        }
        // add_log(env_); // TODO check what this does
        c_reset(env_);
        elapsed_step_ = 0;

        writeState();

        episode_start_times_ = vector(env_->active_agent_count, chrono::steady_clock::now());
        episode_returns_ = vector(env_->active_agent_count, 0.0f);
        episode_lengths_ = vector(env_->active_agent_count, 0);

        return obs_;
    }

    tuple<Tensor, Tensor, Tensor, Tensor, vector<env_info>> step(const Tensor& action) {

        ++elapsed_step_;

        // TODO measure and memcopy raw memory instead
        Tensor actions_contiguous = action.contiguous().to(kFloat32);
        auto act_accessor = action.accessor<float,2>();
        for (int i = 0; i < observation_space_[0]; ++i)
        {
            for (int j = 0; j < action_space_; ++j)
            {
                env_->actions[i*action_space_ + j] = act_accessor[i][j];
            }
        }

        // env_->actions
        c_step(env_);
        if (render_mode_ == "human") { // NOTE will segfault right now unless num envs is == 1
            c_render(env_);
        }

        writeState();

        // TODO truncation
        vector<env_info> infos;
        auto rew_accessor = rewards_.accessor<float,1>();
        auto term_accessor = terminations_.accessor<float,1>();
        for (int i = 0; i < observation_space_[0]; ++i)
        {
            rew_accessor[i] = env_->rewards[i];
            term_accessor[i] = env_->terminals[i];

            episode_returns_[i] += env_->rewards[i];
            episode_lengths_[i] += 1;

            if (static_cast<int>(env_->terminals[i]) == 1)  // TODO or truncation
            {
                const std::chrono::duration<float> episode_time_length = chrono::steady_clock::now() - episode_start_times_[i];
                infos.push_back(env_info{episode_returns_[i], episode_lengths_[i], episode_time_length.count()});
                episode_lengths_[i] = 0;
                episode_returns_[i] = 0.0f;
            }
        }

        return make_tuple(obs_, rewards_, terminations_, truncations_, infos);
    }

private:
    int env_init()
    {
        env_ = new Drive();
        // Make sure the log is initialized to 0
        memset(&env_->log, 0, sizeof(Log));

        env_->human_agent_idx = human_agent_idx_;
        // TODO add to repo. Code doesn't seem to use it though.
        // env_->ini_file = "pufferlib/config/ocean/drive.ini";
        // env_init_config conf = {0};
        // the ini_parse function seems to be missing from github
        // if(ini_parse(env->ini_file, handler, &conf) < 0) {
        //     printf("Error while loading %s", env->ini_file);
        // }

        // TODO The value seems to be None in the code. Don't know what a sensible value is.
        // conf.scenario_length = (int)unpack(kwargs, "scenario_length");
        //
        // if (conf.scenario_length <= 0) {
        //     throw std::invalid_argument(
        //         "scenario_length must be > 0 but is of value: " + std::to_string(conf.scenario_length)
        //     );
        // }
        if (action_type_ == "discrete") {
            env_->action_type = 0;
        }
        else {
            env_->action_type = 1;
        }
        env_->dynamics_model = dynamics_model_;
        env_->reward_vehicle_collision = reward_vehicle_collision_;
        env_->reward_offroad_collision = reward_offroad_collision_;
        env_->reward_goal = reward_goal_;
        env_->reward_goal_post_respawn = reward_goal_post_respawn_;
        env_->reward_ade = reward_ade_;
        env_->scenario_length = scenario_length_;
        env_->collision_behavior = collision_behavior_;
        env_->offroad_behavior = offroad_behavior_;
        env_->max_controlled_agents = max_controlled_agents_;
        env_->dt = dt_;
        env_->init_mode = init_mode_;
        env_->control_mode = control_mode_;
        env_->goal_behavior = goal_behavior_;
        env_->goal_radius = goal_radius_;

        uniform_int_distribution<int> dist(0, available_maps_ - 1);
        int map_id = dist(gen_);
        string map_file_current = map_folder_.string() + (boost::format("/map_%03d.bin") % map_id).str();
        cout << "Loading map: " << map_file_current << endl;
        // env_->entities = load_map_binary(, env_);
        env_->map_name = new char[map_file_current.size() + 1];  // Note will call free on this in c_close()
        std::strcpy(env_->map_name, map_file_current.c_str());
        env_->num_agents = max_num_agents_;

        env_->map_name = strdup(map_file_current.c_str());
        env_->init_steps = init_steps_;
        env_->timestep = init_steps_;

        allocate(env_);  // Note I use the internal allocate and deallocate functions. also calls init
        observation_space_.at(0) = env_->active_agent_count;

        episode_start_times_ = vector(env_->active_agent_count, chrono::steady_clock::now());
        episode_returns_ = vector(env_->active_agent_count, 0.0f);
        episode_lengths_ = vector(env_->active_agent_count, 0);

        return 0;
    }

    void writeState() const
    {
        auto obs_accessor = obs_.accessor<float,2>();
        // TODO use memcopy and time.
        int index = 0;
        int single_obs_size = observation_space_[1];
        for (int i = 0; i < observation_space_[0]; ++i)
        {
            for (int j = 0; j < observation_space_[1]; ++j)
            {
                obs_accessor[i][j] = env_->observations[i*single_obs_size + j];
            }
        }
    }
};

#endif //PPO_CPP_DRIVE_ENV_H