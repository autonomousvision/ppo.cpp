#ifndef MUJOCO_ENV_H
#define MUJOCO_ENV_H

#include <string>
#include <iostream>

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#include <torch/torch.h>

using namespace std;

class MujocoEnv {
public:
    mjModel* model_;
    mjData*  data_;
    mjtNum*  init_qpos_;
    mjtNum*  init_qvel_;
    bool post_constraint_;
    string render_mode_ = "rgb_array"s;
    int frame_skip_;
    mjtNum dt_;
    constexpr static int DEFAULT_SIZE = 480;

    // OpenGL variables
    GLFWwindow* window;
    mjvCamera cam;                      // abstract camera
    mjvOption opt;                      // visualization options
    mjvScene scn;                       // abstract scene
    mjrContext con;                     // custom GPU context
    mjrRect viewport = {0, 0, 0, 0};

    MujocoEnv(const string& xml, const int frame_skip, const bool post_constraint, const string& render_mode="rgb_array"s):
        model_(mj_loadXML(xml.c_str(), NULL, NULL, 0)),
        init_qpos_(new mjtNum[model_->nq]),
        init_qvel_(new mjtNum[model_->nv]),
        post_constraint_(post_constraint),
        render_mode_(render_mode),
        frame_skip_(frame_skip)
    {
        dt_ = model_->opt.timestep * frame_skip_;
        model_->vis.global.offwidth = DEFAULT_SIZE;
        model_->vis.global.offheight = DEFAULT_SIZE;
        data_ = mj_makeData(model_);

        memcpy(init_qpos_, data_->qpos, sizeof(mjtNum) * model_->nq);
        memcpy(init_qvel_, data_->qvel, sizeof(mjtNum) * model_->nv);

        if (render_mode_ == "human"s) {
            // init GLFW
            if (!glfwInit()) {
                mju_error("Could not initialize GLFW");
            }

            // create window, make OpenGL context current, request v-sync
            window = glfwCreateWindow(640, 480, "Demo", nullptr, nullptr);
            glfwMakeContextCurrent(window);
            glfwSwapInterval(0);  // 1 = vsync, 0 = no vsync

            glfwSetWindowUserPointer(window, this);

            // get framebuffer viewport
            glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

            // initialize visualization data structures
            mjv_defaultCamera(&cam);
            // Camera type should be mjCAMERA_TRACKBALL
            cam.type = 0;
            mjv_defaultOption(&opt);
            mjv_defaultScene(&scn);
            mjr_defaultContext(&con);

            // create scene and context
            mjv_makeScene(model_, &scn, 2000);
            mjr_makeContext(model_, &con, mjFONTSCALE_150);

            // Release context since we might be switching to other threads later.
            glfwMakeContextCurrent(nullptr);
        }
    }

    virtual ~MujocoEnv()
    {
        if (render_mode_ == "human"s) {
            //free visualization storage
            mjv_freeScene(&scn);
            mjr_freeContext(&con);
            // terminate GLFW (crashes with Linux NVidia drivers)
            glfwTerminate();
        }
        mj_deleteData(data_);
        mj_deleteModel(model_);
        delete[] init_qpos_;
        delete[] init_qvel_;
    }

    void mujocoStep(const Tensor& action) {
        auto act_accessor = action.accessor<float,1>();
        for (int i = 0; i < model_->nu; ++i) {
            data_->ctrl[i] = static_cast<mjtNum>(act_accessor[i]);
        }
        for (int i = 0; i < frame_skip_; ++i) {
            mj_step(model_, data_);
        }

        if (post_constraint_) {
            mj_rnePostConstraint(model_, data_);
        }

        if (render_mode_ == "human"s) {
            glfwMakeContextCurrent(window);

            // Get target body position
            mjtNum target_pos[3];
            mju_copy3(target_pos, data_->qpos);

            // Update camera lookat point
            cam.lookat[0] = target_pos[0];
            cam.lookat[1] = target_pos[1];
            cam.lookat[2] = 0.5;

            cam.distance = 5.0;
            cam.azimuth = 90;    // Side view
            cam.elevation = 0;   // Horizontal
            glfwPostEmptyEvent();  // Thread-safe! Wakes up glfwWaitEvents()
            // update scene and render
            mjv_updateScene(model_, data_, &opt, nullptr, &cam, mjCAT_ALL, &scn);
            mjr_render(viewport, &scn, &con);

            // swap OpenGL buffers (blocking call due to v-sync)
            glfwSwapBuffers(window);
            // Seems to be important when doing multi-threaded rendering.
            glfwMakeContextCurrent(nullptr);
        }
    }
};

#endif //MUJOCO_ENV_H
