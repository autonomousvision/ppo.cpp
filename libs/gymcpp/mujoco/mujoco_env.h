#ifndef MUJOCO_ENV_H
#define MUJOCO_ENV_H

#include <string>
#include <iostream>

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

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

    // mouse interaction
    bool button_left = false;
    bool button_middle = false;
    bool button_right =  false;
    double lastx = 0;
    double lasty = 0;

    // mouse button callback
    void mouse_button(GLFWwindow* window, int button, int act, int mods) {
        // update button state
        button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
        button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
        button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

        // update mouse position
        glfwGetCursorPos(window, &lastx, &lasty);
    }

    // mouse move callback
    void mouse_move(GLFWwindow* window, const double xpos, const double ypos) {
        // no buttons down: nothing to do
        if (!button_left && !button_middle && !button_right) {
            return;
        }

        // compute mouse displacement, save
        const double dx = xpos - lastx;
        const double dy = ypos - lasty;
        lastx = xpos;
        lasty = ypos;

        // get current window size
        int width, height;
        glfwGetWindowSize(window, &width, &height);

        // get shift key state
        const bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                                glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

        // determine action based on mouse button
        mjtMouse action;
        if (button_right) {
            action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
        } else if (button_left) {
            action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
        } else {
            action = mjMOUSE_ZOOM;
        }

        // move camera
        mjv_moveCamera(model_, action, dx/height, dy/height, &scn, &cam);
    }


    // scroll callback
    void scroll(GLFWwindow* window, double xoffset, const double yoffset) {
        // emulate vertical mouse motion = 5% of window height
        mjv_moveCamera(model_, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
    }

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
            glfwSwapInterval(1);

            glfwSetWindowUserPointer(window, this);

            auto func = [](GLFWwindow* w, const int button, const int act, const int mods)
            {
                static_cast<MujocoEnv*>(glfwGetWindowUserPointer(w))->mouse_button(w, button, act, mods);
            };
            auto func2 = [](GLFWwindow* w, const double xoffset, const double yoffset)
            {
                static_cast<MujocoEnv*>(glfwGetWindowUserPointer(w))->scroll(w, xoffset, yoffset);
            };
            auto func3 = [](GLFWwindow* w, const double xpos, const double ypos)
            {
                static_cast<MujocoEnv*>(glfwGetWindowUserPointer(w))->mouse_move(w, xpos, ypos);
            };

            glfwSetMouseButtonCallback(window, func);
            glfwSetScrollCallback(window, func2);
            glfwSetCursorPosCallback(window, func3);

            // initialize visualization data structures
            mjv_defaultCamera(&cam);
            mjv_defaultOption(&opt);
            mjv_defaultScene(&scn);
            mjr_defaultContext(&con);

            // create scene and context
            mjv_makeScene(model_, &scn, 2000);
            mjr_makeContext(model_, &con, mjFONTSCALE_150);
        }
    }

    virtual ~MujocoEnv()
    {
        if (render_mode_ == "human"s) {
            //free visualization storage
            mjv_freeScene(&scn);
            mjr_freeContext(&con);
            // terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
            glfwTerminate();
#endif
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

            // get framebuffer viewport
            mjrRect viewport = {0, 0, 0, 0};
            glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

            // update scene and render
            mjv_updateScene(model_, data_, &opt, nullptr, &cam, mjCAT_ALL, &scn);
            mjr_render(viewport, &scn, &con);

            // swap OpenGL buffers (blocking call due to v-sync)
            glfwSwapBuffers(window);

            // process pending GUI events, call GLFW callbacks
            glfwPollEvents();
        }
    }
};

#endif //MUJOCO_ENV_H