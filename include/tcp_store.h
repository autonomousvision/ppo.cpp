//
// Created by jaeger on 18.11.24.
//

#ifndef TCP_STORE_H
#define TCP_STORE_H

#include <iostream>
#include <string>
#include <stdexcept>
#include <atomic>
#include <thread>

#include <zmq.hpp>

class TCPStoreServer {
    int num_done_;
    std::string rdvz_addr_;
    int port_;
    int max_connections_;
    std::thread server_thread_;
    zmq::context_t context_;
    zmq::socket_t socket_0_;
    zmq::socket_t socket_1_;

    const char empty_ = ' ';
public:
    std::atomic_bool running_ = false;

    TCPStoreServer(const std::string& rdvz_addr, const int port, const int max_connections) :
    num_done_(0), rdvz_addr_(rdvz_addr), port_(port), max_connections_(max_connections) {}

    void start() {
        // Initialize ZeroMQ context and socket
        socket_0_ = zmq::socket_t(context_, zmq::socket_type::rep);  // Used to receive change requests
        socket_1_ = zmq::socket_t(context_, zmq::socket_type::pub);  // Used to publish current state.
        socket_0_.bind("tcp://"s + "*:"s + std::to_string(port_));
        socket_1_.bind("tcp://"s + rdvz_addr_ + ":"s + std::to_string(port_ + 1));


        std::cout << "Server started, waiting for requests..." << std::endl;
        server_thread_ = std::thread([&] {
            try {
                running_ = true;
                // Loop to process incoming requests
                while (running_) {
                    zmq::message_t request;
                    const auto result = socket_0_.recv(request, zmq::recv_flags::none);

                    if (!result) {
                        std::cerr << "Invalid message receiveced: " << result << std::endl;
                        throw std::runtime_error("Invalid message receiveced: ");
                    }

                    const auto command = static_cast<char*>(request.data());

                    if (*command == 'i') {
                        // Add 1 to the current value
                        num_done_ += 1;

                        zmq::message_t reply(&empty_, sizeof(char));

                        socket_0_.send(reply, zmq::send_flags::none);

                    } else if (*command == 'r') {
                        num_done_ = 0;

                        zmq::message_t reply(&empty_, sizeof(char));
                        socket_0_.send(reply, zmq::send_flags::none);

                    } else {
                        std::cerr << "Invalid command: " << command << std::endl;
                        throw std::runtime_error("Invalid command: "s + command);
                    }
                    zmq::message_t reply(&num_done_, sizeof(num_done_));
                    socket_1_.send(reply, zmq::send_flags::none);  // Publish new value to all workers
                }
            }
            catch (const zmq::error_t& e) {
                if (not running_ && e.num() == ETERM) {
                    socket_0_.close();
                    socket_1_.close();
                    std::cout << "TCP socket closed, exiting thread..." << std::endl;
                    return;
                }
                std::cerr << "ZeroMQ error: " << e.what() << std::endl;
            }
        });
    }

    ~TCPStoreServer() {
        if (running_) {
            running_ = false;
            context_.close();
            server_thread_.join();
        }
    }
};

class TCPStoreClient {
    std::string rdvz_addr_;
    int port_;
    int num_done_ = 0;
    zmq::context_t context_;
    zmq::socket_t socket_0_;
    zmq::socket_t socket_1_;
public:
    TCPStoreClient(const std::string& rdvz_addr, const int port) : rdvz_addr_(rdvz_addr), port_(port)
    {
        socket_0_ = zmq::socket_t(context_, zmq::socket_type::req);
        socket_1_ = zmq::socket_t(context_, zmq::socket_type::sub);
        socket_1_.set(zmq::sockopt::conflate, 1);

        socket_0_.connect("tcp://"s + rdvz_addr_ + ":"s + std::to_string(port_));
        socket_1_.connect("tcp://"s + rdvz_addr_ + ":"s + std::to_string(port_ + 1));
        socket_1_.set(zmq::sockopt::subscribe, "");  // Subscribe to all topics
    }

    void increment() {
        constexpr char add_identifier = 'i';
        zmq::message_t message(&add_identifier, sizeof(char));
        socket_0_.send(message, zmq::send_flags::none);

        zmq::message_t reply;
        const auto result = socket_0_.recv(reply, zmq::recv_flags::none);

        if (!result) {
            std::cerr << "Invalid message receiveced: " << result << std::endl;
            throw std::runtime_error("Invalid message receiveced: ");
        }
    }

    void reset() {
        constexpr char add_identifier = 'r';
        zmq::message_t message(&add_identifier, sizeof(char));
        socket_0_.send(message, zmq::send_flags::none);

        zmq::message_t reply;
        const auto result = socket_0_.recv(reply, zmq::recv_flags::none);

        if (!result) {
            std::cerr << "Invalid message receiveced: " << result << std::endl;
            throw std::runtime_error("Invalid message receiveced: ");
        }
    }

    int get() {
        zmq::message_t reply;
        const auto result = socket_1_.recv(reply, zmq::recv_flags::dontwait);
        if (!result) {
            // zmq.hpp returns an empty message if the error code is EAGAIN, so we return the previous value.
            return num_done_;
        }
        const auto *p_reply = static_cast<int *>(reply.data());
        num_done_ = *p_reply;
        return num_done_;
    }
};

#endif //TCP_STORE_H
