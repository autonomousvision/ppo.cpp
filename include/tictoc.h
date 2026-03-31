// Convenience class to measure time without typing too much.

#ifndef TICTOC_H
#define TICTOC_H

#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>

class TicToc {
  std::chrono::high_resolution_clock::time_point start_time_;

public:
  void tic() {
    start_time_ = std::chrono::high_resolution_clock::now();
  }
  void toc(const std::string& msg="Elapsed time:", const bool restart=false) {
    const std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
    const double passed_sec = std::chrono::duration<double>(end_time - start_time_).count();

    // Set precision to microseconds, anything below that usually doesn't matter.
    std::cout << std::fixed << std::setprecision(6) << msg << " " << passed_sec << " seconds \n";

    if(restart) {
      start_time_ = std::chrono::high_resolution_clock::now();
    }
  }
  double tocvalue(const bool restart=false) {
    const std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
    const double passed_sec = std::chrono::duration<double>(end_time - start_time_).count();

    if(restart) {
      start_time_ = std::chrono::high_resolution_clock::now();
    }
    return passed_sec;
  }
};

#endif //TICTOC_H
