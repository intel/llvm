#ifndef FINDPRIMESSYCL_H
#define FINDPRIMESSYCL_H

#include <sycl/sycl.hpp>

#include <chrono>
#include <mutex>
#include <vector>

struct work {
  int id;

  size_t size;
  size_t niter;
  size_t nitems;

  std::vector<sycl::opencl::cl_short> VRI;

  size_t result;

  bool success{false};

  std::chrono::high_resolution_clock::time_point start, stop;
  float elapsed{0.}, wait{0.}, run{0.};

  unsigned long submit_time{0}, start_time{0}, end_time{0};

  std::mutex *queueLock{nullptr};

  sycl::queue *deviceQueue{nullptr};
};

float find_prime_s(work *w);

#endif
