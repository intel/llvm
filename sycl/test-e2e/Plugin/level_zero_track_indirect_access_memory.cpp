// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %{build} %level_zero_options %threads_lib -o %t.out
// RUN: env SYCL_PI_LEVEL_ZERO_TRACK_INDIRECT_ACCESS_MEMORY=1 %{run} %t.out 2>&1 | FileCheck %s
//
// CHECK: pass
//
// Test checks memory tracking and deferred release functionality which is
// enabled by SYCL_PI_LEVEL_ZERO_TRACK_INDIRECT_ACCESS_MEMORY env variable.
// Tracking and deferred release is necessary for memory which can be indirectly
// accessed because such memory can't be released as soon as someone calls free.
// It can be released only after completion of all kernels which can possibly
// access this memory indirectly. Currently the Level Zero plugin marks all
// kernels with indirect access flag conservatively. This flag means that kernel
// starts to reference all existing memory allocations (even if not explicitly
// used in the kernel) as soon as it is submitted. That's why basically all
// memory allocations need to be tracked. This is crucial in multi-threaded
// applications because kernels with indirect access flag reference allocations
// from another threads causing the following error if memory is released too
// early:
//
// ../neo/opencl/source/os_interface/linux/drm_command_stream.inl
// Aborted (core dumped)
//
// Such multi-threaded scenario is checked in this test. Test is expected to
// pass when memory tracking is enabled and fail otherwise.

#include <cassert>
#include <iostream>
#include <thread>

#define LENGTH 10

#include <sycl/detail/core.hpp>
using namespace sycl;

void update_d2_data(queue &q) {
  int d2_data[LENGTH][LENGTH];

  try {
    size_t d_size = LENGTH;
    buffer<int, 2> b_d2_data((int *)d2_data, range<2>(d_size, d_size));

    q.submit([&](handler &cgh) {
      accessor acc{b_d2_data, cgh};

      cgh.parallel_for<class write_d2_data>(
          range<2>{d_size, d_size},
          [=](id<2> idx) { acc[idx] = idx[0] * idx[1]; });
    });
    q.wait_and_throw();
  } catch (exception &e) {
    std::cerr << std::string(e.what());
  }

  for (size_t i = 0; i < LENGTH; i++) {
    for (size_t j = 0; j < LENGTH; j++) {
      assert(d2_data[i][j] == i * j);
    }
  }
}
void update_d3_data(queue &q) {
  int d3_data[LENGTH][LENGTH][LENGTH];

  try {
    size_t d_size = LENGTH;
    buffer<int, 3> b_d3_data((int *)d3_data, range<3>(d_size, d_size, d_size));

    q.submit([&](handler &cgh) {
      accessor acc{b_d3_data, cgh};

      cgh.parallel_for<class write_d3_data>(
          range<3>{d_size, d_size, d_size},
          [=](id<3> idx) { acc[idx] = idx[0] * idx[1] * idx[2]; });
    });
    q.wait_and_throw();
  } catch (exception &e) {
    std::cerr << std::string(e.what());
  }

  for (size_t i = 0; i < LENGTH; i++) {
    for (size_t j = 0; j < LENGTH; j++) {
      for (size_t k = 0; k < LENGTH; k++) {
        assert(d3_data[i][j][k] == i * j * k);
      }
    }
  }
}

int main() {
  static const size_t n = 8;
  std::thread d2_threads[n];
  std::thread d3_threads[n];

  auto thread_body = [&](int type) {
    queue q;
    switch (type) {
    case 1:
      update_d2_data(q);
      break;
    case 2:
      update_d3_data(q);
      break;
    }
  };

  for (size_t i = 0; i < n; ++i) {
    d2_threads[i] = std::thread(thread_body, 1);
    d3_threads[i] = std::thread(thread_body, 2);
  }

  for (size_t i = 0; i < n; ++i) {
    d2_threads[i].join();
    d3_threads[i].join();
  }

  {
    queue q;

    update_d2_data(q);
    update_d3_data(q);
  }

  std::cout << "pass" << std::endl;
  return 0;
}
