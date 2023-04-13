//==---- hostpipe_base.cpp - Verify the basic func of fpga hostpipe ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: accelerator
// RUN: %clangxx -fsycl %s -o %t.out -fsycl-targets=%sycl_triple
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <algorithm>
#include <numeric>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include <sycl/ext/intel/experimental/pipe_properties.hpp>
#include <sycl/ext/intel/experimental/pipes.hpp>
#include <sycl/ext/intel/fpga_device_selector.hpp>

using namespace sycl;
using namespace std::chrono;

// forward declare kernel and pipe names to reduce name mangling
class LoopBackKernelID;
class LoopBackKernelID2222;
class H2DPipeID;
class D2HPipeID;

// the host pipes
using ValueT = int;
constexpr size_t kPipeMinCapacity = 8;
constexpr size_t kReadyLatency = 0;
constexpr size_t kBitsPerSymbol = 1;
using default_pipe_properties =
    decltype(sycl::ext::oneapi::experimental::properties());
using H2DPipe = sycl::ext::intel::experimental::pipe<
    H2DPipeID, ValueT, kPipeMinCapacity, default_pipe_properties>;
using D2HPipe = sycl::ext::intel::experimental::pipe<
    D2HPipeID, ValueT, kPipeMinCapacity, default_pipe_properties>;

// forward declare the test functions
void BasicTest(queue &, ValueT *, ValueT *, size_t, size_t);
void LaunchCollectTest(queue &, ValueT *, ValueT *, size_t, size_t);
/////////////////////////////////////////

int main(int argc, char *argv[]) {

  bool passed = true;

  size_t count = 16;
  if (argc > 1)
    count = atoi(argv[1]);

  if (count <= 0) {
    std::cerr << "ERROR: 'count' must be positive\n";
    return 1;
  }
  if (count < kPipeMinCapacity) {
    std::cerr
        << "ERROR: 'count' must be greater than the minimum pipe capacity\n";
    return 1;
  }

  try {
    // create the device queue
    default_selector selector;
    queue q(selector);

    // create input and output data
    std::vector<ValueT> in(count), out(count);
    std::generate(in.begin(), in.end(), [] { return ValueT(rand() % 77); });

    // validation lambda
    auto validate = [](auto &in, auto &out, size_t size) {
      for (int i = 0; i < size; i++) {
        if (out[i] != in[i]) {
          std::cout << "out[" << i << "] != in[" << i << "]"
                    << " (" << out[i] << " != " << in[i] << ")\n";
          return false;
        }
      }
      return true;
    };

    // Basic Test
    std::cout << "Running Basic Test" << std::endl;
    std::fill(out.begin(), out.end(), 0);
    BasicTest(q, in.data(), out.data(), count, 3);

    passed &= validate(in, out, count);
    std::cout << std::endl;

    // Launch and Collect Test
    std::cout << "Running Launch and Collect Test" << std::endl;
    std::fill(out.begin(), out.end(), 0);
    LaunchCollectTest(q, in.data(), out.data(), kPipeMinCapacity, 3);

    passed &= validate(in, out, kPipeMinCapacity);
    std::cout << std::endl;

  } catch (exception const &e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";
    std::terminate();
  }

  if (passed) {
    std::cout << "Test passed." << std::endl;
    return 0;
  } else {
    std::cout << "Test failed." << std::endl;
    return 1;
  }
}

template <typename KernelId, typename InHostPipe, typename OutHostPipe>
event SubmitLoopBackKernel(queue &q, size_t count) {
  return q.single_task<KernelId>([=] {
    for (size_t i = 0; i < count; i++) {
      auto d = InHostPipe::read();
      OutHostPipe::write(d);
    }
  });
}

void BasicTest(queue &q, ValueT *in, ValueT *out, size_t count,
               size_t repeats) {
  std::cout << "\t Submitting Loopback Kernel" << std::endl;
  auto e = SubmitLoopBackKernel<LoopBackKernelID, H2DPipe, D2HPipe>(
      q, count * repeats);

  for (size_t r = 0; r < repeats; r++) {
    std::cout << "\t " << r << ": "
              << "Doing " << count << " writes & reads" << std::endl;
    for (size_t i = 0; i < count; i++) {
      H2DPipe::write(q, in[i]);
      out[i] = D2HPipe::read(q);
    }
  }

  std::cout << "\t Waiting on kernel to finish" << std::endl;
  e.wait();
  std::cout << "\t Done" << std::endl;
}

void LaunchCollectTest(queue &q, ValueT *in, ValueT *out, size_t count,
                       size_t repeats) {
  std::cout << "\t Submitting Loopback Kernel" << std::endl;
  auto e = SubmitLoopBackKernel<LoopBackKernelID, H2DPipe, D2HPipe>(
      q, count * repeats);

  for (size_t r = 0; r < repeats; r++) {
    std::cout << "\t " << r << ": "
              << "Doing " << count << " writes" << std::endl;
    for (size_t i = 0; i < count; i++) {
      H2DPipe::write(q, in[i]);
    }

    std::cout << "\t " << r << ": "
              << "Doing " << count << " reads" << std::endl;
    for (size_t i = 0; i < count; i++) {
      out[i] = D2HPipe::read(q);
    }
  }

  std::cout << "\t Waiting on kernel to finish" << std::endl;
  e.wait();
  std::cout << "\t Done" << std::endl;
}

