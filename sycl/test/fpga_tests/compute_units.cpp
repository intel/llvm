// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==------------ compute_units.cpp - SYCL FPGA compute units test ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/INTEL/fpga_extensions.hpp>

using namespace sycl;

constexpr float kTestData = 555;

constexpr size_t kComputeUnits = 5;
using Pipes = INTEL::PipeArray<class MyPipe, float, 4, kComputeUnits + 1>;

class source_kernel;
class sink_kernel;

template <std::size_t ID> class chain_kernel;

// Write the first piece of data to the pipeline
void SourceKernel(queue &q, float data) {

  q.submit([&](handler &h) {
    h.single_task<source_kernel>([=]() { Pipes::PipeAt<0>::write(data); });
  });
}

// Grab the data out of the pipeline and return to host in out_data array
void SinkKernel(queue &q, std::array<float, 1> &out_data) {

  buffer<float, 1> out_buf(out_data.data(), 1);

  q.submit([&](handler &h) {
    auto out_accessor = out_buf.get_access<access::mode::write>(h);
    h.single_task<sink_kernel>(
        [=]() { out_accessor[0] = Pipes::PipeAt<kComputeUnits>::read(); });
  });
}

template <int TestNumber> int test_compute_units(queue q) {

  std::array<float, 1> out_data = {0};

  SourceKernel(q, kTestData);

  INTEL::submit_compute_units<kComputeUnits, chain_kernel>(q, [=](auto ID) {
    // read from id, not id-1 because the index_sequence starts from 0
    float f = Pipes::PipeAt<ID>::read();
    Pipes::PipeAt<ID + 1>::write(f);
  });

  SinkKernel(q, out_data);

  if (out_data[0] != kTestData) {
    std::cout << "Test: " << TestNumber << "\nResult mismatches " << out_data[0]
              << " Vs expected " << kTestData << std::endl;
    return -1;
  }

  return 0;
}

int main() {
  cl::sycl::queue Queue;

  if (!Queue.get_device()
           .get_info<cl::sycl::info::device::kernel_kernel_pipe_support>()) {
    std::cout << "SYCL_INTEL_data_flow_pipes not supported, skipping"
              << std::endl;
    return 0;
  }

  int Result = test_compute_units</*test number*/ 1>(Queue);
  return Result;
}
