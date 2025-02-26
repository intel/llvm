//==------------------- buffer.cpp - SYCL buffer basic test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A basic test using the --offload-new-driver flag.

// REQUIRES: level_zero
// RUN: %clangxx -fsycl --offload-new-driver %s -o %t.out
// RUN: %{run} %t.out

// RUN: %clangxx -fsycl -fsycl-targets=spirv64 --offload-new-driver %s -o %t1.out
// RUN: %{run} %t1.out

#include <sycl/detail/core.hpp>

int main() {
  // Creating buffer of 4 elements to be used inside the kernel code.
  sycl::buffer<size_t, 1> Buffer(4);

  // Creating SYCL queue.
  sycl::queue Queue;

  // Size of index space for kernel.
  sycl::range<1> NumOfWorkItems{Buffer.size()};

  // Submitting command group(work) to queue.
  Queue.submit([&](sycl::handler &cgh) {
    // Getting write only access to the buffer on a device.
    sycl::accessor Accessor{Buffer, cgh, sycl::write_only};
    // Executing kernel.
    cgh.parallel_for<class FillBuffer>(NumOfWorkItems, [=](sycl::id<1> WIid) {
      // Fill buffer with indexes.
      Accessor[WIid] = WIid.get(0);
    });
  });

  // Getting read only access to the buffer on the host.
  // Implicit barrier waiting for queue to complete the work.
  sycl::host_accessor HostAccessor{Buffer, sycl::read_only};

  // Check the results.
  bool MismatchFound = false;
  for (size_t I = 0; I < Buffer.size(); ++I) {
    if (HostAccessor[I] != I) {
      std::cout << "The result is incorrect for element: " << I
                << " , expected: " << I << " , got: " << HostAccessor[I]
                << std::endl;
      MismatchFound = true;
    }
  }

  if (!MismatchFound) {
    std::cout << "The results are correct!" << std::endl;
  }

  return MismatchFound;
}
