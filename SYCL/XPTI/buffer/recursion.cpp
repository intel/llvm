// REQUIRES: xptifw, opencl
// RUN: %clangxx %s -DXPTI_COLLECTOR -DXPTI_CALLBACK_API_EXPORTS %xptifw_lib %shared_lib %fPIC %cxx_std_optionc++17 -o %t_collector.dll
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env XPTI_TRACE_ENABLE=1 XPTI_FRAMEWORK_DISPATCHER=%xptifw_dispatcher XPTI_SUBSCRIBERS=%t_collector.dll %BE_RUN_PLACEHOLDER %t.out | FileCheck %s 2>&1

// It looks like order of events diffres on Windows
#ifdef XPTI_COLLECTOR

#include "../Inputs/buffer_info_collector.cpp"

#else
#include <iostream>
#include <sycl/sycl.hpp>
bool func(sycl::queue &Queue, int depth = 0) {
  bool MismatchFound = false;
  // Create a buffer of 4 ints to be used inside the kernel code.
  sycl::buffer<int, 1> Buffer(4);

  // Size of index space for kernel.
  sycl::range<1> NumOfWorkItems{Buffer.size()};

  // Submit command group(work) to queue.
  Queue.submit([&](sycl::handler &cgh) {
    // Get write only access to the buffer on a device.
    auto Accessor = Buffer.get_access<sycl::access::mode::write>(cgh);
    // Execute kernel.
    cgh.parallel_for<class FillBuffer>(NumOfWorkItems, [=](sycl::id<1> WIid) {
      Accessor[WIid] = static_cast<int>(WIid.get(0));
    });
  });

  // Get read only access to the buffer on the host.
  // This introduces an implicit barrier which blocks execution until the
  // command group above completes.
  const auto HostAccessor = Buffer.get_access<sycl::access::mode::read>();

  // Check the results.
  for (size_t I = 0; I < Buffer.size(); ++I) {
    if (HostAccessor[I] != I) {
      std::cout << "The result is incorrect for element: " << I
                << " , expected: " << I << " , got: " << HostAccessor[I]
                << std::endl;
      MismatchFound = true;
    }
  }

  if (depth > 0)
    MismatchFound &= func(Queue, depth - 1);
  return MismatchFound;
}
int main() {
  bool MismatchFound = false;
  // Create a SYCL queue.
  sycl::queue Queue{};

  // CHECK:{{[0-9]+}}|Create buffer|[[USERID1:0x[0-9,a-f,x]+]]|0x0|{{i(nt)*}}|4|1|{4,0,0}|{{.*}}recursion.cpp:17:24
  // CHECK:{{[0-9]+}}|Associate buffer|[[USERID1]]|[[BEID1:.*]]
  // CHECK:{{[0-9]+}}|Create buffer|[[USERID2:0x[0-9,a-f,x]+]]|0x0|{{i(nt)*}}|4|1|{4,0,0}|{{.*}}recursion.cpp:17:24
  // CHECK:{{[0-9]+}}|Associate buffer|[[USERID2]]|[[BEID2:.*]]
  // CHECK:{{[0-9]+}}|Create buffer|[[USERID3:0x[0-9,a-f,x]+]]|0x0|{{i(nt)*}}|4|1|{4,0,0}|{{.*}}recursion.cpp:17:24
  // CHECK:{{[0-9]+}}|Associate buffer|[[USERID3]]|[[BEID3:.*]]
  // CHECK:{{[0-9]+}}|Release buffer|[[USERID3]]|[[BEID3]]
  // CHECK:{{[0-9]+}}|Destruct buffer|[[USERID3]]
  // CHECK:{{[0-9]+}}|Release buffer|[[USERID2]]|[[BEID2]]
  // CHECK:{{[0-9]+}}|Destruct buffer|[[USERID2]]
  // CHECK:{{[0-9]+}}|Release buffer|[[USERID1]]|[[BEID1]]
  // CHECK:{{[0-9]+}}|Destruct buffer|[[USERID1]]
  MismatchFound &= func(Queue, 2);

  return MismatchFound;
}

#endif
