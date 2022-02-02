// REQUIRES: xptifw, opencl
// RUN: %clangxx %s -DXPTI_COLLECTOR -DXPTI_CALLBACK_API_EXPORTS %xptifw_lib %shared_lib %fPIC %cxx_std_optionc++17 -o %t_collector.dll
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env XPTI_TRACE_ENABLE=1 XPTI_FRAMEWORK_DISPATCHER=%xptifw_dispatcher XPTI_SUBSCRIBERS=%t_collector.dll SYCL_DEVICE_FILTER=opencl %t.out | FileCheck %s 2>&1

#ifdef XPTI_COLLECTOR

#include "../Inputs/buffer_info_collector.cpp"

#else

#include <sycl/sycl.hpp>

int main() {
  bool MismatchFound = false;
  sycl::queue Queue{};

  int Array[4];
  {
    sycl::range<1> NumOfWorkItems{4};
    // CHECK:{{[0-9]+}}|Create buffer|[[USERID1:[0-9,a-f,x]+]]|0x{{.*}}|{{i(nt)*}}|4|1|{4,0,0}|{{.*}}host_array.cpp:[[# @LINE + 1]]:26
    sycl::buffer<int, 1> Buffer1(Array, NumOfWorkItems);

    // CHECK:{{[0-9]+}}|Associate buffer|[[USERID1]]|[[BEID1:.*]]
    Queue.submit([&](sycl::handler &cgh) {
      // Get write only access to the buffer on a device.
      auto Accessor1 = Buffer1.get_access<sycl::access::mode::write>(cgh);
      // Execute kernel.
      cgh.parallel_for<class FillBuffer>(NumOfWorkItems, [=](sycl::id<1> WIid) {
        Accessor1[WIid] = static_cast<int>(WIid.get(0));
      });
    });
  }

  // Check the results.
  for (size_t I = 0; I < 4; ++I) {
    if (Array[I] != I) {
      std::cout << "The result is incorrect for element: " << I
                << " , expected: " << I << " , got: " << Array[I] << std::endl;
      MismatchFound = true;
    }
  }

  return MismatchFound;
}
// CHECK:{{[0-9]+}}|Release buffer|[[USERID1]]|[[BEID1]]
// CHECK:{{[0-9]+}}|Destruct buffer|[[USERID1]]
#endif
