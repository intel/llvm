// REQUIRES: xptifw, opencl
// RUN: %clangxx %s -DXPTI_COLLECTOR -DXPTI_CALLBACK_API_EXPORTS %xptifw_lib %shared_lib %fPIC %cxx_std_optionc++17 -o %t_collector.dll
// RUN: %{build} -o %t.out
// RUN: env XPTI_TRACE_ENABLE=1 XPTI_FRAMEWORK_DISPATCHER=%xptifw_dispatcher XPTI_SUBSCRIBERS=%t_collector.dll %{run} %t.out | FileCheck %s

#ifdef XPTI_COLLECTOR

#include "../Inputs/memory_info_collector.cpp"

#else
#include <iostream>
#include <sycl/detail/core.hpp>

int main() {
  bool MismatchFound = false;
  sycl::queue Queue{};

  {
    sycl::range<1> NumOfWorkItems{128};
    // CHECK:{{[0-9]+}}|Create buffer|[[USERID1:0x[0-9,a-f,x]+]]|0x0|{{i(nt)*}}|4|1|{128,0,0}|{{.*}}sub_buffer.cpp:[[# @LINE + 1]]:26
    sycl::buffer<int, 1> Buffer1(NumOfWorkItems);
    // CHECK:{{[0-9]+}}|Create buffer|[[USERID1:0x[0-9,a-f,x]+]]|[[USERID1]]|{{i(nt)*}}|4|1|{32,0,0}|{{.*}}sub_buffer.cpp:[[# @LINE + 1]]:26
    sycl::buffer<int, 1> SubBuffer{Buffer1, sycl::range<1>{32},
                                   sycl::range<1>{32}};

    Queue.submit([&](sycl::handler &cgh) {
      // CHECK: {{[0-9]+}}|Construct accessor|[[USERID1]]|[[ACCID1:.*]]|2014|1025|{{.*}}sub_buffer.cpp:[[# @LINE + 1]]:24
      auto Accessor1 = SubBuffer.get_access<sycl::access::mode::write>(cgh);
      // CHECK:{{[0-9]+}}|Associate buffer|[[USERID1]]|[[BEID1:.*]]
      // CHECK:{{[0-9]+}}|Associate buffer|[[USERID1]]|[[BEID2:.*]]
      cgh.parallel_for<class FillBuffer>(
          sycl::range<1>{32}, [=](sycl::id<1> WIid) {
            Accessor1[WIid] = static_cast<int>(WIid.get(0));
          });
    });
    // CHECK: {{[0-9]+}}|Construct accessor|[[USERID1]]|[[ACCID2:.*]]|2018|1024|{{.*}}sub_buffer.cpp:[[# @LINE + 1]]:22
    auto Accessor1 = Buffer1.get_access<sycl::access::mode::read>();
    for (size_t I = 32; I < 64; ++I) {
      if (Accessor1[I] != I - 32) {
        std::cout << "The result is incorrect for element: " << I
                  << " , expected: " << I - 32 << " , got: " << Accessor1[I]
                  << std::endl;
        MismatchFound = true;
      }
    }
  }

  return MismatchFound;
}
// CHECK:{{[0-9]+}}|Release buffer|[[USERID1]]|[[BEID1]]
// CHECK:{{[0-9]+}}|Release buffer|[[USERID1]]|[[BEID2]]
// CHECK:{{[0-9]+}}|Destruct buffer|[[USERID1]]
#endif
