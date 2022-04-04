// REQUIRES: xptifw, opencl, (cpu || accelerator)
// RUN: %clangxx %s -DXPTI_COLLECTOR -DXPTI_CALLBACK_API_EXPORTS %xptifw_lib %shared_lib %fPIC %cxx_std_optionc++17 -o %t_collector.dll
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env XPTI_TRACE_ENABLE=1 XPTI_FRAMEWORK_DISPATCHER=%xptifw_dispatcher XPTI_SUBSCRIBERS=%t_collector.dll SYCL_DEVICE_FILTER=opencl %t.out | FileCheck %s 2>&1

#ifdef XPTI_COLLECTOR

#include "../Inputs/buffer_info_collector.cpp"

#else

#include <sycl/sycl.hpp>

int main() {
  bool MismatchFound = false;

  sycl::device Device{sycl::ext::oneapi::filter_selector{"cpu,accelerator"}};
  auto Devices = Device.create_sub_devices<
      sycl::info::partition_property::partition_equally>(2);

  int Array[4] = {0};
  {
    sycl::queue Queue1{Devices[0]};
    sycl::queue Queue2{Devices[1]};
    sycl::range<1> NumOfWorkItems{4};
    // CHECK:{{[0-9]+}}|Create buffer|[[USERID1:0x[0-9,a-f,x]+]]|0x{{.*}}|{{i(nt)*}}|4|1|{4,0,0}|{{.*}}multiple_queues.cpp:[[# @LINE + 1]]:26
    sycl::buffer<int, 1> Buffer1(Array, NumOfWorkItems);

    Queue1.submit([&](sycl::handler &cgh) {
      // CHECK: {{[0-9]+}}|Construct accessor|[[USERID1]]|[[ACCID1:.*]]|2014|1025|{{.*}}multiple_queues.cpp:[[# @LINE + 1]]:24
      auto Accessor1 = Buffer1.get_access<sycl::access::mode::write>(cgh);
      // CHECK:{{[0-9]+}}|Associate buffer|[[USERID1]]|[[BEID1:.*]]
      cgh.parallel_for<class FillBuffer>(NumOfWorkItems, [=](sycl::id<1> WIid) {
        Accessor1[WIid] = static_cast<int>(WIid.get(0));
      });
    });
    Queue1.wait();

    Queue2.submit([&](sycl::handler &cgh) {
      // CHECK: {{[0-9]+}}|Construct accessor|[[USERID1]]|[[ACCID2:.*]]|2014|1025|{{.*}}multiple_queues.cpp:[[# @LINE + 1]]:24
      auto Accessor1 = Buffer1.get_access<sycl::access::mode::write>(cgh);
      // CHECK:{{[0-9]+}}|Associate buffer|[[USERID1]]|[[BEID2:.*]]
      cgh.parallel_for<class MulBuffer>(NumOfWorkItems, [=](sycl::id<1> WIid) {
        Accessor1[WIid] *= static_cast<int>(WIid.get(0));
      });
    });
  }
  // CHECK:{{[0-9]+}}|Release buffer|[[USERID1]]|[[BEID1]]
  // CHECK:{{[0-9]+}}|Release buffer|[[USERID1]]|[[BEID2]]
  // CHECK:{{[0-9]+}}|Destruct buffer|[[USERID1]]

  // Check the results.
  for (size_t I = 0; I < 4; ++I) {
    if (Array[I] != I * I) {
      std::cout << "The result is incorrect for element: " << I
                << " , expected: " << I * I << " , got: " << Array[I]
                << std::endl;
      MismatchFound = true;
    }
  }

  return MismatchFound;
}
#endif
