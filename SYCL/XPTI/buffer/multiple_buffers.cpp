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

  // CHECK:{{[0-9]+}}|Create buffer|[[#USERID1:]]|{{.*}}multiple_buffers.cpp:19:24|{{.*}}multiple_buffers.cpp:19:24
  sycl::buffer<int, 1> Buffer1(4);
  // CHECK:{{[0-9]+}}|Create buffer|[[#USERID2:]]|{{.*}}multiple_buffers.cpp:21:24|{{.*}}multiple_buffers.cpp:21:24
  sycl::buffer<int, 1> Buffer2(4);

  sycl::range<1> NumOfWorkItems{Buffer1.size()};

  // CHECK:{{[0-9]+}}|Associate buffer|[[#USERID1]]|[[#BEID1:]]
  // CHECK:{{[0-9]+}}|Associate buffer|[[#USERID2]]|[[#BEID2:]]
  Queue.submit([&](sycl::handler &cgh) {
    // Get write only access to the buffer on a device.
    auto Accessor1 = Buffer1.get_access<sycl::access::mode::write>(cgh);
    auto Accessor2 = Buffer2.get_access<sycl::access::mode::write>(cgh);
    // Execute kernel.
    cgh.parallel_for<class FillBuffer>(NumOfWorkItems, [=](sycl::id<1> WIid) {
      Accessor1[WIid] = static_cast<int>(WIid.get(0));
      Accessor2[WIid] = static_cast<int>(WIid.get(0));
    });
  });

  const auto HostAccessor1 = Buffer1.get_access<sycl::access::mode::read>();
  const auto HostAccessor2 = Buffer2.get_access<sycl::access::mode::read>();

  // Check the results.
  for (size_t I = 0; I < Buffer1.size(); ++I) {
    if (HostAccessor1[I] != I || HostAccessor2[I] != I) {
      std::cout << "The result is incorrect for element: " << I
                << " , expected: " << I << " , got: " << HostAccessor1[I]
                << ", " << HostAccessor2[I] << std::endl;
      MismatchFound = true;
    }
  }

  return MismatchFound;
}
// CHECK:{{[0-9]+}}|Release buffer|[[#USERID2]]|[[#BEID2:]]
// CHECK:{{[0-9]+}}|Destruct buffer|[[#USERID2]]
// CHECK:{{[0-9]+}}|Release buffer|[[#USERID1]]|[[#BEID1:]]
// CHECK:{{[0-9]+}}|Destruct buffer|[[#USERID1]]
#endif
