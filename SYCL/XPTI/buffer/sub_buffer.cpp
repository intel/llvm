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

  {
    sycl::range<1> NumOfWorkItems{4};
    // CHECK:{{[0-9]+}}|Create buffer|[[#USERID1:]]|{{.*}}sub_buffer.cpp:21:26|{{.*}}sub_buffer.cpp:21:26
    sycl::buffer<int, 1> Buffer1(NumOfWorkItems);
    // CHECK:{{[0-9]+}}|Create buffer|[[#USERID1:]]|{{.*}}sub_buffer.cpp:23:26|{{.*}}sub_buffer.cpp:23:26
    sycl::buffer<int, 1> SubBuffer{Buffer1, sycl::range<1>{1},
                                   sycl::range<1>{2}};

    // CHECK:{{[0-9]+}}|Associate buffer|[[#USERID1]]|[[#BEID1:]]
    // CHECK:{{[0-9]+}}|Associate buffer|[[#USERID1]]|[[#BEID2:]]
    Queue.submit([&](sycl::handler &cgh) {
      // Get write only access to the buffer on a device.
      auto Accessor1 = SubBuffer.get_access<sycl::access::mode::write>(cgh);
      // Execute kernel.
      cgh.parallel_for<class FillBuffer>(
          sycl::range<1>{2}, [=](sycl::id<1> WIid) {
            Accessor1[WIid] = static_cast<int>(WIid.get(0));
          });
    });

    auto Accessor1 = Buffer1.get_access<sycl::access::mode::read>();
    // Check the results.
    for (size_t I = 1; I < 3; ++I) {
      if (Accessor1[I] != I - 1) {
        std::cout << "The result is incorrect for element: " << I
                  << " , expected: " << I - 1 << " , got: " << Accessor1[I]
                  << std::endl;
        MismatchFound = true;
      }
    }
  }

  return MismatchFound;
}
// CHECK:{{[0-9]+}}|Release buffer|[[#USERID1]]|[[#BEID1:]]
// CHECK:{{[0-9]+}}|Release buffer|[[#USERID1]]|[[#BEID2:]]
// CHECK:{{[0-9]+}}|Destruct buffer|[[#USERID1]]
#endif
