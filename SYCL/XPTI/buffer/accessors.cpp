// REQUIRES: xptifw, opencl
// RUN: %clangxx %s -DXPTI_COLLECTOR -DXPTI_CALLBACK_API_EXPORTS %xptifw_lib %shared_lib %fPIC %cxx_std_optionc++17 -o %t_collector.dll
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env XPTI_TRACE_ENABLE=1 XPTI_FRAMEWORK_DISPATCHER=%xptifw_dispatcher XPTI_SUBSCRIBERS=%t_collector.dll SYCL_DEVICE_FILTER=opencl %t.out | FileCheck %s 2>&1

#ifdef XPTI_COLLECTOR

#include "../Inputs/buffer_info_collector.cpp"

#else

#include <sycl/sycl.hpp>

using namespace sycl::access;

int main() {
  bool MismatchFound = false;
  sycl::queue Queue{};

  // CHECK:{{[0-9]+}}|Create buffer|[[#BUFFERID:]]|{{.*}}accessors.cpp:21:24|{{.*}}accessors.cpp:21:24
  sycl::buffer<int, 1> Buf(4);

  sycl::range<1> Range{Buf.size()};

  Queue.submit([&](sycl::handler &cgh) {
    // CHECK: {{[0-9]+}}|Construct accessor|[[#BUFFERID]]|[[#ACCID1:]]|2015|1024|{{.*}}accessors.cpp:27:15|{{.*}}accessors.cpp:27:15
    auto A1 = Buf.get_access<mode::read, target::constant_buffer>(cgh);
    // CHECK: {{[0-9]+}}|Construct accessor|[[#BUFFERID]]|[[#ACCID2:]]|2014|1025|{{.*}}accessors.cpp:29:15|{{.*}}accessors.cpp:29:15
    auto A2 = Buf.get_access<mode::write>(cgh);
    // CHECK: {{[0-9]+}}|Construct accessor|0|[[#ACCID3:]]|2016|1026|{{.*}}accessors.cpp:31:61|{{.*}}accessors.cpp:31:61
    sycl::accessor<int, 1, mode::read_write, target::local> A3(Range, cgh);
    // CHECK: {{[0-9]+}}|Construct accessor|[[#BUFFERID]]|[[#ACCID4:]]|2014|1027|{{.*}}accessors.cpp:33:15|{{.*}}accessors.cpp:33:15
    auto A4 = Buf.get_access<mode::discard_write>(cgh);
    // CHECK: {{[0-9]+}}|Construct accessor|[[#BUFFERID]]|[[#ACCID5:]]|2014|1028|{{.*}}accessors.cpp:35:15|{{.*}}accessors.cpp:35:15
    auto A5 = Buf.get_access<mode::discard_read_write, target::device>(cgh);
    // CHECK: {{[0-9]+}}|Construct accessor|[[#BUFFERID]]|[[#ACCID6:]]|2014|1029|{{.*}}accessors.cpp:37:15|{{.*}}accessors.cpp:37:15
    auto A6 = Buf.get_access<mode::atomic>(cgh);
    cgh.parallel_for<class FillBuffer>(Range, [=](sycl::id<1> WIid) {});
  });
  // CHECK: {{[0-9]+}}|Construct accessor|[[#BUFFERID]]|[[#ACCID1:]]|2018|1024|{{.*}}accessors.cpp:41:15|{{.*}}accessors.cpp:41:15
  { auto HA = Buf.get_access<mode::read>(); }
  // CHECK: {{[0-9]+}}|Construct accessor|[[#BUFFERID]]|[[#ACCID1:]]|2018|1025|{{.*}}accessors.cpp:43:15|{{.*}}accessors.cpp:43:15
  { auto HA = Buf.get_access<mode::write>(); }
  // CHECK: {{[0-9]+}}|Construct accessor|[[#BUFFERID]]|[[#ACCID1:]]|2018|1026|{{.*}}accessors.cpp:45:15|{{.*}}accessors.cpp:45:15
  { auto HA = Buf.get_access<mode::read_write>(); }
  // CHECK: {{[0-9]+}}|Construct accessor|[[#BUFFERID]]|[[#ACCID1:]]|2018|1027|{{.*}}accessors.cpp:47:15|{{.*}}accessors.cpp:47:15
  { auto HA = Buf.get_access<mode::discard_write>(); }
  // CHECK: {{[0-9]+}}|Construct accessor|[[#BUFFERID]]|[[#ACCID1:]]|2018|1028|{{.*}}accessors.cpp:49:15|{{.*}}accessors.cpp:49:15
  { auto HA = Buf.get_access<mode::discard_read_write>(); }
  // CHECK: {{[0-9]+}}|Construct accessor|[[#BUFFERID]]|[[#ACCID1:]]|2018|1029|{{.*}}accessors.cpp:51:15|{{.*}}accessors.cpp:51:15
  { auto HA = Buf.get_access<mode::atomic>(); }

  return 0;
}
// CHECK:{{[0-9]+}}|Destruct buffer|[[#BUFFERID]]
#endif
