// REQUIRES: xptifw, opencl
// RUN: %clangxx %s -DXPTI_COLLECTOR -DXPTI_CALLBACK_API_EXPORTS %xptifw_lib %shared_lib %fPIC %cxx_std_optionc++17 -o %t_collector.dll
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env XPTI_TRACE_ENABLE=1 XPTI_FRAMEWORK_DISPATCHER=%xptifw_dispatcher XPTI_SUBSCRIBERS=%t_collector.dll %BE_RUN_PLACEHOLDER %t.out | FileCheck %s 2>&1

#ifdef XPTI_COLLECTOR

#include "../Inputs/buffer_info_collector.cpp"

#else

#include <sycl/sycl.hpp>

using namespace sycl::access;

int main() {
  bool MismatchFound = false;
  sycl::queue Queue{};

  // CHECK:{{[0-9]+}}|Create buffer|[[BUFFERID:[0-9,a-f,x]+]]|0x0|{{i(nt)*}}|4|1|{3,0,0}|{{.*}}accessors.cpp:[[# @LINE + 1]]:24
  sycl::buffer<int, 1> Buf(3);

  sycl::range<1> Range{Buf.size()};

  Queue.submit([&](sycl::handler &cgh) {
    // CHECK: {{[0-9]+}}|Construct accessor|[[BUFFERID]]|[[ACCID1:.+]]|2015|1024|{{.*}}accessors.cpp:[[# @LINE + 1]]:15
    auto A1 = Buf.get_access<mode::read, target::constant_buffer>(cgh);
    // CHECK: {{[0-9]+}}|Construct accessor|[[BUFFERID]]|[[ACCID2:.*]]|2014|1025|{{.*}}accessors.cpp:[[# @LINE + 1]]:15
    auto A2 = Buf.get_access<mode::write>(cgh);
    // CHECK: {{[0-9]+}}|Construct accessor|0x0|[[ACCID3:.*]]|2016|1026|{{.*}}accessors.cpp:[[# @LINE + 1]]:61
    sycl::accessor<int, 1, mode::read_write, target::local> A3(Range, cgh);
    // CHECK: {{[0-9]+}}|Construct accessor|[[BUFFERID]]|[[ACCID4:.*]]|2014|1027|{{.*}}accessors.cpp:[[# @LINE + 1]]:15
    auto A4 = Buf.get_access<mode::discard_write>(cgh);
    // CHECK: {{[0-9]+}}|Construct accessor|[[BUFFERID]]|[[ACCID5:.*]]|2014|1028|{{.*}}accessors.cpp:[[# @LINE + 1]]:15
    auto A5 = Buf.get_access<mode::discard_read_write, target::device>(cgh);
    // CHECK: {{[0-9]+}}|Construct accessor|[[BUFFERID]]|[[ACCID6:.*]]|2014|1029|{{.*}}accessors.cpp:[[# @LINE + 1]]:15
    auto A6 = Buf.get_access<mode::atomic>(cgh);
    cgh.parallel_for<class FillBuffer>(Range, [=](sycl::id<1> WIid) {
      (void)A1;
      (void)A2;
      (void)A3;
      (void)A4;
      (void)A5;
      (void)A6;
    });
  });
  // CHECK: {{[0-9]+}}|Construct accessor|[[BUFFERID]]|[[ACCID7:.*]]|2018|1024|{{.*}}accessors.cpp:[[# @LINE + 1]]:15
  { auto HA = Buf.get_access<mode::read>(); }
  // CHECK: {{[0-9]+}}|Construct accessor|[[BUFFERID]]|[[ACCID8:.*]]|2018|1025|{{.*}}accessors.cpp:[[# @LINE + 1]]:15
  { auto HA = Buf.get_access<mode::write>(); }
  // CHECK: {{[0-9]+}}|Construct accessor|[[BUFFERID]]|[[ACCID9:.*]]|2018|1026|{{.*}}accessors.cpp:[[# @LINE + 1]]:15
  { auto HA = Buf.get_access<mode::read_write>(); }
  // CHECK: {{[0-9]+}}|Construct accessor|[[BUFFERID]]|[[ACCID10:.*]]|2018|1027|{{.*}}accessors.cpp:[[# @LINE + 1]]:15
  { auto HA = Buf.get_access<mode::discard_write>(); }
  // CHECK: {{[0-9]+}}|Construct accessor|[[BUFFERID]]|[[ACCID11:.*]]|2018|1028|{{.*}}accessors.cpp:[[# @LINE + 1]]:15
  { auto HA = Buf.get_access<mode::discard_read_write>(); }
  // CHECK: {{[0-9]+}}|Construct accessor|[[BUFFERID]]|[[ACCID12:.*]]|2018|1029|{{.*}}accessors.cpp:[[# @LINE + 1]]:15
  { auto HA = Buf.get_access<mode::atomic>(); }

  return 0;
}
// CHECK:{{[0-9]+}}|Destruct buffer|[[BUFFERID]]
#endif
