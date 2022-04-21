// REQUIRES: xptifw, opencl
// RUN: %clangxx %s -DXPTI_COLLECTOR -DXPTI_CALLBACK_API_EXPORTS %xptifw_lib %shared_lib %fPIC %cxx_std_optionc++17 -o %t_collector.dll
// RUN: %clangxx -fsycl -O2 %s -o %t.opt.out
// RUN: env XPTI_TRACE_ENABLE=1 XPTI_FRAMEWORK_DISPATCHER=%xptifw_dispatcher XPTI_SUBSCRIBERS=%t_collector.dll %BE_RUN_PLACEHOLDER %t.opt.out | FileCheck %s --check-prefixes=CHECK,CHECK-OPT
// RUN: %clangxx -fsycl -fno-sycl-dead-args-optimization %s -o %t.noopt.out
// RUN: env XPTI_TRACE_ENABLE=1 XPTI_FRAMEWORK_DISPATCHER=%xptifw_dispatcher XPTI_SUBSCRIBERS=%t_collector.dll %BE_RUN_PLACEHOLDER %t.noopt.out | FileCheck %s --check-prefixes=CHECK,CHECK-NOOPT

#ifdef XPTI_COLLECTOR

#include "../Inputs/buffer_info_collector.cpp"

#else

#include <sycl/sycl.hpp>

using namespace sycl::access;
constexpr sycl::specialization_id<int> int_id(42);

class Functor1 {
public:
  Functor1(short X_,
           cl::sycl::accessor<int, 1, mode::read_write, target::device> &Acc_)
      : X(X_), Acc(Acc_) {}

  void operator()() const { Acc[0] += X; }

private:
  short X;
  cl::sycl::accessor<int, 1, mode::read_write, target::device> Acc;
};

class Functor2 {
public:
  Functor2(short X_,
           cl::sycl::accessor<int, 1, mode::read_write, target::device> &Acc_)
      : X(X_), Acc(Acc_) {}

  void operator()(sycl::id<1> id = 0) const { Acc[id] += X; }

private:
  short X;
  cl::sycl::accessor<int, 1, mode::read_write, target::device> Acc;
};

int main() {
  bool MismatchFound = false;
  sycl::queue Queue{};

  // CHECK:{{[0-9]+}}|Create buffer|[[BUFFERID:[0-9,a-f,x]+]]|0x0|{{i(nt)*}}|4|1|{5,0,0}|{{.*}}.cpp:[[# @LINE + 1]]:24
  sycl::buffer<int, 1> Buf(5);
  sycl::range<1> Range{Buf.size()};
  short Val = Buf.size();
  auto PtrDevice = sycl::malloc_device<int>(7, Queue);
  auto PtrShared = sycl::malloc_shared<int>(8, Queue);
  Queue
      .submit([&](sycl::handler &cgh) {
        // CHECK: {{[0-9]+}}|Construct accessor|[[BUFFERID]]|[[ACCID1:.+]]|2014|1026|{{.*}}.cpp:[[# @LINE + 1]]:19
        auto A1 = Buf.get_access<mode::read_write>(cgh);
        // CHECK: {{[0-9]+}}|Construct accessor|0x0|[[ACCID2:.*]]|2016|1026|{{.*}}.cpp:[[# @LINE + 1]]:65
        sycl::accessor<int, 1, mode::read_write, target::local> A2(Range, cgh);
        // CHECK-OPT:Node create|{{.*}}FillBuffer{{.*}}|{{.*}}.cpp:[[# @LINE - 6 ]]:3|{5, 1, 1}, {0, 0, 0}, {0, 0, 0}, 6
        // CHECK-NOOPT:Node create|{{.*}}FillBuffer{{.*}}|{{.*}}.cpp:[[# @LINE - 7 ]]:3|{5, 1, 1}, {0, 0, 0}, {0, 0, 0}, 12
        cgh.parallel_for<class FillBuffer>(
            Range, [=](sycl::id<1> WIid, sycl::kernel_handler kh) {
              // CHECK-OPT: arg0 : {1, {{[0-9,a-f,x]+}}, 2, 0}
              int h = Val;
              // CHECK-OPT: arg1 : {1, {{.*}}0, 20, 1}
              A2[WIid[0]] = h;
              // CHECK-OPT: arg2 : {0, [[ACCID1]], 4062, 2}
              // CHECK-OPT: arg3 : {1, [[ACCID1]], 8, 3}
              A1[WIid[0]] = A2[WIid[0]];
              // CHECK-OPT: arg4 : {3, {{.*}}, 8, 4}
              PtrDevice[WIid[0]] = WIid[0];
              // CHECK-OPT: arg5 : {3, {{.*}}, 8, 5}
              PtrShared[WIid[0]] = PtrDevice[WIid[0]];
            });
      })
      .wait();

  // CHECK: Wait begin|{{.*}}.cpp:[[# @LINE + 2]]:3
  // CHECK: Wait end|{{.*}}.cpp:[[# @LINE + 1]]:3
  Queue.wait();

  // CHECK: {{[0-9]+}}|Construct accessor|[[BUFFERID]]|[[ACCID3:.*]]|2018|1024|{{.*}}.cpp:[[# @LINE + 1]]:15
  { auto HA = Buf.get_access<mode::read>(); }

  Queue.submit([&](cl::sycl::handler &cgh) {
    // CHECK: {{[0-9]+}}|Construct accessor|[[BUFFERID]]|[[ACCID4:.+]]|2014|1026|{{.*}}.cpp:[[# @LINE + 1]]:16
    auto Acc = Buf.get_access<mode::read_write>(cgh);
    Functor1 F(Val, Acc);
    // CHECK-OPT: Node create|{{.*}}Functor1|{{.*}}.cpp:[[# @LINE - 4 ]]:3|{1, 1, 1}, {0, 0, 0}, {0, 0, 0}, 3
    // CHECK-NOOPT: Node create|{{.*}}Functor1|{{.*}}.cpp:[[# @LINE - 5 ]]:3|{1, 1, 1}, {0, 0, 0}, {0, 0, 0}, 5
    cgh.single_task(F);
    // CHECK-OPT: arg0 : {1, {{[0-9,a-f,x]+}}, 2, 0}
    // CHECK-OPT: arg1 : {0, [[ACCID4]], 4062, 1}
    // CHECK-OPT: arg2 : {1, [[ACCID4]], 8, 2}
  });

  Queue.submit([&](cl::sycl::handler &cgh) {
    // CHECK: {{[0-9]+}}|Construct accessor|[[BUFFERID]]|[[ACCID5:.+]]|2014|1026|{{.*}}.cpp:[[# @LINE + 1]]:16
    auto Acc = Buf.get_access<mode::read_write>(cgh);
    Functor2 F(Val, Acc);
    // CHECK-OPT: Node create|{{.*}}Functor2|{{.*}}.cpp:[[# @LINE - 4 ]]:3|{5, 1, 1}, {0, 0, 0}, {0, 0, 0}, 3
    // CHECK-NOOPT: Node create|{{.*}}Functor2|{{.*}}.cpp:[[# @LINE - 5 ]]:3|{5, 1, 1}, {0, 0, 0}, {0, 0, 0}, 5
    cgh.parallel_for(Range, F);
    // CHECK-OPT: arg0 : {1, {{[0-9,a-f,x]+}}, 2, 0}
    // CHECK-OPT: arg1 : {0, [[ACCID5]], 4062, 1}
    // CHECK-OPT: arg2 : {1, [[ACCID5]], 8, 2}
  });

  return 0;
}
#endif
