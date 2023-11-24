// REQUIRES: gpu, level_zero
// RUN: %{build} %level_zero_options -o %t.out
// RUN: env UR_L0_LEAKS_DEBUG=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{run} %t.out 0 2>&1 | FileCheck %s --check-prefixes=CHECK-STD
// RUN: env UR_L0_LEAKS_DEBUG=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{run} %t.out 1 2>&1 | FileCheck %s --check-prefixes=CHECK-IMM
//
// Check that queue submission mode is honored when creating queue.
//
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

void queue_submit(queue &Q) {
  Q.submit([&](handler &cgh) {
     cgh.single_task([=]() {
       // [kernel code]
     });
   }).wait();
}

// Command argument is 0 / 1 to select standard / immediate command lists.
int main(int argc, char *argv[]) {
  bool Immediate = false;
  if (argc > 1) {
    Immediate = std::stoi(argv[1]) != 0;
  }
  property_list P;
  if (Immediate)
    P = ext::intel::property::queue::immediate_command_list();
  else
    P = ext::intel::property::queue::no_immediate_command_list();

  // CHECK-STD: zeCommandListCreateImmediate = 1
  // CHECK-IMM: zeCommandListCreateImmediate = 2
  queue Q1{P};
  queue_submit(Q1);

  return 0;
}
