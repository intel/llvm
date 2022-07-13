// FIXME unsupported on level_zero until L0 Plugin support becomes available for
// discard_queue_events
// UNSUPPORTED: level_zero
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
//
// RUN: env SYCL_PI_TRACE=2 %CPU_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %CPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// RUN: env SYCL_PI_TRACE=2 %GPU_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %GPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// RUN: env SYCL_PI_TRACE=2 %ACC_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %ACC_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
//
// The test checks that the last parameter is `nullptr` for
// piEnqueueKernelLaunch for USM kernel using local accessor, but
// is not `nullptr` for kernel using buffer accessor.
// {{0|0000000000000000}} is required for various output on Linux and Windows.
//
// CHECK: ---> piEnqueueKernelLaunch(
// CHECK:        pi_event * :
// CHECK-NEXT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
//
// CHECK: ---> piEnqueueKernelLaunch(
// CHECK:        pi_event * :
// CHECK-NOT:        pi_event * : {{0|0000000000000000}}[ nullptr ]
// CHECK: --->  pi_result : PI_SUCCESS
//
// CHECK: The test passed.

#include <cassert>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace cl::sycl;
static constexpr int MAGIC_NUM = -1;
static constexpr size_t BUFFER_SIZE = 16;

void RunKernelHelper(sycl::queue Q,
                     const std::function<void(int *Harray)> &TestFunction) {
  int *Harray = sycl::malloc_host<int>(BUFFER_SIZE, Q);
  assert(Harray != nullptr);
  for (size_t i = 0; i < BUFFER_SIZE; ++i) {
    Harray[i] = MAGIC_NUM;
  }

  TestFunction(Harray);

  // Checks result
  for (size_t i = 0; i < BUFFER_SIZE; ++i) {
    size_t expected = i + 10;
    assert(Harray[i] == expected);
  }
  free(Harray, Q);
}

int main(int Argc, const char *Argv[]) {

  sycl::property_list props{
      sycl::property::queue::in_order{},
      sycl::ext::oneapi::property::queue::discard_events{}};
  sycl::queue Q(props);
  sycl::range<1> Range(BUFFER_SIZE);

  RunKernelHelper(Q, [&](int *Harray) {
    Q.submit([&](sycl::handler &CGH) {
      const size_t LocalMemSize = BUFFER_SIZE;
      using LocalAccessor =
          sycl::accessor<int, 1, sycl::access::mode::read_write,
                         sycl::access::target::local>;
      LocalAccessor LocalAcc(LocalMemSize, CGH);

      CGH.parallel_for<class kernel_using_local_memory>(
          Range, [=](sycl::item<1> itemID) {
            size_t i = itemID.get_id(0);
            int *Ptr = LocalAcc.get_pointer();
            Ptr[i] = i + 5;
            Harray[i] = Ptr[i] + 5;
          });
    });
    Q.wait();
  });

  RunKernelHelper(Q, [&](int *Harray) {
    sycl::buffer<int, 1> Buf(Range);
    Q.submit([&](sycl::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);
      CGH.parallel_for<class kernel_using_buffer_accessor>(
          Range, [=](sycl::item<1> itemID) {
            size_t i = itemID.get_id(0);
            Harray[i] = i + 10;
            Acc[i] = i + 20;
          });
    });
    Q.wait();

    // Checks result
    auto HostAcc = Buf.get_access<sycl::access::mode::read>();
    for (size_t i = 0; i < BUFFER_SIZE; ++i) {
      size_t expected = i + 20;
      assert(HostAcc[i] == expected);
    }
  });

  std::cout << "The test passed." << std::endl;
  return 0;
}
