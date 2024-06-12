// RUN: %{build} -o %t.out
//
// RUN: env SYCL_PI_TRACE=2 %{run} %t.out &> %t.txt ; FileCheck %s --input-file %t.txt
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
#include <sycl/detail/core.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

using namespace sycl;
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
  sycl::nd_range<1> NDRange(BUFFER_SIZE, BUFFER_SIZE);
  sycl::range<1> Range(BUFFER_SIZE);

  RunKernelHelper(Q, [&](int *Harray) {
    Q.submit([&](sycl::handler &CGH) {
      const size_t LocalMemSize = BUFFER_SIZE;
      sycl::local_accessor<int, 1> LocalAcc(LocalMemSize, CGH);

      CGH.parallel_for<class kernel_using_local_memory>(
          NDRange, [=](sycl::nd_item<1> ndi) {
            size_t i = ndi.get_global_id(0);
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
    sycl::host_accessor HostAcc(Buf, sycl::read_only);
    for (size_t i = 0; i < BUFFER_SIZE; ++i) {
      size_t expected = i + 20;
      assert(HostAcc[i] == expected);
    }
  });

  std::cout << "The test passed." << std::endl;
  return 0;
}
