// Complile the kernel with different -fsycl-id-queries-range values
// to check for error thrown when kernel is launched with a range
// larger than the supported one.

// RUN: %{build} -o %t_int.out -fsycl-id-queries-range=int
// RUN: %{build} -o %t_uint.out -fsycl-id-queries-range=uint
// RUN: %{build} -o %t_size.out -fsycl-id-queries-range=size_t

// RUN: %{run} %t_size.out 17179869184 16 2>&1 | FileCheck --check-prefix=CHECK-PASS %s

// clang-format off
// RUN: %{run} %t_int.out 17179869184 8 2>&1 | FileCheck --check-prefix=CHECK-INT-EXCEEDS %s
// RUN: %{run} %t_uint.out 17179869184 4 2>&1 | FileCheck --check-prefix=CHECK-UINT-EXCEEDS %s

// RUN-IF: !cpu, %{run} %t_size.out 17179869184 4 2>&1 | FileCheck --check-prefix=CHECK-SIZE-PER-DIM-EXCEEDS %s
// RUN-IF: cpu, %{run} %t_size.out 17179869184 4 2>&1 | FileCheck --check-prefix=CHECK-PASS %s
// clang-format on

// Tests that launching kernels with large ranges produces proper error
// messages. Validates overflow detection and Level Zero work-group limits

#include <sycl/detail/core.hpp>

using namespace sycl;

// CHECK-PASS: PASS

// CHECK-INT-EXCEEDS: FAIL: The kernel was compiled with -fsycl-id-queries-range=int,
// CHECK-INT-EXCEEDS-SAME: but the provided range/offset exceeds the maximum value
// CHECK-INT-EXCEEDS-SAME: storable in an int. Either reduce the range/offset or
// CHECK-INT-EXCEEDS-SAME: recompile the kernel with -fsycl-id-queries-range=[uint|size_t].

// CHECK-UINT-EXCEEDS: FAIL: The kernel was compiled with -fsycl-id-queries-range=uint,
// CHECK-UINT-EXCEEDS-SAME: but the provided range/offset exceeds the maximum value
// CHECK-UINT-EXCEEDS-SAME: storable in an uint32_t. Either reduce the range/offset or
// CHECK-UINT-EXCEEDS-SAME: recompile the kernel with -fsycl-id-queries-range=size_t.

// CHECK-SIZE-PER-DIM-EXCEEDS: FAIL
void test_nd_range_large_workgroups(queue &q, size_t GlobalSize,
                                    size_t LocalSize) {
  try {
    q.parallel_for(nd_range<1>(range<1>(GlobalSize), range<1>(LocalSize)),
                   [](nd_item<1>) {});
    q.wait_and_throw();
    std::cout << "PASS\n";
  } catch (const sycl::exception &e) {
    std::cout << "FAIL: " << e.what() << std::endl;
  }
}

int main(int argc, char *argv[]) {
  size_t GlobalSize = 17179869184;
  size_t LocalSize = 8;

  // Accept Global and local size as arguments.
  if (argc == 3) {
    GlobalSize = std::stoull(argv[1]);
    LocalSize = std::stoull(argv[2]);
  } else {
    std::cout << "Usage: " << argv[0] << " <global_size> <local_size>\n";
    return 1;
  }

  queue q;
  std::cout << "Device: " << q.get_device().get_info<info::device::name>()
            << "\n";

  test_nd_range_large_workgroups(q, GlobalSize, LocalSize);

  return 0;
}
