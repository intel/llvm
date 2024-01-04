// RUN: %clangxx -fsycl -fsycl-device-only -O2 -S -emit-llvm -o - %s | FileCheck %s

// This test checks the correctness of matrix types converted into strings

//           "matrix_type,use,rows,cols"
// CHECK: !{!"matrix_type::bf16,use::a,12,12"}
// CHECK: !{!"matrix_type::fp16,use::a,12,12"}
// CHECK: !{!"matrix_type::tf32,use::a,12,12"}
// CHECK: !{!"matrix_type::fp32,use::a,12,12"}
// CHECK: !{!"matrix_type::fp64,use::a,12,12"}
// CHECK: !{!"matrix_type::sint8,use::a,12,12"}
// CHECK: !{!"matrix_type::sint16,use::a,12,12"}
// CHECK: !{!"matrix_type::sint32,use::a,12,12"}
// CHECK: !{!"matrix_type::sint64,use::a,12,12"}
// CHECK: !{!"matrix_type::uint8,use::a,12,12"}
// CHECK: !{!"matrix_type::uint16,use::a,12,12"}
// CHECK: !{!"matrix_type::uint32,use::a,12,12"}
// CHECK: !{!"matrix_type::uint64,use::a,12,12"}

#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental::matrix;

constexpr size_t Size = 12;

template <typename T> void test(sycl::queue &q) {
  q.submit([&](sycl::handler &cgh) {
    cgh.single_task([]() {
      joint_matrix<sycl::sub_group, T, use::a, Size, Size, layout::row_major> m;
    });
  });
}

int main() {
  sycl::queue q;

  test<sycl::ext::oneapi::bfloat16>(q);
  test<sycl::half>(q);
  test<sycl::ext::oneapi::experimental::matrix::precision::tf32>(q);
  test<float>(q);
  test<double>(q);
  test<int8_t>(q);
  test<int16_t>(q);
  test<int32_t>(q);
  test<int64_t>(q);
  test<uint8_t>(q);
  test<uint16_t>(q);
  test<uint32_t>(q);
  test<uint64_t>(q);

  return 0;
}
