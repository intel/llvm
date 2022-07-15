// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

//==- abs_fix_test.cpp - Test for complation of abs function -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <ext/intel/esimd.hpp>
using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

#define SIMD 16
#define THREAD_NUM 512
template <typename T0, typename T1> void test_abs(cl::sycl::queue q) {
  std::vector<T1> A(THREAD_NUM * SIMD, 0xFFFFFFFF);
  buffer<T1, 1> A_buf(A.data(), range<1>(A.size()));

  nd_range<1> Range((range<1>(THREAD_NUM)), (range<1>(16)));
  auto e = q.submit([&](handler &cgh) {
    auto A_acc = A_buf.template get_access<access::mode::read_write>(cgh);

    cgh.parallel_for(Range, [=](nd_item<1> it) SYCL_ESIMD_FUNCTION {
      T1 scalar_argument = 0xFFFFFFFF;

      __ESIMD_NS::simd<T1, SIMD> A_load_vec;
      A_load_vec.copy_from(A_acc, 0);

      __ESIMD_NS::simd<T0, SIMD> result;

      result = abs<T0, T1, SIMD>(A_load_vec);

      result.copy_to(A_acc, 0);
    });
  });
  e.wait();
}

int main(int argc, char *argv[]) {
  sycl::property_list properties{sycl::property::queue::enable_profiling()};
  auto q = sycl::queue(properties);

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  auto ctxt = q.get_context();

  test_abs<uint32_t, int32_t>(q);
}
