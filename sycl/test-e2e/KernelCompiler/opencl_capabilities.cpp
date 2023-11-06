//==- opencl_capabilities.cpp ----------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: cm-compiler

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Here we are testing some of the various args that SYCL can and cannot
// pass to an OpenCL kernel that is compiled with the kernel_compiler.

#include <sycl/sycl.hpp>

namespace syclex = sycl::ext::oneapi::experimental;
using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;

// -----------------------
//  local accessor
// -----------------------
auto constexpr LocalAccCLSource = R"===(
    kernel void test_la(global int *a, local float *b, int n) {
        if (get_local_id(0) == 0) {
          for (int i = 0; i < n; i++)
              b[i] = i;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        bool ok = true;
        for (int i = 0; i < n; i++)
            ok &= (b[i] == i);

        a[get_global_id(0)] = ok;
    }
)===";

void test_local_accessor() {
  using namespace sycl;

  sycl::queue q;
  sycl::context ctx = q.get_context();

  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::opencl, LocalAccCLSource);
  exe_kb kbExe1 = syclex::build(kbSrc);
  sycl::kernel test_kernel = kbExe1.ext_oneapi_get_kernel("test_la");

  constexpr cl_int N_slm = 256;
  constexpr int N_wg = 32;

  cl_int init[N_wg];
  sycl::buffer<cl_int, 1> b(init, N_wg);

  q.submit([&](handler &cgh) {
    auto acc_global = b.get_access<access::mode::write>(cgh);
    local_accessor<float, 1> acc_local(N_slm, cgh);

    cgh.set_arg(0, acc_global);
    cgh.set_arg(1, acc_local);
    cgh.set_arg(2, N_slm);

    cgh.parallel_for(nd_range<1>(N_wg, 1), test_kernel);
  });

  sycl::host_accessor Out{b};
  for (int i = 0; i < N_wg; i++)
    assert(Out[i] == 1);
}

// -----------------------
//  USM pointer and scalars
// -----------------------
auto constexpr USMCLSource = R"===(
__kernel void usm_kernel(__global int *usmPtr, int multiplier,  int added) {
  size_t i = get_global_id(0);
  usmPtr[i] = (i * multiplier) + added;
}
)===";

void test_usm_pointer_and_scalar() {
  sycl::queue q;
  sycl::context ctx = q.get_context();

  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::opencl, USMCLSource);
  exe_kb kbExe1 = syclex::build(kbSrc);
  sycl::kernel usm_kernel = kbExe1.ext_oneapi_get_kernel("usm_kernel");

  int multiplier = 2, added = 100; // the scalars submitted to the kernel
  constexpr size_t N = 32;
  int *usmPtr = sycl::malloc_shared<int>(N, q);

  q.submit([&](sycl::handler &cgh) {
    cgh.set_arg(0, usmPtr);
    cgh.set_arg(1, multiplier); // scalar args
    cgh.set_arg(2, added);
    cgh.parallel_for(sycl::range<1>{N}, usm_kernel);
  });
  q.wait();

  for (size_t i = 0; i < N; i++) {
    assert(usmPtr[i] == ((i * multiplier) + added));
  }

  sycl::free(usmPtr, ctx);
}

// -----------------------
//  structure passed by value
// -----------------------

auto constexpr StructSrc = R"===(
struct pair {
    int multiplier;
    int added;
};
__kernel void struct_kernel(__global int *usmPtr, struct pair adjuster) {
  size_t i = get_global_id(0);
  usmPtr[i] = (i * adjuster.multiplier) + adjuster.added;
}
)===";

struct pair {
  int multiplier;
  int added;
};

void test_struct() {
  sycl::queue q;
  sycl::context ctx = q.get_context();

  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::opencl, StructSrc);
  exe_kb kbExe1 = syclex::build(kbSrc);
  sycl::kernel struct_kernel = kbExe1.ext_oneapi_get_kernel("struct_kernel");

  pair adjuster;
  adjuster.multiplier = 2, adjuster.added = 100;
  constexpr size_t N = 32;
  int *usmPtr = sycl::malloc_shared<int>(N, q);

  q.submit([&](sycl::handler &cgh) {
    cgh.set_arg(0, usmPtr);
    cgh.set_arg(1, adjuster); // struct by value
    cgh.parallel_for(sycl::range<1>{N}, struct_kernel);
  });
  q.wait();

  for (size_t i = 0; i < N; i++) {
    assert(usmPtr[i] == ((i * adjuster.multiplier) + adjuster.added));
  }

  sycl::free(usmPtr, ctx);
}

int main() {
#ifdef SYCL_EXT_ONEAPI_KERNEL_COMPILER_OPENCL
  test_local_accessor();
  test_usm_pointer_and_scalar();
  test_struct();
#else
  static_assert(false, "KernelCompiler OpenCL feature test macro undefined");
#endif
  return 0;
}