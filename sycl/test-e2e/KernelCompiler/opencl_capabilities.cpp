//==- opencl_capabilities.cpp ----------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: ocloc && (opencl || level_zero)
// UNSUPPORTED: accelerator

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Here we are testing some of the various args that SYCL can and cannot
// pass to an OpenCL kernel that is compiled with the kernel_compiler.

// no backend supports compiling for multiple devices yet, so we limit
// the queue and context to just one.

// IMPORTANT: LevelZero YES!
// Even though this test is covering which OpenCL capabilities
// are covered by the kernel_compiler, this is not a test of only
// the OpenCL devices. The LevelZero backend works with the kernel_compiler
// so long as ocloc is installed and should be able to
// successfully run and pass these tests.

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>
using namespace sycl;
namespace syclex = sycl::ext::oneapi::experimental;
using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;

// -----------------------
//  local accessor
// -----------------------
auto constexpr LocalAccCLSource = R"===(
    kernel void test_la(global int *buf, local float *slm, int n) {
        if (get_local_id(0) == 0) {
          for (int i = 0; i < n; i++)
              slm[i] = i + get_group_id(0);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        bool ok = true;
        for (int i = 0; i < n; i++)
            ok &= (slm[i] == i + get_group_id(0));

        buf[get_global_id(0)] = ok;
    }
)===";

void test_local_accessor() {
  sycl::device d{sycl::default_selector_v};
  sycl::context ctx{d};
  sycl::queue q{ctx, d};

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
__kernel void usm_kernel(__global int *usmPtr, int multiplier,  float added) {
  size_t i = get_global_id(0);
  usmPtr[i] = (i * multiplier) + added;
}
)===";

void test_usm_pointer_and_scalar() {
  sycl::device d{sycl::default_selector_v};
  sycl::context ctx{d};
  sycl::queue q{ctx, d};

  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::opencl, USMCLSource);
  exe_kb kbExe1 = syclex::build(kbSrc);
  sycl::kernel usm_kernel = kbExe1.ext_oneapi_get_kernel("usm_kernel");

  // the scalars submitted to the kernel
  cl_int multiplier = 2;
  cl_float added = 100.f;
  constexpr size_t N = 32;
  cl_int *usmPtr = sycl::malloc_shared<cl_int>(N, q);

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
//  Note that it is imperative that the struct defined in the OpenCL C string
//  exactly match the one used for any kernel args. Overlooking their duality
//  will lead to difficult to discover errors.
// -----------------------

auto constexpr StructSrc = R"===(
struct pair {
    int multiplier;
    float added;
};
__kernel void struct_kernel(__global int *usmPtr, struct pair adjuster) {
  size_t i = get_global_id(0);
  usmPtr[i] = (i * adjuster.multiplier) + adjuster.added;
}
)===";

struct pair {
  cl_int multiplier;
  cl_float added;
};

void test_struct() {
  sycl::device d{sycl::default_selector_v};
  sycl::context ctx{d};
  sycl::queue q{ctx, d};

  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::opencl, StructSrc);
  exe_kb kbExe1 = syclex::build(kbSrc);
  sycl::kernel struct_kernel = kbExe1.ext_oneapi_get_kernel("struct_kernel");

  pair adjuster;
  adjuster.multiplier = 2, adjuster.added = 100.f;
  constexpr size_t N = 32;
  cl_int *usmPtr = sycl::malloc_shared<cl_int>(N, q);

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
