//==- kernel_compiler_sycl.cpp --- kernel_compiler extension tests   -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: (opencl || level_zero)
// UNSUPPORTED: accelerator

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/sycl.hpp>

auto constexpr AddEmH = R"===(
  int AddEm(int a, int b){
    return a + b + 5;
  }
)===";

auto constexpr SYCLSource = R"===(
#include <sycl/sycl.hpp>
#include "AddEm.h"

SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((sycl::ext::oneapi::experimental::nd_range_kernel<1>))
void ff_cp(int *ptr) {
  //sycl::nd_item<1> Item = sycl::ext::oneapi::experimental::this_nd_item<1>();
  sycl::nd_item<1> Item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
  sycl::id<1> GId = Item.get_global_id();
  ptr[GId.get(0)] = AddEm(GId.get(0), 37);
}
)===";

void test_1(sycl::queue &Queue, sycl::kernel &Kernel) {
  constexpr int Range = 10;
  int *usmPtr = sycl::malloc_shared<int>(Range, Queue);
  int start = 3;

  sycl::nd_range<1> R1{{Range}, {1}};

  bool Passa = true;

  memset(usmPtr, 0, Range * sizeof(int));
  Queue.submit([&](sycl::handler &Handler) {
    Handler.set_arg(0, usmPtr);
    // Handler.set_arg(1, start);
    // Handler.set_arg(2, Range);
    Handler.parallel_for(R1, Kernel);
  });
  Queue.wait();

  for (int i = 0; i < Range; i++) {
    std::cout << usmPtr[i] << " ";
    // assert(usmPtr[i] = i + 42);
  }
  std::cout << std::endl;

  sycl::free(usmPtr, Queue);
}

void test_build_and_run() {
  namespace syclex = sycl::ext::oneapi::experimental;
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
  using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;

  // TODO: remove this dance from other tests
  // this dance avoids a bug on L0, ensuring context is of exactly one device
  // sycl::device d;
  // sycl::context ctx{d};
  // sycl::queue q{ctx, d};

  sycl::queue q;
  sycl::context ctx = q.get_context();

  // TODO: replace is_source_kernel_bundle_supported() with
  // device::ext_oneapi_can_compile()
  bool ok = syclex::is_source_kernel_bundle_supported(
      ctx.get_backend(), syclex::source_language::sycl);
  if (!ok) {
    std::cout << "Apparently this backend does not support SYCL source "
                 "kernel bundle extension: "
              << ctx.get_backend() << std::endl;
    return;
  }

  // TODO: replace with device.ext_support_blah_nha
  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::sycl, SYCLSource,
      syclex::properties{syclex::include_files{"AddEm.h", AddEmH}});
  // compilation of empty prop list, no devices
  exe_kb kbExe = syclex::build(kbSrc);

  sycl::kernel k = kbExe.ext_oneapi_get_kernel(
      "__free_function_ff_cp"); // amend __free_function_  to kernel f name.

  // NOTE THIS NOISE
  // sycl::kernel_bundle<sycl::bundle_state::executable> kb =
  // syclexp::build(kb_src,
  // syclexp::properties{syclexp::registered_kernel_names{"mykernels::bar"}});
  // sycl::kernel k = kb.ext_oneapi_get_kernel("mykernels::bar");

  // 4
  test_1(q, k);
}

int main() {
  // TODO - awaiting guidance
  // #ifndef SYCL_EXT_ONEAPI_KERNEL_COMPILER_SYCL
  //   static_assert(false, "KernelCompiler SYCL feature test macro undefined");
  // #endif

#ifdef SYCL_EXT_ONEAPI_KERNEL_COMPILER
  test_build_and_run();
  // test_error();
#else
  static_assert(false, "Kernel Compiler feature test macro undefined");
#endif
  return 0;
}
