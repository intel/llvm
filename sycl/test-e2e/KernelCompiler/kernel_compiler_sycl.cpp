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

  // intentionally using deprecated routine, as opposed to this_work_item::get_nd_item<1>()
  sycl::nd_item<1> Item = sycl::ext::oneapi::experimental::this_nd_item<1>();

  sycl::id<1> GId = Item.get_global_id();
  ptr[GId.get(0)] = AddEm(GId.get(0), 37);
}
)===";

auto constexpr BadSource = R"===(
#include <sycl/sycl.hpp>

SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((sycl::ext::oneapi::experimental::nd_range_kernel<1>))
void ff_cp(int *ptr) {

  sycl::nd_item<1> Item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();

  sycl::id<1> GId = Item.get_global_id() + no semi colon !!
  ptr[GId.get(0)] = GId.get(0) + 41;
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

  sycl::queue q;
  sycl::context ctx = q.get_context();

  bool ok =
      q.get_device().ext_oneapi_can_compile(syclex::source_language::sycl);
  if (!ok) {
    std::cout << "Apparently this device does not support SYCL source "
                 "kernel bundle extension: "
              << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    return;
  }

  // create from source
  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::sycl, SYCLSource,
      syclex::properties{syclex::include_files{"AddEm.h", AddEmH}});

  // double check kernel_bundle.get_source() / get_backend()
  sycl::context ctxRes = kbSrc.get_context();
  assert(ctxRes == ctx);
  sycl::backend beRes = kbSrc.get_backend();
  assert(beRes == ctx.get_backend());

  // compilation of empty prop list, no devices
  exe_kb kbExe1 = syclex::build(kbSrc);

  // compilation with props and devices
  std::string log;
  std::vector<std::string> flags{"-g", "-fno-fast-math"};
  std::vector<sycl::device> devs = kbSrc.get_devices();
  exe_kb kbExe2 = syclex::build(
      kbSrc, devs,
      syclex::properties{syclex::build_options{flags}, syclex::save_log{&log}});
  assert(log.find("warning: 'this_nd_item<1>' is deprecated") !=
         std::string::npos);

  // amend __free_function_  to kernel f name.
  sycl::kernel k = kbExe2.ext_oneapi_get_kernel("__free_function_ff_cp");

  // NOTE THIS NOISE
  // sycl::kernel_bundle<sycl::bundle_state::executable> kb =
  // syclexp::build(kb_src,
  // syclexp::properties{syclexp::registered_kernel_names{"mykernels::bar"}});
  // sycl::kernel k = kb.ext_oneapi_get_kernel("mykernels::bar");

  // 4
  test_1(q, k);
}

void test_error() {
  namespace syclex = sycl::ext::oneapi::experimental;
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
  using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;

  sycl::queue q;
  sycl::context ctx = q.get_context();

  bool ok =
      q.get_device().ext_oneapi_can_compile(syclex::source_language::sycl);
  if (!ok) {
    return;
  }

  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::sycl, BadSource);
  try {
    exe_kb kbExe = syclex::build(kbSrc);
    assert(false && "we should not be here");
  } catch (sycl::exception &e) {
    // yas!
    assert(e.code() == sycl::errc::build);
  }
}

int main() {

#ifdef SYCL_EXT_ONEAPI_KERNEL_COMPILER
  test_build_and_run();
  test_error();
#else
  static_assert(false, "Kernel Compiler feature test macro undefined");
#endif
  return 0;
}
