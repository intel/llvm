//=- kernel_compiler_sycl_jit_device_globals.cpp - RTC device globals tests -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: (opencl || level_zero)
// UNSUPPORTED: accelerator, opencl && gpu
// UNSUPPORTED-TRACKER: GSD-4287

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %{l0_leak_check} %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

auto constexpr DGSource = R"===(
#include <sycl/sycl.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

syclex::device_global<int32_t> DG;

extern "C" SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (syclex::single_task_kernel)) void ff_dg_adder(int val) {
  DG += val;
}

syclex::device_global<int64_t, decltype(syclex::properties(syclex::device_image_scope))> DG_DIS;

extern "C" SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (syclex::single_task_kernel)) void ff_swap(int64_t *val) {
  int64_t tmp = DG_DIS;
  DG_DIS = *val;
  *val = tmp;
}

)===";

int test_device_global() {
  namespace syclex = sycl::ext::oneapi::experimental;
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
  using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;

  sycl::queue q;
  sycl::context ctx = q.get_context();
  sycl::device d = q.get_device();

  bool ok = d.ext_oneapi_can_compile(syclex::source_language::sycl_jit);
  if (!ok) {
    std::cout << "Apparently this device does not support `sycl_jit` source "
                 "kernel bundle extension: "
              << d.get_info<sycl::info::device::name>() << std::endl;
    return -1;
  }

  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::sycl_jit, DGSource);

  exe_kb kbExe1 = syclex::build(kbSrc);
  auto addK = kbExe1.ext_oneapi_get_kernel("ff_dg_adder");

  // Check presence of device globals.
  assert(kbExe1.ext_oneapi_has_device_global("DG", d));
  // Querying a non-existing device global shall not crash.
  assert(!kbExe1.ext_oneapi_has_device_global("bogus_DG", d));

  void *dgAddr = kbExe1.ext_oneapi_get_device_global_address("DG", d);
  size_t dgSize = kbExe1.ext_oneapi_get_device_global_size("DG", d);
  assert(dgSize == 4);

  int32_t val;
  auto checkVal = [&](int32_t expected) {
    val = -1;
    q.memcpy(&val, dgAddr, dgSize).wait();
    std::cout << "val: " << val << " == " << expected << '\n';
    assert(val == expected);
  };

  // Device globals are zero-initialized.
  checkVal(0);

  // Set the DG.
  val = 123;
  q.memcpy(dgAddr, &val, dgSize).wait();
  checkVal(123);

  // Run a kernel using it.
  val = -17;
  q.submit([&](sycl::handler &CGH) {
    CGH.set_arg(0, val);
    CGH.single_task(addK);
  });
  q.wait();
  checkVal(123 - 17);

  // Test that each bundle has its distinct set of globals.
  exe_kb kbExe2 = syclex::build(kbSrc);
  dgAddr = kbExe2.ext_oneapi_get_device_global_address("DG", d);
  checkVal(0);

  dgAddr = kbExe1.ext_oneapi_get_device_global_address("DG", d);
  checkVal(123 - 17);

  // Test global with `device_image_scope`. We currently cannot read/write these
  // from the host, but they should work device-only.
  auto swapK = kbExe2.ext_oneapi_get_kernel("ff_swap");
  int64_t *valBuf = sycl::malloc_shared<int64_t>(1, q);
  *valBuf = -1;
  auto doSwap = [&]() {
    q.submit([&](sycl::handler &CGH) {
      CGH.set_arg(0, valBuf);
      CGH.single_task(swapK);
    });
    q.wait();
  };

  doSwap();
  assert(*valBuf == 0);
  doSwap();
  assert(*valBuf == -1);

  sycl::free(valBuf, q);

  return 0;
}

int test_error() {
  namespace syclex = sycl::ext::oneapi::experimental;
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
  using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;

  sycl::queue q;
  sycl::context ctx = q.get_context();
  sycl::device d = q.get_device();

  bool ok = d.ext_oneapi_can_compile(syclex::source_language::sycl_jit);
  if (!ok) {
    return 0;
  }

  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::sycl_jit, DGSource);
  exe_kb kbExe = syclex::build(kbSrc);

  try {
    kbExe.ext_oneapi_get_device_global_address("DG_DIS", d);
    assert(false && "we should not be here");
  } catch (sycl::exception &e) {
    assert(e.code() == sycl::errc::invalid);
    assert(std::string(e.what()).find(
               "Cannot query USM pointer for device global with "
               "'device_image_scope' property") != std::string::npos);
  }
  return 0;
}

int main() {
#ifdef SYCL_EXT_ONEAPI_KERNEL_COMPILER
  return test_device_global() || test_error();
#else
  static_assert(false, "Kernel Compiler feature test macro undefined");
#endif
  return 0;
}
