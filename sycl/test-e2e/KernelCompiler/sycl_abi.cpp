//==--- sycl_abi.cpp --- kernel_compiler extension tests -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: (opencl || level_zero)
// REQUIRES: aspect-usm_device_allocations

// UNSUPPORTED: accelerator
// UNSUPPORTED-INTENDED: while accelerator is AoT only, this cannot run there.

// UNSUPPORTED: windows && arch-intel_gpu_bmg_g21
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/17255

// RUN: %{build} -o %t.out
// RUN: %{l0_leak_check} %{run} %t.out

// COM: This test checks that device_copyable STL classes can be passed from
//      host to the runtime-compiled device code.

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

auto constexpr SYCLSource = R"""(
#include <sycl/sycl.hpp>

extern "C" SYCL_EXTERNAL 
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((sycl::ext::oneapi::experimental::single_task_kernel))
void stl_test(int *out, std::optional<int> oi, std::variant<float, int> vfi,
              std::pair<short, long long> *psll, std::tuple<char, short, int> *tcsi,
              std::array<int, 3> ai) {
  int res = 0;
  if (oi.has_value())
    res += *oi;
  if (std::holds_alternative<int>(vfi))
    res += *std::get_if<int>(&vfi);
  res *= std::get<0>(*psll);
  res *= std::get<2>(*tcsi);
  res *= ai[1];
  *out = res;
}
)""";

namespace syclex = sycl::ext::oneapi::experimental;

void run(sycl::queue q, sycl::kernel k, std::optional<int> oi,
         std::variant<float, int> vfi, std::pair<short, long long> psll,
         std::tuple<char, short, int> tcsi, std::array<int, 3> ai) {

  int *res = sycl::malloc_shared<int>(1, q);
  auto *pair = sycl::malloc_shared<decltype(psll)>(1, q);
  auto *tuple = sycl::malloc_shared<decltype(tcsi)>(1, q);
  *pair = psll;
  *tuple = tcsi;
  *res = -1;
  q.submit([&](sycl::handler &cgh) {
     cgh.set_args(res, oi, vfi, pair, tuple, ai);
     cgh.single_task(k);
   }).wait();

  {
    int ref = 0;
    if (oi.has_value())
      ref += *oi;
    if (std::holds_alternative<int>(vfi))
      ref += *std::get_if<int>(&vfi);
    ref *= std::get<0>(psll);
    ref *= std::get<2>(tcsi);
    ref *= ai[1];
    std::cout << *res << " == " << ref << "\n";
    assert(*res == ref);
  }

  sycl::free(res, q);
  sycl::free(pair, q);
  sycl::free(tuple, q);
}

int test_abi(sycl::queue q) {
  namespace syclex = sycl::ext::oneapi::experimental;
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
  using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;

  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      q.get_context(), syclex::source_language::sycl, SYCLSource);
  exe_kb kbExe = syclex::build(kbSrc, syclex::properties{});

  sycl::kernel k = kbExe.ext_oneapi_get_kernel("stl_test");

  std::pair<short, long long> pai{2, 4};
  std::tuple<char, short, int> tup{8, 16, 32};
  std::array<int, 3> arr{64, 128, 256};

  run(q, k, 1, 2, pai, tup, arr);
  run(q, k, std::nullopt, 3.14f, pai, tup, arr);
  run(q, k, 3, 3.14f, pai, tup, arr);
  run(q, k, std::nullopt, 4, pai, tup, arr);

  return 0;
}

int main() {
#ifdef SYCL_EXT_ONEAPI_KERNEL_COMPILER
  sycl::queue q;

  bool ok =
      q.get_device().ext_oneapi_can_compile(syclex::source_language::sycl);
  if (!ok) {
    return -1;
  }
  return test_abi(q);
#else
  static_assert(false, "Kernel Compiler feature test macro undefined");
#endif
  return 0;
}
