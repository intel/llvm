//==----------- sycl_join.cpp --- kernel_compiler extension tests ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: (opencl || level_zero)
// REQUIRES: aspect-usm_shared_allocations

// UNSUPPORTED: accelerator
// UNSUPPORTED-INTENDED: while accelerator is AoT only, this cannot run there.

// UNSUPPORTED: windows && arch-intel_gpu_bmg_g21
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/17255

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %{l0_leak_check} %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

auto constexpr SYCLSource1 = R"""(
#include <sycl/sycl.hpp>

extern "C" SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::single_task_kernel))
void TestKernel1(int *Ptr) {
  *Ptr = 42;
}
)""";

auto constexpr SYCLSource2 = R"""(
#include <sycl/sycl.hpp>

extern "C" SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::single_task_kernel))
void TestKernel2(int *Ptr) {
  *Ptr = 24;
}
)""";

namespace syclex = sycl::ext::oneapi::experimental;
using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;

class RegularSYCLKernel;

constexpr int RegularSYCLKernelWriteValue = 4224;

void RunRegularSYCLKernel(sycl::queue Q, const exe_kb &Bundle, int *Ptr) {
  Q.submit([&](sycl::handler &CGH) {
     CGH.use_kernel_bundle(Bundle);
     CGH.single_task<RegularSYCLKernel>(
         [=]() { *Ptr = RegularSYCLKernelWriteValue; });
   }).wait_and_throw();
}

int main() {

  sycl::queue Q;
  sycl::context Ctx = Q.get_context();

  if (!Q.get_device().ext_oneapi_can_compile(syclex::source_language::sycl)) {
    std::cout << "Apparently this device does not support `sycl` source "
                 "kernel bundle extension: "
              << Q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    return -1;
  }

  int Failed = 0;

  source_kb KBSrc1 = syclex::create_kernel_bundle_from_source(
      Ctx, syclex::source_language::sycl, SYCLSource1);
  source_kb KBSrc2 = syclex::create_kernel_bundle_from_source(
      Ctx, syclex::source_language::sycl, SYCLSource2);

  // Test joining of source kernel bundles.
  {
    std::vector<source_kb> KBSrcs{KBSrc1, KBSrc2};
    source_kb KBSrcJoined = sycl::join(KBSrcs);

    exe_kb KBExeJoined = syclex::build(KBSrcJoined);
    assert(KBExeJoined.ext_oneapi_has_kernel("TestKernel1"));
    assert(KBExeJoined.ext_oneapi_has_kernel("TestKernel2"));

    sycl::kernel K1 = KBExeJoined.ext_oneapi_get_kernel("TestKernel1");
    sycl::kernel K2 = KBExeJoined.ext_oneapi_get_kernel("TestKernel2");

    int *IntPtr = sycl::malloc_shared<int>(1, Q);
    *IntPtr = 0;

    Q.submit([&](sycl::handler &CGH) {
       CGH.set_args(IntPtr);
       CGH.single_task(K1);
     }).wait_and_throw();

    if (*IntPtr != 42) {
      std::cout << "TestKernel1 in joined source bundles failed: " << *IntPtr
                << " != 42\n";
      ++Failed;
    }

    Q.submit([&](sycl::handler &CGH) {
       CGH.set_args(IntPtr);
       CGH.single_task(K2);
     }).wait_and_throw();

    if (*IntPtr != 24) {
      std::cout << "TestKernel1 in joined source bundles failed: " << *IntPtr
                << " != 24\n";
      ++Failed;
    }

    sycl::free(IntPtr, Q);
  }

  auto KBExe1 = std::make_shared<exe_kb>(syclex::build(KBSrc1));
  auto KBExe2 = std::make_shared<exe_kb>(syclex::build(KBSrc2));

  // Test joining of source-based executable kernel bundles.
  {
    std::vector<exe_kb> KBExes{*KBExe1, *KBExe2};

    exe_kb KBExeJoined = sycl::join(KBExes);
    assert(KBExeJoined.ext_oneapi_has_kernel("TestKernel1"));
    assert(KBExeJoined.ext_oneapi_has_kernel("TestKernel2"));

    sycl::kernel K1 = KBExeJoined.ext_oneapi_get_kernel("TestKernel1");
    sycl::kernel K2 = KBExeJoined.ext_oneapi_get_kernel("TestKernel2");

    int *IntPtr = sycl::malloc_shared<int>(1, Q);
    *IntPtr = 0;

    Q.submit([&](sycl::handler &CGH) {
       CGH.set_args(IntPtr);
       CGH.single_task(K1);
     }).wait_and_throw();

    if (*IntPtr != 42) {
      std::cout
          << "TestKernel1 in joined source-based executable bundles failed: "
          << *IntPtr << " != 42\n";
      ++Failed;
    }

    Q.submit([&](sycl::handler &CGH) {
       CGH.set_args(IntPtr);
       CGH.single_task(K2);
     }).wait_and_throw();

    if (*IntPtr != 24) {
      std::cout
          << "TestKernel1 in joined source-based executable bundles failed: "
          << *IntPtr << " != 24\n";
      ++Failed;
    }

    sycl::free(IntPtr, Q);
  }

  // Test joining of executable kernel bundles.
  {
    sycl::kernel_id RegularSYCLKernelID =
        sycl::get_kernel_id<RegularSYCLKernel>();
    std::vector<sycl::kernel_id> RegularSYCLKernelIDs{RegularSYCLKernelID};
    exe_kb RegularKBExe =
        sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            Ctx, RegularSYCLKernelIDs);
    std::vector<exe_kb> KBExes{*KBExe1, *KBExe2, RegularKBExe};

    exe_kb KBExeJoined = sycl::join(KBExes);
    assert(KBExeJoined.ext_oneapi_has_kernel("TestKernel1"));
    assert(KBExeJoined.ext_oneapi_has_kernel("TestKernel2"));
    assert(KBExeJoined.has_kernel(RegularSYCLKernelID, Q.get_device()));
    assert(KBExeJoined.has_kernel<RegularSYCLKernel>());

    sycl::kernel K1 = KBExeJoined.ext_oneapi_get_kernel("TestKernel1");
    sycl::kernel K2 = KBExeJoined.ext_oneapi_get_kernel("TestKernel2");
    sycl::kernel RegularSYCLK = KBExeJoined.get_kernel(RegularSYCLKernelID);

    int *IntPtr = sycl::malloc_shared<int>(1, Q);
    *IntPtr = 0;

    Q.submit([&](sycl::handler &CGH) {
       CGH.set_args(IntPtr);
       CGH.single_task(K1);
     }).wait_and_throw();

    if (*IntPtr != 42) {
      std::cout << "TestKernel1 in joined mixed executable bundles failed: "
                << *IntPtr << " != 42\n";
      ++Failed;
    }

    Q.submit([&](sycl::handler &CGH) {
       CGH.set_args(IntPtr);
       CGH.single_task(K2);
     }).wait_and_throw();

    if (*IntPtr != 24) {
      std::cout << "TestKernel1 in joined mixed executable bundles failed: "
                << *IntPtr << " != 24\n";
      ++Failed;
    }

    Q.submit([&](sycl::handler &CGH) {
       CGH.set_args(IntPtr);
       CGH.single_task(RegularSYCLK);
     }).wait_and_throw();

    if (*IntPtr != RegularSYCLKernelWriteValue) {
      std::cout << "Regular SYCL kernel (explicit) in joined mixed executable "
                   "bundles failed: "
                << *IntPtr << " != " << RegularSYCLKernelWriteValue << "\n";
      ++Failed;
    }
    *IntPtr = 0;

    RunRegularSYCLKernel(Q, KBExeJoined, IntPtr);

    if (*IntPtr != RegularSYCLKernelWriteValue) {
      std::cout << "Regular SYCL kernel (implicit) in joined mixed executable "
                   "bundles failed: "
                << *IntPtr << " != " << RegularSYCLKernelWriteValue << "\n";
      ++Failed;
    }

    sycl::free(IntPtr, Q);
  }

  // Test joining of executable kernel bundles with the original bundles dying
  // before the parent.
  {
    std::vector<exe_kb> KBExes{*KBExe1, *KBExe2};

    KBExe1.reset();
    KBExe2.reset();

    exe_kb KBExeJoined = sycl::join(KBExes);
    assert(KBExeJoined.ext_oneapi_has_kernel("TestKernel1"));
    assert(KBExeJoined.ext_oneapi_has_kernel("TestKernel2"));

    sycl::kernel K1 = KBExeJoined.ext_oneapi_get_kernel("TestKernel1");
    sycl::kernel K2 = KBExeJoined.ext_oneapi_get_kernel("TestKernel2");

    int *IntPtr = sycl::malloc_shared<int>(1, Q);
    *IntPtr = 0;

    Q.submit([&](sycl::handler &CGH) {
       CGH.set_args(IntPtr);
       CGH.single_task(K1);
     }).wait_and_throw();

    if (*IntPtr != 42) {
      std::cout << "TestKernel1 in joined source-based executable bundles with "
                   "dead parents failed: "
                << *IntPtr << " != 42\n";
      ++Failed;
    }

    Q.submit([&](sycl::handler &CGH) {
       CGH.set_args(IntPtr);
       CGH.single_task(K2);
     }).wait_and_throw();

    if (*IntPtr != 24) {
      std::cout << "TestKernel1 in joined source-based executable bundles with "
                   "dead parents failed: "
                << *IntPtr << " != 24\n";
      ++Failed;
    }

    sycl::free(IntPtr, Q);
  }

  return Failed;
}
