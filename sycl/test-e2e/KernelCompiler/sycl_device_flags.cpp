//==------------------- df.cpp --- kernel_compiler extension tests   -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: level_zero
// UNSUPPORTED: windows

// IGC shader dump not available on Windows.

// RUN: %{build} -o %t.out
// RUN: env IGC_DumpToCustomDir=%t.dump IGC_ShaderDumpEnable=1 NEO_CACHE_PERSISTENT=0 %{run} %t.out
// RUN: grep -e '-doubleGRF' %t.dump/OCL_asmaf99e2d4667ef6d3_options.txt
// RUN grep -e '-Xfinalizer "-printregusage"'
// ./dump/OCL_asmaf99e2d4667ef6d3_options.txt

// clang-format off
/*
    clang++ -fsycl -o sdf.bin sycl_device_flags.cpp
    IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=./dump NEO_CACHE_PERSISTENT=0 ./sdf.bin 
    grep -e '-doubleGRF' ./dump/OCL_asmaf99e2d4667ef6d3_options.txt
    grep -e '-Xfinalizer "-printregusage"' ./dump/OCL_asmaf99e2d4667ef6d3_options.txt

    Note: there are files named  xxx_options.txt and xxx_internal_options.txt in
    the IGC dump directory. The file with "internal_options.txt"  is NOT the
    correct file.
*/
// clang-format on

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

// TODO: remove SYCL_EXTERNAL once it is no longer needed.
auto constexpr SYCLSource = R"===(
#include <sycl/sycl.hpp>

// use extern "C" to avoid name mangling
extern "C" SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((sycl::ext::oneapi::experimental::nd_range_kernel<1>))
void add_thirty(int *ptr) {

  sycl::nd_item<1> Item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();

  sycl::id<1> GId = Item.get_global_id();
  ptr[GId.get(0)] = GId.get(0) + 30;
}
)===";

void test_1(sycl::queue &Queue, sycl::kernel &Kernel, int seed) {
  constexpr int Range = 10;
  int *usmPtr = sycl::malloc_shared<int>(Range, Queue);
  int start = 3;

  sycl::nd_range<1> R1{{Range}, {1}};

  bool Passa = true;

  memset(usmPtr, 0, Range * sizeof(int));
  Queue.submit([&](sycl::handler &Handler) {
    Handler.set_arg(0, usmPtr);
    Handler.parallel_for(R1, Kernel);
  });
  Queue.wait();

  for (int i = 0; i < Range; i++) {
    std::cout << usmPtr[i] << " ";
    assert(usmPtr[i] == i + seed);
  }
  std::cout << std::endl;

  sycl::free(usmPtr, Queue);
}

int main() {
  namespace syclex = sycl::ext::oneapi::experimental;
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
  using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;

  sycl::queue q;
  sycl::context ctx = q.get_context();
  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::sycl, SYCLSource);

  // Flags with and without space, inner quotes.
  std::vector<std::string> flags{"-Xs '-doubleGRF'",
                                 "-Xs'-Xfinalizer \"-printregusage\"'"};
  exe_kb kbExe =
      syclex::build(kbSrc, syclex::properties{syclex::build_options{flags}});

  sycl::kernel k = kbExe.ext_oneapi_get_kernel("add_thirty");

  test_1(q, k, 30);

  return 0;
}
