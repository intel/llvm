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
// RUN: env IGC_DumpToCustomDir=%T.dump IGC_ShaderDumpEnable=1 NEO_CACHE_PERSISTENT=0 %{run} %t.out %T.dump/

// clang-format off
/*
    clang++ -fsycl -o sdf.bin sycl_device_flags.cpp
    IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=./dump NEO_CACHE_PERSISTENT=0 ./sdf.bin ./dump
    
    grep -e '-doubleGRF' ./dump/OCL_asmaf99e2d4667ef6d3_options.txt
    grep -e '-Xfinalizer "-printregusage"' ./dump/OCL_asmaf99e2d4667ef6d3_options.txt

    Note: there are files named  xxx_options.txt and xxx_internal_options.txt in
    the IGC dump directory. The file with "internal_options.txt"  is NOT the
    correct file.
*/
// clang-format on

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
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

int test_dump(std::string &dump_dir) {
  // If this has been run with the shader dump environment variables set, then
  // the output files we are looking for should be in ./dump
  // There are two files whose name ends in _options. We do NOT want
  // the file that ends in _internal_options.txt

  std::string command_one =
      "find " + dump_dir +
      " -name \"*_options.txt\" -not -name \"*_internal_options.txt\" -type f "
      "-exec grep -q -e "
      "'-doubleGRF' {} +";
  std::string command_two =
      "find " + dump_dir +
      " -name \"*_options.txt\" -not -name \"*_internal_options.txt\" -type f "
      "-exec grep -q -e "
      "'-Xfinalizer \"-printregusage\"' {} +";

  // 0 means success, any other value is a failure
  int result_one = std::system(command_one.c_str());
  int result_two = std::system(command_two.c_str());

  if (result_one == 0 && result_two == 0) {
    return 0;
  } else {
    std::cout << "result_one: " << result_one << " result_two: " << result_two
              << std::endl;
    return -1;
  }
}

int main(int argc, char *argv[]) {

  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <dump_directory>" << std::endl;
    return 1;
  }
  std::string dump_dir = argv[1];

  namespace syclex = sycl::ext::oneapi::experimental;
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
  using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;

  sycl::queue q;
  sycl::context ctx = q.get_context();

  bool ok =
      q.get_device().ext_oneapi_can_compile(syclex::source_language::sycl);
  if (!ok) {
    std::cout << "compiling from SYCL source not supported" << std::endl;
    return 0; // if kernel compilation is not supported, do nothing.
  }

  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::sycl, SYCLSource);

  // Flags with and without space, inner quotes.
  std::vector<std::string> flags{"-Xs '-doubleGRF'",
                                 "-Xs'-Xfinalizer \"-printregusage\"'"};
  exe_kb kbExe =
      syclex::build(kbSrc, syclex::properties{syclex::build_options{flags}});

  sycl::kernel k = kbExe.ext_oneapi_get_kernel("add_thirty");

  test_1(q, k, 30);

  return test_dump(dump_dir);
}
