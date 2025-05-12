//==--- sycl_include_paths.cpp --- kernel_compiler extension tests ---------==//
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

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %S | FileCheck %s --check-prefixes=CHECK,CHECK-NOCWD

// COM: Run test again in a directory that contains a different version of
//      `header1.hpp`
// RUN: cd %S/include/C ; %{run} %t.out %S | FileCheck %s --check-prefixes=CHECK,CHECK-CWD

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

auto constexpr IncludePathsSource = R"===(
#include <sycl/sycl.hpp>

#ifdef USE_ABSOLUTE_INCLUDE

#ifndef _WIN32
#include "/tmp/sycl-rtc-end-to-end-test/header1.hpp"
#else
#include "c:/tmp/sycl-rtc-end-to-end-test/header1.hpp"
#endif

#else
#include "header1.hpp"
#include "B/header2.hpp"
#endif

extern "C" SYCL_EXTERNAL 
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(sycl::ext::oneapi::experimental::single_task_kernel)
void DEFINE_1() {}

extern "C" SYCL_EXTERNAL 
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(sycl::ext::oneapi::experimental::single_task_kernel)
void DEFINE_2() {}
)===";

static int bundleCounter = 0;
template <typename SourcePropertiesT, typename BuildPropertiesT>
void test_compilation(sycl::context &ctx, SourcePropertiesT srcProps,
                      BuildPropertiesT buildProps) {
  namespace syclex = sycl::ext::oneapi::experimental;
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
  using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;

  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::sycl, IncludePathsSource, srcProps);
  exe_kb kbExe = syclex::build(kbSrc, buildProps);

  std::cout << "bundle " << bundleCounter++ << std::endl;
  for (const std::string &name :
       {"virt1", "virt2", "fsA", "fsB", "fsC", "abs1", "abs2"}) {
    std::cout << name << ' ' << kbExe.ext_oneapi_has_kernel(name) << std::endl;
  }
}

int test_include_paths(const std::string &baseDir) {
  namespace syclex = sycl::ext::oneapi::experimental;

  sycl::queue q;
  sycl::context ctx = q.get_context();

  bool ok = q.get_device().ext_oneapi_can_build(syclex::source_language::sycl);
  if (!ok) {
    std::cout << "Apparently this device does not support `sycl` source kernel "
                 "bundle extension: "
              << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    return -1;
  }

  // Pure virtual includes
  syclex::include_files virtualIncludes;
  virtualIncludes.add("header1.hpp", "#define DEFINE_1 virt1");
  virtualIncludes.add("B/header2.hpp", "#define DEFINE_2 virt2");
  test_compilation(ctx, syclex::properties{virtualIncludes},
                   syclex::properties{});
  // CHECK-LABEL: bundle 0
  // CHECK: virt1 1
  // CHECK: virt2 1

  // File system includes (short form)
  // If CWD contains `header1.hpp` as well, that takes precence over the include
  // path
  syclex::build_options filesystemIncludes1;
  filesystemIncludes1.add("-I" + baseDir + "/include/A");
  filesystemIncludes1.add("-I");
  filesystemIncludes1.add(baseDir + "/include");
  test_compilation(ctx, syclex::properties{},
                   syclex::properties{filesystemIncludes1});
  // CHECK-LABEL: bundle 1
  // CHECK-NOCWD: fsA 1
  // CHECK-CWD:   fsA 0
  // CHECK:       fsB 1
  // CHECK-NOCWD: fsC 0
  // CHECK-CWD:   fsC 1

  // File system includes (long form)
  // Test is otherwise the same as "bundle 1"
  syclex::build_options filesystemIncludes2;
  filesystemIncludes2.add("--include-directory=" + baseDir + "/include/A");
  filesystemIncludes2.add("--include-directory");
  filesystemIncludes2.add(baseDir + "/include");
  test_compilation(ctx, syclex::properties{},
                   syclex::properties{filesystemIncludes2});
  // CHECK-LABEL: bundle 2
  // CHECK-NOCWD: fsA 1
  // CHECK-CWD:   fsA 0
  // CHECK:       fsB 1
  // CHECK-NOCWD: fsC 0
  // CHECK-CWD:   fsC 1

  // Mix virtual and filesystem includes
  // Virtual `header1.hpp` takes precedence over CWD and include paths
  syclex::include_files partialIncludes1{"header1.hpp",
                                         "#define DEFINE_1 virt1"};
  test_compilation(ctx, syclex::properties{partialIncludes1},
                   syclex::properties{filesystemIncludes1});
  // CHECK-LABEL: bundle 3
  // CHECK: virt1 1
  // CHECK: fsA 0
  // CHECK: fsB 1
  // CHECK: fsC 0

  // Virtual `header2.hpp` comes first; `header1.hpp` is included either from
  // CWD or from the include path
  syclex::include_files partialIncludes2{"B/header2.hpp",
                                         "#define DEFINE_2 virt2"};
  test_compilation(ctx, syclex::properties{partialIncludes2},
                   syclex::properties{filesystemIncludes1});
  // CHECK-LABEL: bundle 4
  // CHECK:       virt1 0
  // CHECK:       virt2 1
  // CHECK-NOCWD: fsA 1
  // CHECK-CWD:   fsA 0
  // CHECK:       fsB 0
  // CHECK-NOCWD: fsC 0
  // CHECK-CWD:   fsC 1

  // A bit silly, but including with an absolute path also works
  syclex::include_files absoluteVirtualIncludes{
#ifndef _WIN32
      "/tmp/sycl-rtc-end-to-end-test/header1.hpp",
#else
      "c:/tmp/sycl-rtc-end-to-end-test/header1.hpp",
#endif
      "#define DEFINE_1 abs1\n#define DEFINE_2 abs2"};
  syclex::build_options setDefine{"-DUSE_ABSOLUTE_INCLUDE"};
  test_compilation(ctx, syclex::properties{absoluteVirtualIncludes},
                   syclex::properties{setDefine});
  // CHECK-LABEL: bundle 5
  // CHECK: abs1 1
  // CHECK: abs2 1

  // If virtual files are defined outside of the CWD, we cannot prioritize them
  // higher than actual files in the CWD. However, this is fine (and only
  // included here for illustration), because the specification only covers the
  // situation in which the *same name* is used for the `#include` and the
  // `include_files` property, which is not the case here.
  syclex::build_options mixedIncludes;
#ifndef _WIN32
  mixedIncludes.add("-I/tmp/sycl-rtc-end-to-end-test");
#else
  mixedIncludes.add("-Ic:/tmp/sycl-rtc-end-to-end-test");
#endif
  mixedIncludes.add("-I" + baseDir + "/include");
  test_compilation(ctx, syclex::properties{absoluteVirtualIncludes},
                   syclex::properties{mixedIncludes});
  // CHECK-LABEL: bundle 6
  // CHECK:       fsB 1
  // CHECK-NOCWD: fsC 0
  // CHECK-CWD:   fsC 1
  // CHECK-NOCWD: abs1 1
  // CHECK-CWD:   abs1 0
  // CHECK:       abs2 0

  return 0;
}

int test_errors() {
  namespace syclex = sycl::ext::oneapi::experimental;
  syclex::include_files includes;
  includes.add("foo.h", "/**/");
  try {
    includes.add("foo.h", "/**/");
    assert(false && "Expected exception");
  } catch (sycl::exception &e) {
    assert(e.code() == sycl::errc::invalid && "Expected errc::invalid");
  }

  return 0;
}

int main(int argc, char **argv) {
#ifdef SYCL_EXT_ONEAPI_KERNEL_COMPILER
  assert(argc >= 2);
  return test_include_paths(argv[1]) || test_errors();
#else
  static_assert(false, "Kernel Compiler feature test macro undefined");
#endif
  return 0;
}
