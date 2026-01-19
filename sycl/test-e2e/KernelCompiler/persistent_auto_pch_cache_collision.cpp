// We want to use %{run-unfiltered-devices} for this test, it's easier if it's
// limited to SPIR-V target.
// REQUIRES: target-spir

// %t.out arguments:
//   * path to the persistent pch cache
//   * VAL variable set in the a.hpp from which the precompiled preamble is
//     compiled
//   * `-DOPT_TO_AFFECT_HASH=` to affect the produced hash of the preamble+opts

// RUN: %{build} -o %t.out
// RUN: %{run-aux} rm -rf %t.cache1 %t.cache2

// Two normal persistent-auto-pch gen/use runs
// RUN: %{run-unfiltered-devices} %t.out %t.cache1 42 1 | FileCheck %s -DVAL=42
// RUN: %{run-unfiltered-devices} %t.out %t.cache1 42 1 | FileCheck %s -DVAL=42
// RUN: %{run-unfiltered-devices} %t.out %t.cache2 43 2 | FileCheck %s -DVAL=43
// RUN: %{run-unfiltered-devices} %t.out %t.cache2 43 2 | FileCheck %s -DVAL=43

// Content of the a.hpp changes, but auto-pch doesn't track that, so the
// precompiled preamble from the cache is reused:
// RUN: %{run-unfiltered-devices} %t.out %t.cache1 44 1 | FileCheck %s -DVAL=42

// Simulate collision - actual compilation opts stored on disk have
// "-DOPT_TO_AFFECT_HASH=2" while the hash encoded in the filename is still
// "-DOPT_TO_AFFECT_HASH=1". The "/*" below is the primary reason we're using
// %{run-unfiltered-devices} as we need to reference the cached file of which we
// don't know the name.
// RUN: %{run-aux} cp %t.cache2/* %t.cache1/*
// PCH on disk is ignored due to cache collision:
// RUN: %{run-unfiltered-devices} %t.out %t.cache1 44 1 | FileCheck %s -DVAL=44

// CHECK: Result: [[VAL]]

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

std::string getInclude(int val) {
  return
      R"""(
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/ext/oneapi/kernel_properties/properties.hpp>
inline constexpr int VAL = )""" +
      std::to_string(val) + ";\n";
}

const std::string src = R"""(
#include "a.hpp"

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;
extern "C"
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void foo(int *p) {
  *p = VAL;
}
)""";

int main(int argc, char *argv[]) {
  sycl::queue q;
  assert(argc == 4);
  std::string pch_dir = argv[1];
  int value = std::atoi(argv[2]);
  int hash_def = std::atoi(argv[3]);
  auto kb_src = syclexp::create_kernel_bundle_from_source(
      q.get_context(), syclexp::source_language::sycl, src,
      syclexp::properties{syclexp::include_files{"a.hpp", getInclude(value)}});
  auto kb_exe = syclexp::build(
      kb_src,
      syclexp::properties{syclexp::build_options{std::vector<std::string>{
          "--persistent-auto-pch=" + pch_dir,
          "-DOPT_TO_AFFECT_HASH=" + std::to_string(hash_def)}}});
  sycl::kernel krn = kb_exe.ext_oneapi_get_kernel("foo");
  auto *p = sycl::malloc_shared<int>(1, q);
  q.submit([&](sycl::handler &cgh) {
     cgh.set_args(p);
     cgh.single_task(krn);
   }).wait();
  std::cout << "Result: " << *p << std::endl;
  sycl::free(p, q);
}
