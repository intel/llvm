// We want to use %{run-unfiltered-devices} for this test, it's easier if it's
// limited to SPIR-V target.
// REQUIRES: target-spir

// PCH_DIR needs to be the same between build/run, so use %{run-aux}
// extensively.

// RUN: %{run-aux} %{build} '-DPCH_DIR="%/t.dir"' -o %t.out
// RUN: %{run-aux} rm -rf %t.dir

// Generate:
// RUN: %{run-unfiltered-devices} %t.out

// Use:
// RUN: %{run-unfiltered-devices} %t.out

// File too small:
// RUN: %{run-aux} echo "1" > %t.dir/*
// RUN: %{run-unfiltered-devices} %t.out

// Cache file has garbage:
// RUN: %{run-aux} cp %s %t.dir/*
// RUN: %{run-unfiltered-devices} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

const std::string src = R"""(
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/ext/oneapi/kernel_properties/properties.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;
extern "C"
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void foo(int *p) {
  *p = 42;
}
)""";

int main() {
  sycl::queue q;
  auto kb_src = syclexp::create_kernel_bundle_from_source(
      q.get_context(), syclexp::source_language::sycl, src);
  auto kb_exe = syclexp::build(
      kb_src, syclexp::properties{syclexp::build_options{
                  std::vector<std::string>{"--persistent-auto-pch=" PCH_DIR}}});
}
