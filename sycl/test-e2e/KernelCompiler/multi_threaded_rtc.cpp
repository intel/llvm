// RUN: %{build} '-DPCH_DIR="%/t.dir"' -o %t.out
// RUN: %if hip %{ env SYCL_JIT_AMDGCN_PTX_TARGET_CPU=%{amd_arch} %} %{run} %t.out

// UNSUPPORTED: target-native_cpu
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/20142

// Verify that parallel compilations work.

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>

#include <filesystem>
#include <thread>

namespace syclexp = sycl::ext::oneapi::experimental;
int main() {
  std::error_code ec;
  std::filesystem::remove_all(PCH_DIR, ec); // noexcept overload
  sycl::queue q;
  constexpr int N = 16;
  std::string src_str = R"""(
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

  auto Run = [&](auto... args) {
    auto kb_src = syclexp::create_kernel_bundle_from_source(
        q.get_context(), syclexp::source_language::sycl, src_str);
    auto kb_exe = syclexp::build(kb_src, args...);
  };

  std::thread threads[N];

  for (auto &t : threads)
    t = std::thread{Run};
  for (auto &t : threads)
    t.join();

  auto auto_pch = syclexp::properties{
      syclexp::build_options{std::vector<std::string>{"--auto-pch"}}};

  for (auto &t : threads)
    t = std::thread{Run, auto_pch};
  for (auto &t : threads)
    t.join();

  auto persistent_auto_pch = syclexp::properties{syclexp::build_options{
      std::vector<std::string>{"--persistent-auto-pch=" PCH_DIR}}};

  for (auto &t : threads)
    t = std::thread{Run, persistent_auto_pch};
  for (auto &t : threads)
    t.join();
}
