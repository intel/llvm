// RUN: %{build} '-DPCH_DIR="%/t.dir"' -O3 -o %t.out
// RUN: %if hip %{ env SYCL_JIT_AMDGCN_PTX_TARGET_CPU=%{amd_arch} %} %{run} %t.out

// UNSUPPORTED: target-native_cpu
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/20142

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string_view>
#include <thread>

using namespace std::string_view_literals;
namespace syclexp = sycl::ext::oneapi::experimental;
int main() {
  sycl::queue q;
  constexpr int N = 16;

  auto Run = [&](int i) {
    std::string preamble = R"""(
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/ext/oneapi/kernel_properties/properties.hpp>
)""";

    std::string body = R"""(
namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

extern "C"
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void iota(float start, float *ptr) {
    size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
    ptr[id] = start + static_cast<float>(id) + VAL;
}
)""";

    std::string src_str = preamble +
                          "inline constexpr int VAL = " + std::to_string(i) +
                          ";\n" + body;
    auto kb_src = syclexp::create_kernel_bundle_from_source(
        q.get_context(), syclexp::source_language::sycl, src_str);
    auto kb_exe = syclexp::build(
        kb_src,
        syclexp::properties{syclexp::build_options{
            std::vector<std::string>{"--persistent-auto-pch=" PCH_DIR}}});
  };

  std::thread threads[N];

  for (int i = 0; i < N; ++i) {
    // Use noexcept overload to avoid exception if PCH_DIR doesn't exist:
    std::error_code ec;
    std::filesystem::remove_all(PCH_DIR, ec);

    threads[i] = std::thread{Run, i};
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(100ms);
  }

  for (auto &t : threads)
    if (t.joinable())
      t.join();
}
