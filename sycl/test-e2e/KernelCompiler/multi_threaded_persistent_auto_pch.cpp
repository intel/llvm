// RUN: %{build} '-DPCH_DIR="%/t.dir"' -O3 -o %t.out
// RUN: %if hip %{ env SYCL_JIT_AMDGCN_PTX_TARGET_CPU=%{amd_arch} %} %{run} %t.out

// Test sudden removal of the persistent PCH cache from the file system.

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
  std::error_code ec; // For noexcept overload.
  std::filesystem::remove_all(PCH_DIR, ec);
  sycl::queue q;
  constexpr int N = 16;
  std::chrono::duration<double, std::milli> durations[N];
  std::atomic_bool compile_finished[N];

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
    auto t1 = std::chrono::high_resolution_clock::now();
    auto kb_src = syclexp::create_kernel_bundle_from_source(
        q.get_context(), syclexp::source_language::sycl, src_str);
    auto kb_exe = syclexp::build(
        kb_src,
        syclexp::properties{syclexp::build_options{
            std::vector<std::string>{"--persistent-auto-pch=" PCH_DIR}}});
    auto t2 = std::chrono::high_resolution_clock::now();
    durations[i] = t2 - t1;
    compile_finished[i].store(true);
  };

  std::thread threads[N];

  int removed_iter = -1;
  for (int i = 0; i < N; ++i) {
    if (compile_finished[N / 2].load() && removed_iter == -1) {
      std::filesystem::remove_all(PCH_DIR, ec);
      removed_iter = i;
    }

    threads[i] = std::thread{Run, i};
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(100ms);
  }

  if (removed_iter == -1) {
    std::cout << "Removal has not been tested, adjust N" << std::endl;
    return 1;
  }
  if (removed_iter == 0) {
    std::cout << "Sleep is too long" << std::endl;
    return 2;
  }

  for (auto &t : threads)
    if (t.joinable())
      t.join();

  // Not necessary to verify stability, but provides useful data on how
  // persistent-auto-pch works in multi-threaded environment. Produces something
  // like
  //
  //   BIG
  //   ..
  //   BIG
  //   SMALL
  //   ..
  //   SMALL
  //   Cache cleared
  //   BIG
  //   ..
  //   BIG
  //   SMALL
  //   ..
  //   SMALL
  for (int i = 0; i < N; ++i) {
    if (i == removed_iter)
      std::cout << "Cache cleared" << std::endl;
    std::cout << i << ": " << static_cast<int>(durations[i].count()) << "ms"
              << std::endl;
  }
}
