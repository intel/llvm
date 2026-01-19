// RUN: %{build} -o %t.out
// RUN: %if hip %{ env SYCL_JIT_AMDGCN_PTX_TARGET_CPU=%{amd_arch} %} %{l0_leak_check} %{run} %t.out

// XFAIL: target-native_cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20142

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

#include <chrono>
#include <thread>

namespace syclexp = sycl::ext::oneapi::experimental;

static constexpr size_t NUM = 1024;
static constexpr size_t WGSIZE = 16;

int main() {
  sycl::queue q;

  std::string source = R"""(
#define MACRO_TIME __TIME__
#include "a.hpp"
#include <cstring>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/ext/oneapi/kernel_properties/properties.hpp>
namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

extern "C"
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void foo(char *macro_time, char *var_time) {
    std::strcpy(macro_time, MACRO_TIME);
    std::strcpy(var_time, VAR_TIME);
}
)""";

  auto kb_src = syclexp::create_kernel_bundle_from_source(
      q.get_context(), syclexp::source_language::sycl, source,
      syclexp::properties{syclexp::include_files{
          "a.hpp", "const char * VAR_TIME = MACRO_TIME;"}});

  auto props = syclexp::properties{
      syclexp::build_options{std::vector<std::string>{"--auto-pch"}}};

  size_t len = std::strlen("hh:mm:ss") + 1;
  char *macro_time = sycl::malloc_shared<char>(len, q);
  char *var_time = sycl::malloc_shared<char>(len, q);

  auto Run = [&](auto Prefix) {
    auto krn = syclexp::build(kb_src, props).ext_oneapi_get_kernel("foo");
    q.submit([&](sycl::handler &cgh) {
       cgh.set_args(macro_time, var_time);

       cgh.single_task(krn);
     }).wait();

    return std::pair{std::string{macro_time}, std::string{var_time}};
  };

  auto [gen_macro, gen_var] = Run("Gen");
  using namespace std::chrono_literals;
  std::this_thread::sleep_for(2s);
  auto [use_macro, use_var] = Run("Use");

  // If "captured" into a variable, the time is frozen at the moment auto-pch is
  // generated:
  assert(gen_var == use_var);

  // If it's just a `#define <..> __TIME__`, and is only "used" outside the
  // preamble, then the time matches the compilation (auto-pch use):
  assert(gen_macro != use_macro);

  sycl::free(macro_time, q);
  sycl::free(var_time, q);
}
