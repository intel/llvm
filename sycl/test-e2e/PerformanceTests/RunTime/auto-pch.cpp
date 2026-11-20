// RUN: %{build} '-DPCH_DIR="%/t.dir"' -O3 -o %t.out
// RUN: %if hip %{ env SYCL_JIT_AMDGCN_PTX_TARGET_CPU=%{amd_arch} %} %{run} %t.out

// UNSUPPORTED: target-native_cpu
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/20142

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>

#include "../perf_harness.hpp"

#include <filesystem>
#include <sstream>
#include <string_view>

using namespace std::string_view_literals;
namespace syclexp = sycl::ext::oneapi::experimental;

void run(perf::Benchmark &Bench, std::vector<std::string_view> ExtraHeaders) {
  std::string preamble = [&]() {
    std::stringstream preamble;

    // These are necessary:
    preamble << R"""(
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/ext/oneapi/kernel_properties.hpp>
)""";

    for (std::string_view Header : ExtraHeaders)
      preamble << "#include <" << Header << ">\n";

    preamble << "void preamble_stop();\n";
    return preamble.str();
  }();

  // Each iteration will have
  //
  //   #define VAL <iteration>
  //
  // between preamble and body

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

  sycl::queue q;

  std::string Headers = ExtraHeaders.empty() ? "<none>" : "";
  for (std::string_view Header : ExtraHeaders)
    Headers += (Headers.empty() ? "" : " ") + std::string(Header);

  // The warm-up call is the first build, so the persistent PCH is created
  // there and reused by the timed builds.
  auto Run = [&](const char *Pch, auto props) {
    int i = 0;
    Bench.add_perf_test(
        "create+build, headers=" + Headers + ", pch=" + Pch,
        [&] {
          std::string src_str =
              preamble + "#define VAL " + std::to_string(i++) + "\n" + body;
          sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source> kb_src =
              syclexp::create_kernel_bundle_from_source(
                  q.get_context(), syclexp::source_language::sycl, src_str);
          sycl::kernel_bundle<sycl::bundle_state::executable> kb_exe =
              syclexp::build(kb_src, props);
        },
        perf::Opts().warmup(1).reps(4).min_time(0));
  };

  Run("none", syclexp::properties{});
  Run("auto", syclexp::properties{syclexp::build_options{
                  std::vector<std::string>{"--auto-pch"}}});
  std::error_code ec;
  std::filesystem::remove_all(PCH_DIR, ec);

  Run("persistent",
      syclexp::properties{syclexp::build_options{
          std::vector<std::string>{"--persistent-auto-pch=" PCH_DIR}}});
}

int main(int argc, char **argv) {
  perf::Benchmark Bench(
      argc, argv, "Kernel compiler auto-PCH",
      "kernel_compiler SYCL build time per headers and PCH mode.",
      perf::Unit::ms);
  run(Bench, {});
  run(Bench, {"sycl/half_type.hpp"});
  run(Bench, {"sycl/ext/oneapi/bfloat16.hpp"});
  run(Bench, {"sycl/marray.hpp"});
  run(Bench, {"sycl/vector.hpp"});
  run(Bench, {"sycl/multi_ptr.hpp"});
  run(Bench, {"sycl/builtins.hpp"});
  run(Bench, {"sycl/ext/oneapi/matrix/matrix.hpp"});
  run(Bench, {"sycl/sycl.hpp"});
}
