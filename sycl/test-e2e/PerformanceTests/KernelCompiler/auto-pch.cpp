// RUN: %{build} -O3 -o %t.out
// RUN: %if hip %{ env SYCL_JIT_AMDGCN_PTX_TARGET_CPU=%{amd_arch} %} %{run} %t.out --pch-dir=%/t.dir

// UNSUPPORTED: target-native_cpu
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/20142

// PERF-DESCRIPTION: kernel_compiler SYCL build time per headers and PCH mode.

#include "../perf_harness.hpp"

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>

#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace syclexp = sycl::ext::oneapi::experimental;

enum class PCHMode { None, Auto, Persistent };

void add_cases(std::vector<std::string> ExtraHeaders) {
  std::string preamble = [&]() {
    std::stringstream preamble;

    // These are necessary:
    preamble << R"""(
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/ext/oneapi/kernel_properties.hpp>
)""";

    for (const std::string &Header : ExtraHeaders)
      preamble << "#include <" << Header << ">\n";

    preamble << "void preamble_stop();\n";
    return preamble.str();
  }();

  // Each build will have
  //
  //   #define VAL <unique number>
  //
  // between preamble and body so that in-memory caches miss.

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

  std::string Headers;
  for (const std::string &Header : ExtraHeaders)
    Headers += (Headers.empty() ? "" : " ") + Header;
  if (Headers.empty())
    Headers = "<none>";

  for (auto [Mode, ModeName] :
       {std::pair{PCHMode::None, "none"}, std::pair{PCHMode::Auto, "auto"},
        std::pair{PCHMode::Persistent, "persistent"}}) {
    perf::add_case(
        "create+build, headers=" + Headers + ", pch=" + ModeName, "ms",
        [=, Mode = Mode](perf::State &S) {
          static int Counter = 0;
          static sycl::queue q;

          std::vector<std::string> Opts;
          if (Mode == PCHMode::Auto)
            Opts.push_back("--auto-pch");
          if (Mode == PCHMode::Persistent) {
            std::string Dir = perf::arg("pch-dir");
            // Start from an empty directory so the first (warm-up) build
            // creates the PCH and the timed ones reuse it.
            if (S.iteration() == 0) {
              std::error_code ec;
              std::filesystem::remove_all(Dir, ec);
            }
            Opts.push_back("--persistent-auto-pch=" + Dir);
          }

          std::string src_str = preamble + "#define VAL " +
                                std::to_string(Counter++) + "\n" + body;
          // Kept alive past measure() so destruction isn't timed.
          std::optional<
              sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>>
              kb_src;
          std::optional<sycl::kernel_bundle<sycl::bundle_state::executable>>
              kb_exe;
          S.measure([&] {
            kb_src = syclexp::create_kernel_bundle_from_source(
                q.get_context(), syclexp::source_language::sycl, src_str);
            kb_exe = syclexp::build(
                *kb_src, syclexp::properties{syclexp::build_options{Opts}});
          });
          perf::check(kb_exe->ext_oneapi_has_kernel("iota"),
                      "built bundle has kernel iota");
        },
        perf::Options{1, 5});
  }
}

int main(int argc, char **argv) {
  add_cases({});
  add_cases({"sycl/half_type.hpp"});
  add_cases({"sycl/ext/oneapi/bfloat16.hpp"});
  add_cases({"sycl/marray.hpp"});
  add_cases({"sycl/vector.hpp"});
  add_cases({"sycl/multi_ptr.hpp"});
  add_cases({"sycl/builtins.hpp"});
  add_cases({"sycl/ext/oneapi/matrix/matrix.hpp"});
  add_cases({"sycl/sycl.hpp"});

  return perf::main(argc, argv);
}
