// RUN: %{build} '-DPCH_DIR="%/t.dir"' -O3 -o %t.out
// RUN: %if hip %{ env SYCL_JIT_AMDGCN_PTX_TARGET_CPU=%{amd_arch} %} %{run} %t.out

// UNSUPPORTED: target-native_cpu
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/20142

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string_view>

using namespace std::string_view_literals;
namespace syclexp = sycl::ext::oneapi::experimental;

void run(std::vector<std::string_view> ExtraHeaders) {
  std::string preamble = [&]() {
    std::stringstream preamble;

    // These are necessary:
    preamble << R"""(
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/ext/oneapi/kernel_properties/properties.hpp>
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

  auto Run = [&](auto props) {
    for (int i = 0; i < 5; ++i) {
      std::string src_str =
          preamble + "#define VAL " + std::to_string(i) + "\n" + body;
      auto t1 = std::chrono::high_resolution_clock::now();
      sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source> kb_src =
          syclexp::create_kernel_bundle_from_source(
              q.get_context(), syclexp::source_language::sycl, src_str);
      sycl::kernel_bundle<sycl::bundle_state::executable> kb_exe =
          syclexp::build(kb_src, props);
      auto t2 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> iter_duration = t2 - t1;
      std::cout << static_cast<int>(iter_duration.count()) << "ms" << " ";
    }
  };

  if (ExtraHeaders.empty())
    std::cout << "<none> ";
  for (std::string_view Header : ExtraHeaders)
    std::cout << Header << " ";
  std::cout << "| ";
  Run(syclexp::properties{});
  std::cout << "| ";
  Run(syclexp::properties{
      syclexp::build_options{std::vector<std::string>{"--auto-pch"}}});
  std::error_code ec;
  std::filesystem::remove_all(PCH_DIR, ec);

  std::cout << "| ";
  Run(syclexp::properties{syclexp::build_options{
      std::vector<std::string>{"--persistent-auto-pch=" PCH_DIR}}});
  std::cout << std::endl;
}

int main(int argc, char **argv) {
  // So that output could be copy-pasted into GH comments and rendered as a
  // table:
  std::cout << "Extra Headers | Without PCH | With Auto-PCH | With Persistent "
               "Auto-PCH"
            << std::endl;
  std::cout << "-|-|-|-" << std::endl;
  run({});
  run({"sycl/half_type.hpp"});
  run({"sycl/ext/oneapi/bfloat16.hpp"});
  run({"sycl/marray.hpp"});
  run({"sycl/vector.hpp"});
  run({"sycl/multi_ptr.hpp"});
  run({"sycl/builtins.hpp"});
  run({"sycl/ext/oneapi/matrix/matrix.hpp"});
}
