#include <sycl/sycl.hpp>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace syclexp = sycl::ext::oneapi::experimental;

struct ProbeResult {
  std::string variant;
  std::string kernel_name;
  bool has_kernel = false;
  bool got_kernel = false;
  std::string build_log;
  std::string error;
};

static std::string make_kernel_source(const std::string &field_type,
                                      const std::string &field_check,
                                      const std::string &variant_suffix) {
  std::ostringstream source;
  source << R"(// Minimal source-kernel probe for by-value struct arguments.
#include <sycl/sycl.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

struct MyCoolStruct {
  float a;
)";
  source << "  " << field_type << " b;\n";
  source << R"(  unsigned int c;
  size_t d;
};

extern "C" {

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void )";
  source << "check_device_repr_" << variant_suffix;
  source << R"((MyCoolStruct thing, unsigned long long *status) {
  const size_t i =
      syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  if (i != 0) {
    return;
  }

  status[0] = (thing.a == 1.0f && thing.b == )";
  source << field_check;
  source << R"( && thing.c == 57u &&
               thing.d == static_cast<size_t>(420))
                  ? 1ull
                  : 0ull;
}

}
)";
  return source.str();
}

static ProbeResult probe_variant(const sycl::context &context,
                                 const sycl::device &device,
                                 const std::string &field_type,
                                 const std::string &field_check,
                                 const std::string &variant_suffix) {
  ProbeResult result;
  result.variant = field_type;
  result.kernel_name = "check_device_repr_" + variant_suffix;

  auto source = make_kernel_source(field_type, field_check, variant_suffix);
  auto source_bundle = syclexp::create_kernel_bundle_from_source(
      context, syclexp::source_language::sycl, source);

  std::vector<sycl::device> devices{device};
  auto executable_bundle =
      syclexp::build(source_bundle, devices,
                     syclexp::properties{syclexp::save_log(&result.build_log)});

  result.has_kernel =
      executable_bundle.ext_oneapi_has_kernel(result.kernel_name);

  try {
    (void)executable_bundle.ext_oneapi_get_kernel(result.kernel_name);
    result.got_kernel = true;
  } catch (const std::exception &error) {
    result.error = error.what();
  }

  return result;
}

static void print_result(const ProbeResult &result) {
  std::cout << "variant: " << result.variant << '\n';
  std::cout << "  kernel: " << result.kernel_name << '\n';
  std::cout << "  has_kernel: " << (result.has_kernel ? "true" : "false")
            << '\n';
  std::cout << "  get_kernel: " << (result.got_kernel ? "ok" : "failed")
            << '\n';
  if (!result.error.empty()) {
    std::cout << "  error: " << result.error << '\n';
  }
  if (!result.build_log.empty()) {
    std::cout << "  build_log:\n" << result.build_log << '\n';
  }
}

int main() {
  try {
    sycl::queue queue{sycl::default_selector_v};
    const auto &device = queue.get_device();
    const auto &context = queue.get_context();

    std::cout << "device: " << device.get_info<sycl::info::device::name>()
              << "\n\n";

    const auto ull_result =
        probe_variant(context, device, "unsigned long long", "234ull", "ull");
    const auto double_result =
        probe_variant(context, device, "double", "234.0", "double");

    print_result(ull_result);
    std::cout << '\n';
    print_result(double_result);
    std::cout << '\n';

    const bool reproduced = ull_result.has_kernel && ull_result.got_kernel &&
                            !double_result.has_kernel &&
                            !double_result.got_kernel;

    if (!reproduced) {
      std::cerr << "Unexpected result: the double variant did not reproduce "
                   "the source-kernel lookup failure.\n";
      return 1;
    }

    std::cout << "Reproduced: a by-value struct containing double prevents "
                 "source-kernel lookup, while unsigned long long works.\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "fatal error: " << error.what() << '\n';
    return 2;
  }
}
