// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s
#include <sycl/sycl.hpp>

using namespace sycl;
namespace syclexp = sycl::ext::oneapi::experimental;

// This test checks that a diagnostic is emitted when
// instantiating a work group memory class with the properties set to anything
// other than empty_properties_t

template <typename PropertyListT, typename... PropertyListTs>
void test_properties() {
  // expected-error-re@sycl/ext/oneapi/experimental/work_group_memory.hpp:* 2{{static assertion failed due to requirement 'std::is_same_v<{{.*}}, sycl::ext::oneapi::experimental::properties<sycl::ext::oneapi::experimental::detail::properties_type_list<>>>'}}
  syclexp::work_group_memory<int, PropertyListT>{syclexp::indeterminate};
  if constexpr (sizeof...(PropertyListTs))
    test_properties<PropertyListTs...>();
}

int main() {
  test_properties<int, syclexp::work_group_progress_key>();
  return 0;
}
