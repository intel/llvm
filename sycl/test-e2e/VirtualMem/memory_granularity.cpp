// This test checks whether memory granularity returned is greater than 0 and
// that the recommended granularity is the same size or greater than the minimum
// granularity for virtual memory extension.

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>

#include <cassert>

namespace syclext = sycl::ext::oneapi::experimental;

int main() {
  sycl::queue Q;
  sycl::context Context = Q.get_context();
  sycl::device Device = Q.get_device();

  size_t DevRecommended = syclext::get_mem_granularity(
      Device, Context, syclext::granularity_mode::recommended);
  size_t ContextRecommended = syclext::get_mem_granularity(
      Context, syclext::granularity_mode::recommended);
  size_t DevMinimum = syclext::get_mem_granularity(
      Device, Context, syclext::granularity_mode::minimum);
  size_t ContextMinimum =
      syclext::get_mem_granularity(Context, syclext::granularity_mode::minimum);

  assert(DevRecommended > 0 &&
         "recommended granularity for device should be greater than 0");
  assert(ContextRecommended > 0 &&
         "recommended granularity for context should be greater than 0");

  assert(DevMinimum > 0 &&
         "minimum granularity for device should be greater than 0");
  assert(ContextMinimum > 0 &&
         "minimum granularity for context should be greater than 0");

  assert(DevRecommended >= DevMinimum &&
         "recommended granularity size must be at least minimum granularity "
         "for device");
  assert(ContextRecommended >= ContextMinimum &&
         "recommended granularity size must be at least minimum granularity "
         "for context");

  return 0;
}
