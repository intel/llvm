// REQUIRES: aspect-ext_oneapi_virtual_mem

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <string_view>

#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>

namespace syclext = sycl::ext::oneapi::experimental;

template <typename... ArgsT>
int CheckReturnedGranularitySize(std::string_view TestName, ArgsT... Args) {
  if (syclext::get_mem_granularity(Args...) == 0) {
    std::cout << "Failed check: " << TestName << std::endl;
    return 1;
  }
  return 0;
}

int main() {
  int Failed = 0;
  sycl::queue Q;
  sycl::context Context = Q.get_context();
  sycl::device Device = Q.get_device();

  Failed += CheckReturnedGranularitySize(
      "returned granularity size must be greater then 0 for device and "
      "granularity_mode::recommended",
      Device, Context, syclext::granularity_mode::recommended);
  Failed += CheckReturnedGranularitySize(
      "returned granularity size must be greater then 0 for device and "
      "granularity_mode::minimum",
      Device, Context, syclext::granularity_mode::minimum);
  Failed += CheckReturnedGranularitySize(
      "returned granularity size must be greater then 0 for context and "
      "granularity_mode::recommended",
      Context, syclext::granularity_mode::recommended);
  Failed += CheckReturnedGranularitySize(
      "returned granularity size must be greater then 0 for context and "
      "granularity_mode::minimum",
      Context, syclext::granularity_mode::minimum);

  constexpr size_t NumberOfElements = 2;
  std::array<size_t, NumberOfElements> RecommendedGranularities = {
      syclext::get_mem_granularity(Device, Context,
                                   syclext::granularity_mode::recommended),
      syclext::get_mem_granularity(Context,
                                   syclext::granularity_mode::recommended)};

  std::array<size_t, NumberOfElements> MinimumGranularities = {
      syclext::get_mem_granularity(Device, Context,
                                   syclext::granularity_mode::minimum),
      syclext::get_mem_granularity(Context,
                                   syclext::granularity_mode::minimum)};

  std::array<std::string_view, NumberOfElements> TestNames = {
      "Failed check: granularity_mode::recommended size must be at least "
      "granularity_mode::minimum for device",
      "Failed check: granularity_mode::recommended size must be at least "
      "granularity_mode::minimum for context"};

  for (size_t Index = 0; Index < NumberOfElements; ++Index) {
    if (RecommendedGranularities[Index] < MinimumGranularities[Index]) {
      std::cout << TestNames[Index] << std::endl;
      ++Failed;
    }
  }
  return Failed;
}