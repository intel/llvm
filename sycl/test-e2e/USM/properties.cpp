// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/ext/intel/experimental/usm_properties.hpp>
#include <sycl/usm/usm_allocator.hpp>

int main() {
  sycl::queue q;

  // Ensure properties are supported when constructing the allocator:
  sycl::usm_allocator<int, sycl::usm::alloc::shared> allocator{
      q,
      {sycl::ext::oneapi::property::usm::device_read_only{},
       sycl::ext::intel::experimental::property::usm::buffer_location{1}}};
}
