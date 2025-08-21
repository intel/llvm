// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/named_sub_group_sizes.hpp>
#include <sycl/sub_group.hpp>
#include <sycl/usm.hpp>

struct SGSizePrimaryKernelFunctor {
  SGSizePrimaryKernelFunctor(uint32_t *OutPtr) : Out{OutPtr} {}

  void operator()(sycl::nd_item<1> Item) const {
    *Out = Item.get_sub_group().get_max_local_range()[0];
  }

  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::sub_group_size_primary};
  }

  uint32_t *Out;
};

int main() {
  sycl::queue Q;

  uint32_t *OutPtr = sycl::malloc_shared<uint32_t>(1, Q);
  Q.parallel_for(sycl::nd_range<1>{1, 1}, SGSizePrimaryKernelFunctor{OutPtr})
      .wait();

  assert(*OutPtr ==
         Q.get_device().get_info<sycl::info::device::primary_sub_group_size>());
  return 0;
}
