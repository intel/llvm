#include <sycl/sycl.hpp>
namespace syclex = sycl::ext::oneapi::experimental;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclex::nd_range_kernel<1>))
void mykernel(size_t *data) {
  // Get a handle to the root-group.
  auto root = syclex::this_work_item::get_root_group<1>();

  // Write to some global memory location.
  data[root.get_local_linear_id()] = root.get_local_linear_id();

  // Synchronize all work-items executing the kernel, making all writes visible.
  sycl::group_barrier(root);
}

int main() {
  sycl::queue q;

  // When a kernel uses root-group synchronization, the total number of
  // work-groups is limited.  This limit is specific to the kernel, the device,
  // the work-group size (which is 32 in this example), the launch parameters,
  // and the amount of dynamically allocated work-group local memory (which is
  // zero in this example).
  syclex::properties props{syclex::use_root_sync};
  auto maxWGs = syclex::get_kernel_info<mykernel,
    syclex::info::kernel::max_num_work_groups_sync>(q, 32, props, 0);

  // Construct an nd-range which launches the maximum number of work-groups.
  auto ndr = sycl::nd_range<1>{maxWGs * 32, 32};

  // When a kernel uses root-group synchronization, it must be launched with the
  // "use_root_sync" property.  Construct a launch configuration with this
  // property.
  syclex::launch_config cfg{ndr, props};

  size_t *data = sycl::malloc_device<size_t>(maxWGs * 32, q);
  syclex::nd_launch(q, cfg, syclex::kernel_function<mykernel>, data);
  syclex::nd_launch(q, cfg, syclex::kernel_function<mykernel>, data);
  q.wait();
}
