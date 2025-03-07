// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/sycl.hpp>
namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

static constexpr size_t NUM = 1024;
static constexpr size_t WGSIZE = 16;

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void iota(float start, float *ptr) {
  // Get the ID of this kernel iteration.
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();

  ptr[id] = start + static_cast<float>(id);
}

int main() {
  sycl::queue q;
  sycl::context ctxt = q.get_context();

  // Get a kernel bundle that contains the free function kernel "iota".
  auto exe_bndl =
    syclexp::get_kernel_bundle<iota, sycl::bundle_state::executable>(ctxt);

  // Get a kernel object for the "iota" function from that bundle.
  sycl::kernel k_iota = exe_bndl.ext_oneapi_get_kernel<iota>();

  float *ptr = sycl::malloc_shared<float>(NUM, q);
  q.submit([&](sycl::handler &cgh) {
    // Set the values of the kernel arguments.
    cgh.set_args(3.14f, ptr);

    sycl::nd_range ndr{{NUM}, {WGSIZE}};
    cgh.parallel_for(ndr, k_iota);
  }).wait();
}
