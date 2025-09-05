// RUN: %{build} %cuda_options -o %t.out
// RUN: %{run} %t.out
// REQUIRES: target-nvidia, cuda_dev_kit

#include <cuda.h>
#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>
#include <vector>

int main() {
  sycl::queue q;
  sycl::context ctxt = q.get_context();
  sycl::kernel_id k_id = sycl::get_kernel_id<class mykernel>();
  auto bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctxt, {k_id});
  assert(!bundle.empty());
  sycl::kernel krn = bundle.get_kernel(k_id);
  sycl::buffer<int> buf(sycl::range<1>(1));
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor acc(buf, cgh);
    cgh.single_task<class mykernel>(krn, [=]() { acc[0] = 42; });
  });
  const auto img = *(bundle.begin());
  const auto bytes = img.ext_oneapi_get_backend_content();
  CUmodule m;
  CUresult result =
      cuModuleLoadData(&m, reinterpret_cast<const void *>(bytes.data()));
  assert(result == CUDA_SUCCESS);
  return 0;
}
