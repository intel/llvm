// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Device globals aren't supported on opencl:gpu yet.
// UNSUPPORTED: opencl && gpu

// TODO: Fails at JIT compilation for some reason.
// TODO: Reenable windows/linux, see https://github.com/intel/llvm/issues/14598
// UNSUPPORTED: hip, windows, linux
#define SYCL_FALLBACK_ASSERT 1

#include <sycl/detail/core.hpp>

#include <sycl/builtins.hpp>
#include <sycl/detail/spirv.hpp>
#include <sycl/ext/oneapi/experimental/ballot_group.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/ext/oneapi/sub_group_mask.hpp>

// DeviceGlobalUSMMem::~DeviceGlobalUSMMem() has asserts to ensure some
// resources have been cleaned up when it's executed. Those asserts used to fail
// when "AssertHappened" buffer used in fallback implementation of the device
// assert was a data member of the queue_impl.
sycl::ext::oneapi::experimental::device_global<int32_t> dg;

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &cgh) {
     sycl::range<1> R{16};
     cgh.parallel_for(sycl::nd_range<1>{R, R}, [=](sycl::nd_item<1> ndi) {
       if (ndi.get_global_linear_id() == 0)
         dg.get() = 42;
       auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
       auto active = sycl::ext::oneapi::group_ballot(sg, 1);
     });
   }).wait();

  return 0;
}
