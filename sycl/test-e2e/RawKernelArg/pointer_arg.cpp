// REQUIRES: aspect-usm_shared_allocations
// REQUIRES: ocloc && (opencl || level_zero)

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests the pointer form of raw_kernel_arg, which says that an argument is a
// pointer instead of leaving the runtime to bind the bytes it is made of. That
// is what makes a pointer argument reach the kernel on a backend which takes a
// pointer through a different entry point than a value: OpenCL passes a value
// argument to clSetKernelArg, which rejects a USM pointer, and Native CPU puts
// the address of its own copy of the bytes into the argument slot. Hence no
// Level Zero requirement here, unlike the tests of the byte form.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/raw_kernel_arg.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

#include <cassert>

namespace oneapiext = sycl::ext::oneapi::experimental;

auto constexpr CLSource = R"===(
__kernel void WriteScalar(int in, __global int *out) {
  out[get_global_id(0)] = in;
}
)===";

constexpr size_t N = 8;

int main() {
  sycl::queue Q;

  auto SourceKB = oneapiext::create_kernel_bundle_from_source(
      Q.get_context(), oneapiext::source_language::opencl, CLSource);
  auto ExecKB = oneapiext::build(SourceKB);
  sycl::kernel Kernel = ExecKB.ext_oneapi_get_kernel("WriteScalar");

  int *Out = sycl::malloc_shared<int>(N, Q);
  int In = 42;

  // Both arguments are raw, the pointer as a pointer and the scalar as bytes.
  // The queue is out of order, so the sentinel has to be waited for rather than
  // left to race with the kernel that overwrites it.
  Q.memset(Out, 0xFF, N * sizeof(int)).wait();
  Q.submit([&](sycl::handler &CGH) {
     CGH.set_arg(0, oneapiext::raw_kernel_arg{&In, sizeof(In)});
     CGH.set_arg(1, oneapiext::raw_kernel_arg{&Out, oneapiext::pointer_arg});
     CGH.parallel_for(sycl::range<1>{N}, Kernel);
   }).wait();
  for (size_t I = 0; I < N; ++I)
    assert(Out[I] == In);

  // The same through set_args, and with the pointer bound to a different
  // allocation, so that a stale argument would show up.
  int *Other = sycl::malloc_shared<int>(N, Q);
  In = 7;
  Q.memset(Other, 0xFF, N * sizeof(int)).wait();
  Q.submit([&](sycl::handler &CGH) {
     CGH.set_args(oneapiext::raw_kernel_arg{&In, sizeof(In)},
                  oneapiext::raw_kernel_arg{&Other, oneapiext::pointer_arg});
     CGH.parallel_for(sycl::range<1>{N}, Kernel);
   }).wait();
  for (size_t I = 0; I < N; ++I)
    assert(Other[I] == In);

  sycl::free(Other, Q);
  sycl::free(Out, Q);
  return 0;
}
