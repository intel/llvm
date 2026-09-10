// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests the nd_launch overloads that take an already built sycl::kernel and its
// arguments as a parameter pack, which bypass the handler when every argument
// can be bound as plain bytes. Arguments that the scheduler has to track, an
// accessor here, must still reach the kernel through the command group path.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/raw_kernel_arg.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/ext/oneapi/work_group_scratch_memory.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include "common.hpp"

namespace syclext = sycl::ext::oneapi;

constexpr size_t N = 1024;
constexpr size_t WGSize = 8;

enum class Sign : int { Plus = 1 };

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((oneapiext::nd_range_kernel<1>))
void addScalars(int *Ptr, int A, int B) {
  size_t I = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  Ptr[I] += A + B;
}

// A mixture of argument kinds, so that a wrong size or a wrong order shows up
// as a wrong result rather than as a silent pass.
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((oneapiext::nd_range_kernel<1>))
void addMixed(int *Ptr, int A, unsigned long B, float C, Sign S) {
  size_t I = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  Ptr[I] +=
      static_cast<int>(S) * (A + static_cast<int>(B) + static_cast<int>(C));
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((oneapiext::nd_range_kernel<1>))
void addViaAccessor(sycl::accessor<int, 1> Acc, int A) {
  size_t I = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  Acc[I] += A;
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((oneapiext::nd_range_kernel<1>))
void usesScratch(int *Ptr) {
  size_t I = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  int *Scratch =
      reinterpret_cast<int *>(oneapiext::get_work_group_scratch_memory());
  Scratch[I % WGSize] = static_cast<int>(I);
  Ptr[I] = Scratch[I % WGSize];
}

template <auto *Func> sycl::kernel getKernel(sycl::queue &Q) {
  auto Bundle =
      oneapiext::get_kernel_bundle<Func, sycl::bundle_state::executable>(
          Q.get_context());
  return Bundle.template ext_oneapi_get_kernel<Func>();
}

int main() {
  sycl::queue Q{sycl::property::queue::in_order{}};
  int *Memory = sycl::malloc_shared<int>(N, Q);
  sycl::nd_range<1> Ndr{sycl::range<1>{N}, sycl::range<1>{WGSize}};

  int Failed = 0;

  // Typed arguments: no handler is created for these.
  sycl::kernel ScalarsKernel = getKernel<addScalars>(Q);
  Q.memset(Memory, 0, N * sizeof(int));
  oneapiext::nd_launch(Q, Ndr, ScalarsKernel, Memory, 3, 4);
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 7, I, "typed arguments");

  // The same scalar arguments as raw bytes, which is how a caller that only
  // knows the signature as sizes has to pass them. The pointer stays typed:
  // `raw_kernel_arg` always binds as plain bytes, and a USM pointer bound that
  // way only reaches the kernel on Level Zero, hence the separate
  // nd_launch_kernel_obj_direct_raw_ptr.cpp for that case.
  int A = 10, B = 20;
  Q.memset(Memory, 0, N * sizeof(int));
  oneapiext::nd_launch(Q, Ndr, ScalarsKernel, Memory,
                       oneapiext::raw_kernel_arg{&A, sizeof(A)},
                       oneapiext::raw_kernel_arg{&B, sizeof(B)});
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 30, I, "raw_kernel_arg arguments");

  // Mixed argument kinds, including an enum and a float.
  Q.memset(Memory, 0, N * sizeof(int));
  oneapiext::nd_launch(Q, Ndr, getKernel<addMixed>(Q), Memory, 1, 2ul, 3.0f,
                       Sign::Plus);
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 6, I, "mixed argument kinds");

  // Ordering against the preceding commands on an in-order queue has to hold
  // for a launch that bypasses the scheduler.
  Q.memset(Memory, 0, N * sizeof(int));
  for (int I = 0; I < 8; ++I)
    oneapiext::nd_launch(Q, Ndr, ScalarsKernel, Memory, 1, 0);
  Q.wait();
  for (size_t I = 0; I < N; ++I)
    Failed += Check(Memory, 8, I, "in-order accumulation");

  // An accessor cannot be bound as plain bytes, so this has to fall back to the
  // command group path and still produce the right result.
  {
    std::vector<int> Data(N, 5);
    sycl::buffer<int, 1> Buf{Data.data(), sycl::range<1>{N}};
    sycl::kernel AccessorKernel = getKernel<addViaAccessor>(Q);
    Q.submit([&](sycl::handler &CGH) {
      sycl::accessor Acc{Buf, CGH, sycl::read_write};
      oneapiext::nd_launch(CGH, Ndr, AccessorKernel, Acc, 4);
    });
    Q.wait();
    sycl::host_accessor Host{Buf};
    for (size_t I = 0; I < N; ++I)
      Failed +=
          Check(&Host[0], 9, I, "accessor through the command group path");
  }

  // These overloads cannot carry a work_group_scratch_size property, so a
  // kernel that allocates work group scratch memory has to be reported rather
  // than launched, exactly as the command group path reports it.
  {
    bool Reported = false;
    try {
      oneapiext::nd_launch(Q, Ndr, getKernel<usesScratch>(Q), Memory);
      Q.wait_and_throw();
    } catch (const sycl::exception &E) {
      Reported = E.code() == sycl::errc::memory_allocation;
    }
    if (!Reported) {
      std::cout << "Failed: work group scratch memory without a size property "
                   "was not reported"
                << std::endl;
      ++Failed;
    }
  }

  sycl::free(Memory, Q);
  return Failed;
}
