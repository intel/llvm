// Tests using an OpenCL-C defined kernel with multiple local accessors

// REQUIRES: ocloc && (opencl || level_zero)
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

using namespace sycl;
namespace syclex = sycl::ext::oneapi::experimental;
using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;
using exe_kb = sycl::kernel_bundle<sycl::bundle_state::executable>;

auto constexpr LocalAccCLSource = R"===(
    kernel void test_la(__global int *out, __local int* local_ptr1,
                        __local int2* local_ptr2, int n) {
        __local int4 local_data[1];

        size_t gid = get_global_id(0);
        size_t lid = get_local_id(0);
        size_t wg_size = get_num_groups(0);

        local_ptr1[lid] = lid;
        local_ptr2[lid].x = n;
        local_ptr2[lid].y = wg_size;

        if (lid == 0) {
          local_data[lid] = (int4)(0xA, 0xB, 0xC, 0xD);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        int acc = local_data[0].x + local_data[0].y + local_data[0].z +
                  local_data[0].w;
        out[gid] = (local_ptr1[lid] * local_ptr2[lid].x) +
                   (local_ptr2[lid].y * acc);
    }
)===";

int main() {
  sycl::queue Queue;

  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      Queue.get_context(), syclex::source_language::opencl, LocalAccCLSource);
  exe_kb kbExe1 = syclex::build(kbSrc);
  sycl::kernel test_kernel = kbExe1.ext_oneapi_get_kernel("test_la");

  constexpr size_t Size = 1024;
  constexpr size_t LocalSize = 256;

  int32_t *Ptr = malloc_device<int32_t>(Size, Queue);

  int32_t N = 42;
  Queue
      .submit([&](handler &cgh) {
        local_accessor<int32_t, 1> acc_local1(LocalSize, cgh);
        local_accessor<sycl::int2, 1> acc_local2(LocalSize, cgh);

        cgh.set_arg(0, Ptr);
        cgh.set_arg(1, acc_local1);
        cgh.set_arg(2, acc_local2);
        cgh.set_arg(3, N);

        cgh.parallel_for(nd_range<1>(Size, LocalSize), test_kernel);
      })
      .wait();

  std::vector<int32_t> HostData(Size);
  Queue.copy(Ptr, HostData.data(), Size).wait();

  constexpr int32_t Acc = 0xA + 0xB + 0xC + 0xD;
  constexpr int32_t WorkGroups = Size / LocalSize;
  constexpr int32_t Tmp = Acc * WorkGroups;
  for (size_t i = 0; i < Size; i++) {
    int32_t local_id = i % LocalSize;
    int32_t Ref = (local_id * N) + Tmp;
    assert(HostData[i] == Ref);
  }

  sycl::free(Ptr, Queue);

  return 0;
}
