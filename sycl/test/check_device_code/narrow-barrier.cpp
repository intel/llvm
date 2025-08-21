// RUN: %clangxx -fsycl-device-only -fsycl-unnamed-lambda -S -Xclang -emit-llvm -Xclang -no-enable-noundef-analysis -O2 %s -o - | FileCheck %s

// The test checks if SYCLOptimizeBarriers pass can perform barrier scope
// narrowing in case if there are no fenced global accesses.

// CHECK: tail call spir_func void @_Z22__spirv_ControlBarrieriii(i32{{.*}}2, i32{{.*}}2, i32{{.*}}912)
// CHECK: tail call spir_func void @_Z22__spirv_ControlBarrieriii(i32{{.*}}2, i32{{.*}}2, i32{{.*}}400)

#include <sycl/sycl.hpp>

constexpr size_t WORK_GROUP_SIZE = 1024;
constexpr size_t NUMBER_OF_WORK_GROUPS = 64;
constexpr size_t NUMBER_OF_ITERATIONS = 100;

struct GroupBarrierKernel {

  GroupBarrierKernel(sycl::handler &h, float *sum)
      : sum(sum), local(WORK_GROUP_SIZE, h) {}

  void operator()(sycl::nd_item<1> it) const {

    const size_t item_id = it.get_local_id()[0];
    const size_t item_range = it.get_local_range()[0];
    const size_t group_id = it.get_group().get_group_id()[0];

    for (int i = 0; i < item_id; i += item_range) {
      local[i] = i;
    }

    sycl::group_barrier(it.get_group());
    for (int offset = 1; offset < item_range; offset *= 2) {
      local[item_id] += local[item_id + offset];
      sycl::group_barrier(it.get_group());
    }

    if (it.get_group().leader()) {
      sycl::group_barrier(it.get_group());
      sum[group_id] = local[0];
    }
  }

  float *sum;
  sycl::local_accessor<float> local;
};

int main(int argc, char *argv[]) {
  sycl::queue q{sycl::property::queue::enable_profiling{}};
  float *sum = sycl::malloc_shared<float>(NUMBER_OF_WORK_GROUPS, q);

  double modern_ns = 0;
  for (int r = 0; r < NUMBER_OF_ITERATIONS + 1; ++r) {
    sycl::event e = q.submit([&](sycl::handler &h) {
      auto k = GroupBarrierKernel(h, sum);
      h.parallel_for(sycl::nd_range<1>{NUMBER_OF_WORK_GROUPS * WORK_GROUP_SIZE,
                                       WORK_GROUP_SIZE},
                     k);
    });
    e.wait();
  }
}
