// RUN: %{build} -fsycl-embed-ir -o %t.out
// RUN: env SYCL_RT_WARNING_LEVEL=2 %{run} %t.out
// XFAIL: hip

// COM: Test fails on hip due to unsupported CG kinds being tested. This test
// only checks fusion does not crash on non-kernel CG (target independent test),
// so having multiple CG kinds has higher priority than running the test on all
// backends.

// Test non-kernel device command groups are not fused

#include "sycl/detail/pi.h"
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  constexpr size_t dataSize = 512;
  constexpr float Pattern{10};

  queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};
  ext::codeplay::experimental::fusion_wrapper fw(q);

  constexpr size_t count = 64;
  auto *dst = malloc_device<float>(count, q);
  auto *src = malloc_device<float>(count, q);

  {
    // CHECK: Not fusing 'copy acc to ptr' command group. Can only fuse device kernel command groups
    buffer<float> src(dataSize);
    std::shared_ptr<float> dst(new float[dataSize]);
    fw.start_fusion();
    q.submit([&](handler &cgh) {
      accessor acc(src, cgh, read_only);
      cgh.copy(acc, dst);
    });
    fw.complete_fusion();
  }

  {
    // CHECK: Not fusing 'copy ptr to acc' command group. Can only fuse device kernel command groups
    buffer<float> dst(dataSize);
    std::shared_ptr<float> src(new float[dataSize]);
    fw.start_fusion();
    q.submit([&](handler &cgh) {
      accessor acc(dst, cgh, write_only);
      cgh.copy(src, acc);
    });
    fw.complete_fusion();
  }

  {
    // CHECK: Not fusing 'copy acc to acc' command group. Can only fuse device kernel command groups
    buffer<float> dst(dataSize);
    buffer<float> src(dataSize);
    fw.start_fusion();
    q.submit([&](handler &cgh) {
      accessor acc0(src, cgh, read_only);
      accessor acc1(dst, cgh, write_only);
      cgh.copy(acc0, acc1);
    });
    fw.complete_fusion();
  }

  {
    // CHECK: Not fusing 'barrier' command group. Can only fuse device kernel command groups
    fw.start_fusion();
    q.submit([&](handler &cgh) { cgh.ext_oneapi_barrier(); });
    fw.complete_fusion();
  }

  {
    // CHECK: Not fusing 'barrier waitlist' command group. Can only fuse device kernel command groups
    buffer<float> dst(dataSize);
    buffer<float> src(dataSize);
    std::vector<event> event_list;
    event_list.push_back(q.submit([&](handler &cgh) {
      accessor acc0(src, cgh, read_only);
      accessor acc1(dst, cgh, write_only);
      cgh.copy(acc0, acc1);
    }));
    fw.start_fusion();
    q.submit([&](handler &cgh) { cgh.ext_oneapi_barrier(event_list); });
    fw.complete_fusion();
  }

  {
    // CHECK: Not fusing 'fill' command group. Can only fuse device kernel command groups
    buffer<float> dst(dataSize);
    fw.start_fusion();
    q.submit([&](handler &cgh) {
      accessor acc(dst, cgh, write_only);
      cgh.fill(acc, Pattern);
    });
    fw.complete_fusion();
  }

  {
    // CHECK: Not fusing 'copy usm' command group. Can only fuse device kernel command groups
    fw.start_fusion();
    q.submit([&](handler &cgh) { cgh.memcpy(dst, src, count); });
    fw.complete_fusion();
  }

  {
    // CHECK: Not fusing 'fill usm' command group. Can only fuse device kernel command groups
    fw.start_fusion();
    q.submit([&](handler &cgh) { cgh.fill(dst, Pattern, count); });
    fw.complete_fusion();
  }

  {
    // CHECK: Not fusing 'prefetch usm' command group. Can only fuse device kernel command groups
    fw.start_fusion();
    q.submit([&](handler &cgh) { cgh.prefetch(dst, count); });
    fw.complete_fusion();
  }

  free(src, q);
  free(dst, q);
}
