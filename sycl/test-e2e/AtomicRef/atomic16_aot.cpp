// REQUIRES: arch-intel_gpu_cri, ocloc, target-spir, linux
// RUN: mkdir -p %t.dir
// RUN: %{build} -fsycl-targets=intel_gpu_cri,intel_gpu_bmg_g31,intel_gpu_pvc -fsycl-device-code-split=per_kernel -save-temps=obj -o %t.dir/atomic16.out
// RUN: FileCheck %s --check-prefix=CRI --input-file=%t.dir/atomic16_aot-sycl-spir64_gen-unknown-unknown-cri.table
// RUN: FileCheck %s --check-prefix=OTHER --input-file=%t.dir/atomic16_aot-sycl-spir64_gen-unknown-unknown-bmg_g31.table
// RUN: FileCheck %s --check-prefix=OTHER --input-file=%t.dir/atomic16_aot-sycl-spir64_gen-unknown-unknown-pvc.table
// RUN: %{run} %t.dir/atomic16.out

// Native atomic16 kernels must survive CRI AOT filtering, but not appear in
// BMG/PVC images. No spir64 image is provided to hide a missing CRI image.
// CRI: [Code|Properties|Symbols]
// CRI-NEXT: {{.*}}.sym
// CRI-EMPTY:
// CRI-NOT: {{.}}
// OTHER: [Code|Properties|Symbols]
// OTHER-EMPTY:
// OTHER-NOT: {{.}}

#include <sycl/atomic_ref.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

#include <cassert>

int main() {
  sycl::queue Queue;
  sycl::half HalfValue = 1;
  sycl::ext::oneapi::bfloat16 BFloatValue{2};
  {
    sycl::buffer<sycl::half> HalfBuffer(&HalfValue, sycl::range<1>(1));
    sycl::buffer<sycl::ext::oneapi::bfloat16> BFloatBuffer(&BFloatValue,
                                                           sycl::range<1>(1));
    Queue
        .submit([&](sycl::handler &Handler) {
          auto Half =
              HalfBuffer.get_access<sycl::access::mode::read_write>(Handler);
          auto BFloat =
              BFloatBuffer.get_access<sycl::access::mode::read_write>(Handler);
          Handler.single_task<class Atomic16AOT>([=]() {
            sycl::atomic_ref<sycl::half, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                HalfAtomic(Half[0]);
            sycl::atomic_ref<sycl::ext::oneapi::bfloat16,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                BFloatAtomic(BFloat[0]);
            HalfAtomic.fetch_add(sycl::half{3});
            BFloatAtomic.fetch_add(sycl::ext::oneapi::bfloat16{4});
          });
        })
        .wait_and_throw();
  }
  assert(static_cast<float>(HalfValue) == 4.0f);
  assert(static_cast<float>(BFloatValue) == 6.0f);
}