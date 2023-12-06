// Use explicit spir64 triple as not yet supported on CUDA/HIP.
// RUN: %clangxx -fsycl-device-only -fsycl-targets=spir64-unknown-unknown -S -Xclang -no-enable-noundef-analysis %s -o - | FileCheck %s

#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/sycl.hpp>

using sycl::ext::oneapi::bfloat16;

SYCL_EXTERNAL uint16_t some_bf16_intrinsic(uint16_t x, uint16_t y);
SYCL_EXTERNAL void foo(long x, sycl::half y);

__attribute__((noinline)) float op(float a, float b) {
  // CHECK: define {{.*}} spir_func float @_Z2opff(float [[a:%.*]], float [[b:%.*]])
  bfloat16 A{a};
  // CHECK: [[A:%.*]] = call spir_func zeroext i16 @__devicelib_ConvertFToBF16INTEL(ptr {{.*}})
  // CHECK-NOT: fptoui

  bfloat16 B{b};
  // CHECK: [[B:%.*]] = call spir_func zeroext i16 @__devicelib_ConvertFToBF16INTEL(ptr {{.*}})
  // CHECK-NOT: fptoui

  bfloat16 C = A + B;
  // CHECK: store i16
  // CHECK: [[RTCASTI:%.*]] = addrspacecast ptr [[RT:%.*]] to ptr addrspace(4)
  // CHECK: [[A_float:%.*]] = call spir_func float @__devicelib_ConvertBF16ToFINTEL(ptr {{.*}})
  // CHECK: [[B_float:%.*]] = call spir_func float @__devicelib_ConvertBF16ToFINTEL(ptr {{.*}})
  // CHECK: [[Add:%.*]] = fadd float [[A_float]], [[B_float]]
  // CHECK: store float [[Add]], ptr [[RT]], align 4
  // CHECK: [[C:%.*]] = call spir_func zeroext i16 @__devicelib_ConvertFToBF16INTEL(ptr {{.*}}) [[RTCASTI]])

  // CHECK-NOT: uitofp
  // CHECK-NOT: fptoui

  long L = bfloat16(3.14f);
  // CHECK: [[L:%.*]] = call spir_func zeroext i16 @__devicelib_ConvertFToBF16INTEL(ptr {{.*}})
  // CHECK: store i16 [[L]]
  // CHECK: [[L_float:%.*]] = call spir_func float @__devicelib_ConvertBF16ToFINTEL(ptr
  // CHECK: [[L:%.*]] = fptosi float [[L_float]] to i{{32|64}}

  sycl::half H = bfloat16(2.71f);
  // CHECK: [[H:%.*]] = call spir_func zeroext i16 @__devicelib_ConvertFToBF16INTEL(ptr {{.*}})
  // CHECK: store i16 [[H]], ptr {{.*}}, align 2
  // CHECK: [[H_float:%.*]] = call spir_func float @__devicelib_ConvertBF16ToFINTEL(ptr
  // CHECK: [[H:%.*]] = fptrunc float [[H_float]] to half
  foo(L, H);

  return A;
  // CHECK: [[RetVal:%.*]] = call spir_func float @__devicelib_ConvertBF16ToFINTEL(ptr {{.*}})
  // CHECK: ret float [[RetVal]]
  // CHECK-NOT: uitofp
  // CHECK-NOT: fptoui
}

int main(int argc, char *argv[]) {
  float data[3] = {7.0, 8.1, 0.0};
  sycl::queue deviceQueue;
  sycl::buffer<float, 1> buf{data, sycl::range<1>{3}};

  deviceQueue.submit([&](sycl::handler &cgh) {
    auto numbers = buf.get_access<sycl::access::mode::read_write>(cgh);
    cgh.single_task<class simple_kernel>(
        [=]() { numbers[2] = op(numbers[0], numbers[1]); });
  });
  return 0;
}
