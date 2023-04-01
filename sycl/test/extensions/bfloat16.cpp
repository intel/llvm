// RUN: %clangxx -fsycl-device-only -fsycl-targets=%sycl_triple -S -Xclang -no-enable-noundef-analysis %s -o - | FileCheck %s

// UNSUPPORTED: cuda || hip_amd

#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/sycl.hpp>

using sycl::ext::oneapi::bfloat16;

SYCL_EXTERNAL uint16_t some_bf16_intrinsic(uint16_t x, uint16_t y);
SYCL_EXTERNAL void foo(long x, sycl::half y);

__attribute__((noinline)) float op(float a, float b) {
  // CHECK: define {{.*}} spir_func float @_Z2opff(float [[a:%.*]], float [[b:%.*]])
  bfloat16 A{a};
  // CHECK: [[A:%.*]] = call spir_func zeroext i16 @__devicelib_ConvertFToBF16INTEL(float {{.*}})
  // CHECK-NOT: fptoui

  bfloat16 B{b};
  // CHECK: [[B:%.*]] = call spir_func zeroext i16 @__devicelib_ConvertFToBF16INTEL(float {{.*}})
  // CHECK-NOT: fptoui

  bfloat16 C = A + B;
  // CHECK: [[RTCASTI:%.*]] = addrspacecast float* [[RT:%.*]] to float addrspace(4)*
  // CHECK: [[A_float:%.*]] = call spir_func float @__devicelib_ConvertBF16ToFINTEL(i16 {{.*}})
  // CHECK: [[B_float:%.*]] = call spir_func float @__devicelib_ConvertBF16ToFINTEL(i16 {{.*}})
  // CHECK: [[Add:%.*]] = fadd float [[A_float]], [[B_float]]
  // CHECK: store float [[Add]], float* [[RT]], align 4
  // CHECK: [[C:%.*]] = call spir_func zeroext i16 @__devicelib_ConvertFToBF16INTEL(float {{.*}}) [[RTCASTI]])

  // CHECK-NOT: uitofp
  // CHECK-NOT: fptoui

  long L = bfloat16(3.14f);
  // CHECK: [[L:%.*]] = call spir_func zeroext i16 @__devicelib_ConvertFToBF16INTEL(float {{.*}})
  // CHECK: [[P8:%.*]] = addrspacecast i16* [[VI9:%.*]] to i16 addrspace(4)*
  // CHECK: store i16 [[L]], i16* [[VI9]]
  // CHECK: [[L_float:%.*]] = call spir_func float @__devicelib_ConvertBF16ToFINTEL(i16 {{.*}} [[P8]])
  // CHECK: [[L:%.*]] = fptosi float [[L_float]] to i{{32|64}}

  sycl::half H = bfloat16(2.71f);
  // CHECK: [[H:%.*]] = call spir_func zeroext i16 @__devicelib_ConvertFToBF16INTEL(float {{.*}})
  // CHECK: [[P11:%.*]] = addrspacecast i16* [[VI13:%.*]] to i16 addrspace(4)*
  // CHECK: store i16 [[H]], i16* [[VI13]], align 2
  // CHECK: [[H_float:%.*]] = call spir_func float @__devicelib_ConvertBF16ToFINTEL(i16 {{.*}} [[P11]])
  // CHECK: [[H:%.*]] = fptrunc float [[H_float]] to half
  foo(L, H);

  return A;
  // CHECK: [[RetVal:%.*]] = call spir_func float @__devicelib_ConvertBF16ToFINTEL(i16 {{.*}})
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
