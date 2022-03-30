// RUN: %clangxx -fsycl-device-only -fsycl-targets=%sycl_triple -S -Xclang -no-enable-noundef-analysis %s -o - | FileCheck %s

// UNSUPPORTED: cuda || hip_amd

#include <sycl/ext/intel/experimental/bfloat16.hpp>
#include <sycl/sycl.hpp>

using sycl::ext::intel::experimental::bfloat16;

SYCL_EXTERNAL uint16_t some_bf16_intrinsic(uint16_t x, uint16_t y);
SYCL_EXTERNAL void foo(long x, sycl::half y);

__attribute__((noinline)) float op(float a, float b) {
  // CHECK: define {{.*}} spir_func float @_Z2opff(float [[a:%.*]], float [[b:%.*]])
  bfloat16 A{a};
  // CHECK: [[A:%.*]] = tail call spir_func zeroext i16 @_Z27__spirv_ConvertFToBF16INTELf(float [[a]])
  // CHECK-NOT: fptoui

  bfloat16 B{b};
  // CHECK: [[B:%.*]] = tail call spir_func zeroext i16 @_Z27__spirv_ConvertFToBF16INTELf(float [[b]])
  // CHECK-NOT: fptoui

  bfloat16 C = A + B;
  // CHECK: [[A_float:%.*]] = tail call spir_func float @_Z27__spirv_ConvertBF16ToFINTELt(i16 zeroext [[A]])
  // CHECK: [[B_float:%.*]] = tail call spir_func float @_Z27__spirv_ConvertBF16ToFINTELt(i16 zeroext [[B]])
  // CHECK: [[Add:%.*]] = fadd float [[A_float]], [[B_float]]
  // CHECK: [[C:%.*]] = tail call spir_func zeroext i16 @_Z27__spirv_ConvertFToBF16INTELf(float [[Add]])
  // CHECK-NOT: uitofp
  // CHECK-NOT: fptoui

  bfloat16 D = bfloat16::from_bits(some_bf16_intrinsic(A.raw(), C.raw()));
  // CHECK: [[D:%.*]] = tail call spir_func zeroext i16 @_Z19some_bf16_intrinsictt(i16 zeroext [[A]], i16 zeroext [[C]])
  // CHECK-NOT: uitofp
  // CHECK-NOT: fptoui

  long L = bfloat16(3.14f);
  // CHECK: [[L_bfloat16:%.*]] = tail call spir_func zeroext i16 @_Z27__spirv_ConvertFToBF16INTELf(float 0x40091EB860000000)
  // CHECK: [[L_float:%.*]] = tail call spir_func float @_Z27__spirv_ConvertBF16ToFINTELt(i16 zeroext [[L_bfloat16]])
  // CHECK: [[L:%.*]] = fptosi float [[L_float]] to i{{32|64}}

  sycl::half H = bfloat16(2.71f);
  // CHECK: [[H_bfloat16:%.*]] = tail call spir_func zeroext i16 @_Z27__spirv_ConvertFToBF16INTELf(float 0x4005AE1480000000)
  // CHECK: [[H_float:%.*]] = tail call spir_func float @_Z27__spirv_ConvertBF16ToFINTELt(i16 zeroext [[H_bfloat16]])
  // CHECK: [[H:%.*]] = fptrunc float [[H_float]] to half
  foo(L, H);

  return D;
  // CHECK: [[RetVal:%.*]] = tail call spir_func float @_Z27__spirv_ConvertBF16ToFINTELt(i16 zeroext [[D]])
  // CHECK: ret float [[RetVal]]
  // CHECK-NOT: uitofp
  // CHECK-NOT: fptoui
}

int main(int argc, char *argv[]) {
  float data[3] = {7.0, 8.1, 0.0};
  cl::sycl::queue deviceQueue;
  cl::sycl::buffer<float, 1> buf{data, cl::sycl::range<1>{3}};

  deviceQueue.submit([&](cl::sycl::handler &cgh) {
    auto numbers = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.single_task<class simple_kernel>(
        [=]() { numbers[2] = op(numbers[0], numbers[1]); });
  });
  return 0;
}
