// RUN: %clangxx -fsycl-device-only -S -Xclang -emit-llvm %s -o - | FileCheck %s

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/bfloat16.hpp>

using sycl::ext::intel::experimental::bfloat16;

SYCL_EXTERNAL uint16_t some_bf16_intrinsic(uint16_t x, uint16_t y);

__attribute__((noinline))
float op(float a, float b) {
  bfloat16 A {a};
// CHECK: [[A:%.*]] = tail call spir_func zeroext i16 @_Z27__spirv_ConvertFToBF16INTELf(float %a)
// CHECK-NOT: fptoui

  bfloat16 B {b};
// CHECK: [[B:%.*]] = tail call spir_func zeroext i16 @_Z27__spirv_ConvertFToBF16INTELf(float %b)
// CHECK-NOT: fptoui

  bfloat16 C = A + B;
// CHECK: [[A_float:%.*]] = tail call spir_func float @_Z27__spirv_ConvertBF16ToFINTELt(i16 zeroext [[A]])
// CHECK: [[B_float:%.*]] = tail call spir_func float @_Z27__spirv_ConvertBF16ToFINTELt(i16 zeroext [[B]])
// CHECK: [[Add:%.*]] = fadd float [[A_float]], [[B_float]]
// CHECK: [[C:%.*]] = tail call spir_func zeroext i16 @_Z27__spirv_ConvertFToBF16INTELf(float [[Add]])
// CHECK-NOT: uitofp
// CHECK-NOT: fptoui

  bfloat16 D = some_bf16_intrinsic(A, C);
// CHECK: [[D:%.*]] = tail call spir_func zeroext i16 @_Z19some_bf16_intrinsictt(i16 zeroext [[A]], i16 zeroext [[C]])
// CHECK-NOT: uitofp
// CHECK-NOT: fptoui

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
