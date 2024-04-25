// RUN: %clangxx -fsycl-device-only -S -O0 -Xclang -emit-llvm -o - %s | FileCheck %s

// This test checks the storage type of sycl::vec on device for
// all data types.
#include <sycl/sycl.hpp>

using namespace sycl;

// Create sycl::vec for each type and verify the storage
// type on device side.
#define CHECK(Q, T, N)                                                         \
  {                                                                            \
    Q.submit([&](handler &CGH) {                                               \
      CGH.single_task([=]() {                                                  \
        vec<T, 2> InVec##N##2 {static_cast<T>(5)};                             \
        vec<T, 3> InVec##N##3 {static_cast<T>(5)};                             \
        vec<T, 4> InVec##N##4 {static_cast<T>(5)};                             \
        vec<T, 8> InVec##N##8 {static_cast<T>(5)};                             \
        vec<T, 16> InVec##N##16 {static_cast<T>(5)};                           \
      });                                                                      \
    });                                                                        \
  }

int main() {

  queue q;

  // CHECK: %"class.sycl::_V1::ext::oneapi::bfloat16" = type { i16 }
  // CHECK: {{.*}}std::array{{.*}}" = type { [2 x %"class.sycl::_V1::ext::oneapi::bfloat16"] }
  // CHECK: {{.*}}std::array{{.*}}" = type { [3 x %"class.sycl::_V1::ext::oneapi::bfloat16"] }
  // CHECK: {{.*}}std::array{{.*}}" = type { [4 x %"class.sycl::_V1::ext::oneapi::bfloat16"] }
  // CHECK: {{.*}}std::array{{.*}}" = type { [8 x %"class.sycl::_V1::ext::oneapi::bfloat16"] }
  // CHECK: {{.*}}std::array{{.*}}" = type { [16 x %"class.sycl::_V1::ext::oneapi::bfloat16"] }
  CHECK(q, ext::oneapi::bfloat16, BF)

  // CHECK: {{.*}}InVecINT2 = {{.*}}sycl::_V1::vec{{.*}}" { <2 x i32> {{.*}}
  // CHECK: {{.*}}InVecINT3 = {{.*}}sycl::_V1::vec{{.*}}" { <3 x i32> {{.*}}
  // CHECK: {{.*}}InVecINT4 = {{.*}}sycl::_V1::vec{{.*}}" { <4 x i32> {{.*}}
  // CHECK: {{.*}}InVecINT8 = {{.*}}sycl::_V1::vec{{.*}}" { <8 x i32> {{.*}}
  // CHECK: {{.*}}InVecINT16 = {{.*}}sycl::_V1::vec{{.*}}" { <16 x i32> {{.*}}
  CHECK(q, int, INT)

  // CHECK: {{.*}}InVecFLOAT2 = {{.*}}sycl::_V1::vec{{.*}}" { <2 x float> {{.*}}
  // CHECK: {{.*}}InVecFLOAT3 = {{.*}}sycl::_V1::vec{{.*}}" { <3 x float> {{.*}}
  // CHECK: {{.*}}InVecFLOAT4 = {{.*}}sycl::_V1::vec{{.*}}" { <4 x float> {{.*}}
  // CHECK: {{.*}}InVecFLOAT8 = {{.*}}sycl::_V1::vec{{.*}}" { <8 x float> {{.*}}
  // CHECK: {{.*}}InVecFLOAT16 = {{.*}}sycl::_V1::vec{{.*}}" { <16 x float> {{.*}}
  CHECK(q, float, FLOAT)

  // CHECK: {{.*}}InVecCHAR2 = {{.*}}sycl::_V1::vec{{.*}}" { <2 x i8> {{.*}}
  // CHECK: {{.*}}InVecCHAR3 = {{.*}}sycl::_V1::vec{{.*}}" { <3 x i8> {{.*}}
  // CHECK: {{.*}}InVecCHAR4 = {{.*}}sycl::_V1::vec{{.*}}" { <4 x i8> {{.*}}
  // CHECK: {{.*}}InVecCHAR8 = {{.*}}sycl::_V1::vec{{.*}}" { <8 x i8> {{.*}}
  // CHECK: {{.*}}InVecCHAR16 = {{.*}}sycl::_V1::vec{{.*}}" { <16 x i8> {{.*}}
  CHECK(q, char, CHAR)

  // CHECK: {{.*}}InVecBOOL2 = {{.*}}sycl::_V1::vec{{.*}}" { <2 x i8> {{.*}}
  // CHECK: {{.*}}InVecBOOL3 = {{.*}}sycl::_V1::vec{{.*}}" { <3 x i8> {{.*}}
  // CHECK: {{.*}}InVecBOOL4 = {{.*}}sycl::_V1::vec{{.*}}" { <4 x i8> {{.*}}
  // CHECK: {{.*}}InVecBOOL8 = {{.*}}sycl::_V1::vec{{.*}}" { <8 x i8> {{.*}}
  // CHECK: {{.*}}InVecBOOL16 = {{.*}}sycl::_V1::vec{{.*}}" { <16 x i8> {{.*}}
  CHECK(q, bool, BOOL)

  // CHECK: {{.*}}InVecHALF2 = {{.*}}sycl::_V1::vec{{.*}}" { <2 x half> {{.*}}
  // CHECK: {{.*}}InVecHALF3 = {{.*}}sycl::_V1::vec{{.*}}" { <3 x half> {{.*}}
  // CHECK: {{.*}}InVecHALF4 = {{.*}}sycl::_V1::vec{{.*}}" { <4 x half> {{.*}}
  // CHECK: {{.*}}InVecHALF8 = {{.*}}sycl::_V1::vec{{.*}}" { <8 x half> {{.*}}
  // CHECK: {{.*}}InVecHALF16 = {{.*}}sycl::_V1::vec{{.*}}" { <16 x half> {{.*}}
  CHECK(q, half, HALF)

  // CHECK: {{.*}}InVecBYTE2 = {{.*}}sycl::_V1::vec{{.*}}" { <2 x i8> {{.*}}
  // CHECK: {{.*}}InVecBYTE3 = {{.*}}sycl::_V1::vec{{.*}}" { <3 x i8> {{.*}}
  // CHECK: {{.*}}InVecBYTE4 = {{.*}}sycl::_V1::vec{{.*}}" { <4 x i8> {{.*}}
  // CHECK: {{.*}}InVecBYTE8 = {{.*}}sycl::_V1::vec{{.*}}" { <8 x i8> {{.*}}
  // CHECK: {{.*}}InVecBYTE16 = {{.*}}sycl::_V1::vec{{.*}}" { <16 x i8> {{.*}}
  CHECK(q, std::byte, BYTE)

  // Check alignment of the allocated sycl::vec

  // CHECK: %InVecBF2 = alloca %"class.sycl::_V1::vec{{.*}}", align 4
  // CHECK: %InVecBF3 = alloca %"class.sycl::_V1::vec{{.*}}", align 8
  // CHECK: %InVecBF4 = alloca %"class.sycl::_V1::vec{{.*}}", align 8
  // CHECK: %InVecBF8 = alloca %"class.sycl::_V1::vec{{.*}}", align 16
  // CHECK: %InVecBF16 = alloca %"class.sycl::_V1::vec{{.*}}", align 32

  // CHECK: %InVecINT2 = alloca %"class.sycl::_V1::vec{{.*}}", align 8
  // CHECK: %InVecINT3 = alloca %"class.sycl::_V1::vec{{.*}}", align 16
  // CHECK: %InVecINT4 = alloca %"class.sycl::_V1::vec{{.*}}", align 16
  // CHECK: %InVecINT8 = alloca %"class.sycl::_V1::vec{{.*}}", align 32
  // CHECK: %InVecINT16 = alloca %"class.sycl::_V1::vec{{.*}}", align 64

  // CHECK: %InVecFLOAT2 = alloca %"class.sycl::_V1::vec{{.*}}", align 8
  // CHECK: %InVecFLOAT3 = alloca %"class.sycl::_V1::vec{{.*}}", align 16
  // CHECK: %InVecFLOAT4 = alloca %"class.sycl::_V1::vec{{.*}}", align 16
  // CHECK: %InVecFLOAT8 = alloca %"class.sycl::_V1::vec{{.*}}", align 32
  // CHECK: %InVecFLOAT16 = alloca %"class.sycl::_V1::vec{{.*}}", align 64

  // CHECK: %InVecCHAR2 = alloca %"class.sycl::_V1::vec{{.*}}", align 2
  // CHECK: %InVecCHAR3 = alloca %"class.sycl::_V1::vec{{.*}}", align 4
  // CHECK: %InVecCHAR4 = alloca %"class.sycl::_V1::vec{{.*}}", align 4
  // CHECK: %InVecCHAR8 = alloca %"class.sycl::_V1::vec{{.*}}", align 8
  // CHECK: %InVecCHAR16 = alloca %"class.sycl::_V1::vec{{.*}}", align 16

  // CHECK: %InVecBOOL2 = alloca %"class.sycl::_V1::vec{{.*}}", align 2
  // CHECK: %InVecBOOL3 = alloca %"class.sycl::_V1::vec{{.*}}", align 4
  // CHECK: %InVecBOOL4 = alloca %"class.sycl::_V1::vec{{.*}}", align 4
  // CHECK: %InVecBOOL8 = alloca %"class.sycl::_V1::vec{{.*}}", align 8
  // CHECK: %InVecBOOL16 = alloca %"class.sycl::_V1::vec{{.*}}", align 16

  // CHECK: %InVecHALF2 = alloca %"class.sycl::_V1::vec{{.*}}", align 4
  // CHECK: %InVecHALF3 = alloca %"class.sycl::_V1::vec{{.*}}", align 8
  // CHECK: %InVecHALF4 = alloca %"class.sycl::_V1::vec{{.*}}", align 8
  // CHECK: %InVecHALF8 = alloca %"class.sycl::_V1::vec{{.*}}", align 16
  // CHECK: %InVecHALF16 = alloca %"class.sycl::_V1::vec{{.*}}", align 32

  // CHECK: %InVecBYTE2 = alloca %"class.sycl::_V1::vec{{.*}}", align 2
  // CHECK: %InVecBYTE3 = alloca %"class.sycl::_V1::vec{{.*}}", align 4
  // CHECK: %InVecBYTE4 = alloca %"class.sycl::_V1::vec{{.*}}", align 4
  // CHECK: %InVecBYTE8 = alloca %"class.sycl::_V1::vec{{.*}}", align 8
  // CHECK: %InVecBYTE16 = alloca %"class.sycl::_V1::vec{{.*}}", align 16

  return 0;
};
