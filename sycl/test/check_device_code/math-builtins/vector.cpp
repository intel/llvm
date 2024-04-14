// RUN: %clangxx -fsycl-device-only -S -O0 -Xclang -emit-llvm -o - %s | FileCheck %s --dump-input=always

// This test checks that sycl::vec uses std::array as storage type on device for all
// data types.
#include <sycl/sycl.hpp>

using namespace sycl;

// Create sycl::vec for each type and verify the storage
// type on device side.
#define CHECK(Q, T, N)                                                     \
  {                                                                        \
    Q.submit([&](handler &CGH) {                                           \
      CGH.single_task([=]() {                                              \
        vec<T, 2> InVec##N##2{static_cast<T>(5)};                          \
        vec<T, 3> InVec##N##3{static_cast<T>(5)};                          \
        vec<T, 4> InVec##N##4{static_cast<T>(5)};                          \
        vec<T, 8> InVec##N##8{static_cast<T>(5)};                          \
        vec<T, 16> InVec##N##16{static_cast<T>(5)};                        \
      });                                                                  \
    });                                                                    \   
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

  // CHECK: {{.*}}InVecINT2 = {{.*}}std::array{{.*}}" { [2 x i32] {{.*}}
  // CHECK: {{.*}}InVecINT3 = {{.*}}std::array.{{.*}}" { [4 x i32] {{.*}}
  // CHECK: {{.*}}InVecINT4 = {{.*}}std::array.{{.*}}" { [4 x i32] {{.*}}
  // CHECK: {{.*}}InVecINT8 = {{.*}}std::array.{{.*}}" { [8 x i32] {{.*}}
  // CHECK: {{.*}}InVecINT16 = {{.*}}std::array.{{.*}}" { [16 x i32] {{.*}}
  CHECK(q, int, INT)

  // CHECK: {{.*}}InVecFLOAT2 = {{.*}}std::array.{{.*}}" { [2 x float] {{.*}}
  // CHECK: {{.*}}InVecFLOAT3 = {{.*}}std::array.{{.*}}" { [4 x float] {{.*}}
  // CHECK: {{.*}}InVecFLOAT4 = {{.*}}std::array.{{.*}}" { [4 x float] {{.*}}
  // CHECK: {{.*}}InVecFLOAT8 = {{.*}}std::array.{{.*}}" { [8 x float] {{.*}}
  // CHECK: {{.*}}InVecFLOAT16 = {{.*}}std::array.{{.*}}" { [16 x float] {{.*}}
  CHECK(q, float, FLOAT)

  // CHECK: {{.*}}InVecCHAR2 = {{.*}}std::array.{{.*}}" { [2 x i8] {{.*}}
  // CHECK: {{.*}}InVecCHAR3 = {{.*}}std::array.{{.*}}" { [4 x i8] {{.*}}
  // CHECK: {{.*}}InVecCHAR4 = {{.*}}std::array.{{.*}}" { [4 x i8] {{.*}}
  // CHECK: {{.*}}InVecCHAR8 = {{.*}}std::array.{{.*}}" { [8 x i8] {{.*}}
  // CHECK: {{.*}}InVecCHAR16 = {{.*}}std::array.{{.*}}" { [16 x i8] {{.*}}
  CHECK(q, char, CHAR)

  // CHECK: {{.*}}InVecBOOL2 = {{.*}}std::array.{{.*}}" { [2 x i8] {{.*}}
  // CHECK: {{.*}}InVecBOOL3 = {{.*}}std::array.{{.*}}" { [4 x i8] {{.*}}
  // CHECK: {{.*}}InVecBOOL4 = {{.*}}std::array.{{.*}}" { [4 x i8] {{.*}}
  // CHECK: {{.*}}InVecBOOL8 = {{.*}}std::array.{{.*}}" { [8 x i8] {{.*}}
  // CHECK: {{.*}}InVecBOOL16 = {{.*}}std::array.{{.*}}" { [16 x i8] {{.*}}
  CHECK(q, bool, BOOL)

  // CHECK: {{.*}}InVecHALF2 = {{.*}}std::array.{{.*}}" { [2 x half] {{.*}}
  // CHECK: {{.*}}InVecHALF3 = {{.*}}std::array.{{.*}}" { [4 x half] {{.*}}
  // CHECK: {{.*}}InVecHALF4 = {{.*}}std::array.{{.*}}" { [4 x half] {{.*}}
  // CHECK: {{.*}}InVecHALF8 = {{.*}}std::array.{{.*}}" { [8 x half] {{.*}}
  // CHECK: {{.*}}InVecHALF16 = {{.*}}std::array.{{.*}}" { [16 x half] {{.*}}
  CHECK(q, half, HALF)

  // CHECK: {{.*}}InVecBYTE2 = {{.*}}std::array.{{.*}}" { [2 x i8] {{.*}}
  // CHECK: {{.*}}InVecBYTE3 = {{.*}}std::array.{{.*}}" { [4 x i8] {{.*}}
  // CHECK: {{.*}}InVecBYTE4 = {{.*}}std::array.{{.*}}" { [4 x i8] {{.*}}
  // CHECK: {{.*}}InVecBYTE8 = {{.*}}std::array.{{.*}}" { [8 x i8] {{.*}}
  // CHECK: {{.*}}InVecBYTE16 = {{.*}}std::array.{{.*}}" { [16 x i8] {{.*}}
  CHECK(q, std::byte, BYTE)
   
  return 0;
};
