// REQUIRES: amdgpu-registered-target

// Every __spirv_GroupNonUniform*Shuffle* overload that libspirv declares has to
// be defined for amdgcn, as there is no runtime translation of these SPIR-V
// instructions for this target. Linking the device library must therefore leave
// no undefined reference behind.

// RUN: %clang --target=%target -mcpu=%cpu %libclc_lib -cl-std=CL3.0 -O3 -emit-llvm -c -o %t.bc %s
// RUN: llvm-nm -u %t.bc | FileCheck %s --allow-empty --implicit-check-not=__spirv_GroupNonUniformShuffle

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define __TEST_SHUFFLES(TYPE)                                                  \
  __kernel void test_##TYPE(__global TYPE *out, uint arg) {                    \
    TYPE value = out[0];                                                       \
    out[1] = __spirv_GroupNonUniformShuffle(3, value, arg);                    \
    out[2] = __spirv_GroupNonUniformShuffleXor(3, value, arg);                 \
    out[3] = __spirv_GroupNonUniformShuffleUp(3, value, arg);                  \
    out[4] = __spirv_GroupNonUniformShuffleDown(3, value, arg);                \
  }

#define __TEST_TYPE(TYPE)                                                      \
  __TEST_SHUFFLES(TYPE)                                                        \
  __TEST_SHUFFLES(TYPE##2)                                                     \
  __TEST_SHUFFLES(TYPE##3)                                                     \
  __TEST_SHUFFLES(TYPE##4)                                                     \
  __TEST_SHUFFLES(TYPE##8)                                                     \
  __TEST_SHUFFLES(TYPE##16)

__TEST_TYPE(char)
__TEST_TYPE(uchar)
__TEST_TYPE(short)
__TEST_TYPE(ushort)
__TEST_TYPE(int)
__TEST_TYPE(uint)
__TEST_TYPE(long)
__TEST_TYPE(ulong)
__TEST_TYPE(half)
__TEST_TYPE(float)
__TEST_TYPE(double)
