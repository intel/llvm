__kernel void my_kernel(__global int *in, __global int *out) {
  size_t i = get_global_id(0);
  out[i] = in[i] * 2 + 100;
}

#define TEST_PARAM_OpType(NAME, N, TYPE)                                       \
  __kernel void OpType##NAME##N(TYPE a, __global TYPE *b, __global TYPE *out,  \
                                __local TYPE *loc) {                           \
    *loc = a * a;                                                              \
    *out = *loc + ((*b) * (*b));                                               \
  }

// clang-format off

// fp16 and fp64 kernels are compiled into separate SPIR-V files since certain
// devices don't support them.
#if defined(ENABLE_FP16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
TEST_PARAM_OpType(Float, 16, half)
#elif defined (ENABLE_FP64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
TEST_PARAM_OpType(Float, 64, double)
#else
TEST_PARAM_OpType(Int, 8, char)
TEST_PARAM_OpType(Int, 16, short)
TEST_PARAM_OpType(Int, 32, int)
TEST_PARAM_OpType(Int, 64, long)
TEST_PARAM_OpType(Float, 32, float)
#endif
