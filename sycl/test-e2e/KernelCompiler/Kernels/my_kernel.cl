__kernel void my_kernel(__global int *in, __global int *out) {
  size_t i = get_global_id(0);
  out[i] = in[i]*2 + 100;
}
