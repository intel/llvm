# ESIMD Examples

This folder contains simple ESIMD examples. The main purpose of having them
is to show the basic ESIMD APIs in well known examples.

1) The most basic example - ["sum_two_arrays"](./sum_two_arrays.md).
   
   Please see the full source here: ["sum_two_arrays"](./sum_two_arrays.md).
   ```c++
   float *a = malloc_shared<float>(Size, q); // USM memory for A
   float *b = new float[Size];               // B uses HOST memory
   buffer<float, 1> buf_b(b, Size);

   // Initialize 'a' and 'b' here.
    
   // Compute: a[i] += b[i];
   q.submit([&](handler &cgh) {
     auto acc_b = buf_b.get_access<access::mode::read>(cgh);
     cgh.parallel_for(Size / VL, [=](id<1> i) [[intel::sycl_explicit_simd]] {
       auto element_offset = i * VL;
       simd<float, VL> vec_a(a + element_offset); // Pointer arithmetic uses element offset
       simd<float, VL> vec_b(acc_b, element_offset * sizeof(float)); // accessor API uses byte-offset

       vec_a += vec_b;
       vec_a.copy_to(a + element_offset);
     });
   }).wait_and_throw();
   ```
2) Calling ESIMD from SYCL using invoke_simd - ["invoke_simd"](./invoke_simd.md).
   Please see the full source code here: ["invoke_simd"](./invoke_simd.md)
   ```c++
   [[intel::device_indirectly_callable]] simd<int, VL> __regcall scale(
     simd<int, VL> x, int n) SYCL_ESIMD_FUNCTION {
     esimd::simd<int, VL> vec = x;
     esimd::simd<int, VL> result = vec * n;
    return result;
   }

   int main(void) { 
     int *in = new int[SIZE];
     int *out = new int[SIZE];
     buffer<int, 1> bufin(in, range<1>(SIZE));
     buffer<int, 1> bufout(out, range<1>(SIZE));

     // scale factor
     int n = 2;

     sycl::range<1> GlobalRange{SIZE};
     sycl::range<1> LocalRange{VL};
    
     q.submit([&](handler &cgh) {
      auto accin = bufin.get_access<access::mode::read>(cgh);
      auto accout = bufout.get_access<access::mode::write>(cgh);

      cgh.parallel_for<class Scale>(
          nd_range<1>(GlobalRange, LocalRange), [=](nd_item<1> item) {
            sycl::sub_group sg = item.get_sub_group();
            unsigned int offset = item.get_global_linear_id();

            int in_val = sg.load(accin.get_pointer() + offset);

            int out_val = invoke_simd(sg, scale, in_val, uniform{n});

            sg.store(accout.get_pointer() + offset, out_val);
          });
    });
    ```
3) TODO: Add more examples here.
