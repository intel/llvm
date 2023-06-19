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

2) TODO: Add more examples here.
