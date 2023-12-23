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
3)  Calling ESIMD from SYCL using invoke_simd and Shared Local Memory (SLM) - ["invoke_simd_slm"](./invoke_simd_slm.md).
  
    Please see the full source code here: ["invoke_simd_slm"](./invoke_simd_slm.md)
    ```c++
     [[intel::device_indirectly_callable]] SYCL_EXTERNAL void __regcall invoke_slm_load_store(
     local_accessor<int, 1> *local_acc, uint32_t slm_byte_offset, int *in, int *out,
     simd<uint32_t, VL> global_byte_offsets) SYCL_ESIMD_FUNCTION {
      esimd::simd<uint32_t, VL> esimd_global_byte_offsets = global_byte_offsets;
      // Read SLM in ESIMD context.
      auto local1 = esimd::block_load<int, VL>(*local_acc, slm_byte_offset);
      auto local2 = esimd::block_load<int, VL>(*local_acc, slm_byte_offset + 
                                               LOCAL_RANGE * sizeof(int));
      auto global = esimd::gather(in, esimd_global_byte_offsets);
      auto res = global + local1 + local2;
      esimd::scatter(out, esimd_global_byte_offsets, res);
     }

     int main(void) {
       auto *in = malloc_shared<int>(GLOBAL_RANGE, q);
       auto *out = malloc_shared<int>(GLOBAL_RANGE, q);
       ...
       q.submit([&](handler &cgh) {
         auto local_acc = local_accessor<int, 1>(LOCAL_RANGE * 2, cgh);
         cgh.parallel_for(nd_range, [=](nd_item<1> item) {
           uint32_t global_id = item.get_global_id(0);
           uint32_t local_id = item.get_local_id(0);
           // Write/initialize SLM in SYCL context.
           auto local_acc_copy = local_acc;
           local_acc_copy[local_id] = global_id * 2;
           local_acc_copy[local_id + LOCAL_RANGE] = global_id * 3;
           item.barrier();

           uint32_t la_byte_offset = (local_id / VL) * VL * sizeof(int);
           uint32_t global_byte_offset = global_id * sizeof(int);
           sycl::sub_group sg = item.get_sub_group();
           // Pass the local-accessor to initialized SLM memory to ESIMD context.
           // Pointer to a local copy of the local accessor is passed instead of a local-accessor value now
           // to work-around a known/temporary issue in GPU driver.
           auto local_acc_arg = uniform{&local_acc_copy};
           invoke_simd(sg, invoke_slm_load_store, local_acc_arg,
                       uniform{la_byte_offset}, uniform{in}, uniform{out},
                       global_byte_offset);
         });
       })
    ```
4) Dot Product Accumulate Systolic (DPAS) API - ["dpas"](./dpas.md)

   Please see the full source code here: ["dpas"](./dpas.md)

   ```c++
    // Res = A * B.
    // Assume the HW is PVC.

    constexpr int SystolicDepth = 8;
    constexpr int RepeatCount = 4;
    constexpr int ExecSize = 16; // 16 for PVC, 8 for DG2.

    // Let A and B be matrices of unsigned 4-bit integers.
    constexpr xmx::dpas_argument_type BPrec = xmx::dpas_argument_type::u4;
    constexpr xmx::dpas_argument_type APrec = xmx::dpas_argument_type::u4;

    constexpr int AElemBitSize = 4; // 4-bit integers.
    constexpr int BElemBitSize = 4; // 4-bit integers.

    // Elements of A and B will are packed into uint8_t,
    // meaning that one uint8_t holds two 4-bit unsigned integers.
    // Packaging for A and res is horizontal, for B is vertical.
    using PackedType = unsigned char;
    using APackedType = PackedType;
    using BPackedType = PackedType;

    // Result type, according to documentation is either int or uint.
    using ResType = unsigned int; // as both A and B are unsigned.

    constexpr int OpsPerChannel =
        std::min(32 / std::max(AElemBitSize, BElemBitSize), 8);

    // A(M x K) * B(K x N) + C(M x N).
    // where:
    constexpr int M = RepeatCount;
    constexpr int K = SystolicDepth * OpsPerChannel;
    constexpr int N = ExecSize;

    int main() {
      unsigned n_errs = 0;
      try {
        queue q(gpu_selector_v, create_exception_handler());
        auto dev = q.get_device();
        std::cout << "Running on " << dev.get_info<info::device::name>()
                  << std::endl;

        constexpr unsigned Size = 128;
        constexpr unsigned VL = 16;

        constexpr int APackedSize =
            M * K * AElemBitSize / (sizeof(APackedType) * 8);
        constexpr int BPackedSize =
            K * N * BElemBitSize / (sizeof(BPackedType) * 8);

        auto a_packed = aligned_alloc_shared<APackedType>(128, APackedSize, q);
        auto b_packed = aligned_alloc_shared<BPackedType>(128, BPackedSize, q);
        auto res = aligned_alloc_shared<ResType>(128, M * N, q);

        std::unique_ptr<APackedType, usm_deleter> guard_a(a_packed, usm_deleter{q});
        std::unique_ptr<BPackedType, usm_deleter> guard_b(b_packed, usm_deleter{q});
        std::unique_ptr<ResType, usm_deleter> guard_res(res, usm_deleter{q});

        // Initialize a_packed;
        unsigned value = 0;
        for (int i = 0; i < M; i++) {
          for (int j = 0; j < K; j++) {
            value += 1;
            write_to_horizontally_packed_matrix_a(a_packed, i, j,
                                                  static_cast<APackedType>(value));
          }
        }

        // Initialize b_packed;
        for (int i = 0; i < K; i++) {
          for (int j = 0; j < N; j++) {
            int value = (i + j % 4) == 0 ? 1 : (2 + i + j) % 3;
            write_to_vertically_packed_matrix_b(b_packed, i, j,
                                                static_cast<BPackedType>(value));
            assert(value == (int)(static_cast<BPackedType>(value)) && "ERROR");
          }
        }

        q.single_task([=]() [[intel::sycl_explicit_simd] {
           esimd::simd<APackedType, APackedSize> a(a_packed,
                                                   esimd::overaligned_tag<16>{});
           esimd::simd<BPackedType, BPackedSize> b(b_packed,
                                                   esimd::overaligned_tag<16>{});
           esimd::simd<ResType, M * N> c;

           // Compute C = AxB;
           c = xmx::dpas<8, RepeatCount, ResType, BPackedType, APackedType, BPrec,
                         APrec>(b, a);
           c.copy_to(res);
         }).wait();
   ...
   }
   ```

4) Using simd_view to construct views of simd objects - ["simd_view"](./simd_view.md).
   Please see the full source code here: ["simd_view"](./simd_view.md).

   ```c++
   float *a = malloc_shared<float>(Size, q); // USM memory for A.

   // Initialize a.

   // For elements of 'a' with indices, which are:
   //   * multiple of 4: multiply by 6;
   //   * multiple of 2: multiply by 3;
   q.parallel_for(Size / VL, [=](id<1> i) [[intel::sycl_explicit_simd]] {
      auto element_offset = i * VL;
      simd<float, VL> vec_a(a + element_offset);

      // simd_view of simd<float, VL> using the even-index elements.
      auto vec_a_even_elems_view = vec_a.select<VL / 2, 2>(0);
      vec_a_even_elems_view *= 3;

      // simd_view with even indices constructed from previous
      // simd_view of simd<float, VL> using the even-index elements.
      // This results in a simd_view containing every fourth element
      // of vec_a.
      auto vec_a_mult_four_view = vec_a_even_elems_view.select<VL / 4, 2>(0);
      vec_a_mult_four_view *= 2;

      // Copy back to the memory.
      vec_a.copy_to(a + element_offset);
   }).wait_and_throw();
   ```

5) Using BFN API to efficiently compute a binary function of three parameters.
   Please see the full source code here: ["bfn"]("bfn.md").

   Illustrates using a special API to compute a binary function of three operands.
   The function is computed for every bit in the integral input.

   ```c++
   // Let's compute f(x, y, z) = ~x | y ^ z.
   constexpr esimd::bfn_t F = ~bfn_t::x | bfn_t::y ^ bfn_t::z;

   q.parallel_for(Size / VL, [=](id<1> i) [[intel::sycl_explicit_simd]] {
     using namespace esimd;
     auto offset = i * VL;
     simd<unsigned, VL> X(in1 + offset);
     simd<unsigned, VL> Y(in2 + offset);
     simd<unsigned, VL> Z(in3 + offset);

     simd<unsigned, VL> val = bfn<F>(X, Y, Z);
     val.copy_to(res + offset);
   }).wait();
   ```

6) TODO: Add more examples here.
