#include <iomanip>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
using namespace access;

// Using-enum-declaration is C++20.
constexpr auto global_space = address_space::global_space;
constexpr auto local_space = address_space::local_space;

// constexpr auto blocked = group_algorithm_data_placement::blocked;
// constexpr auto striped = group_algorithm_data_placement::striped;

template <int Size>
using cl_unsigned = std::conditional_t<
    Size == 1, opencl::cl_uchar,
    std::conditional_t<
        Size == 2, opencl::cl_ushort,
        std::conditional_t<Size == 4, opencl::cl_uint, opencl::cl_ulong>>>;

template <size_t... Inds, class F>
void loop_impl(std::integer_sequence<size_t, Inds...>, F &&f) {
  (f(std::integral_constant<int, Inds>{}), ...);
}

template <size_t count, class F> void loop(F &&f) {
  loop_impl(std::make_index_sequence<count>{}, std::forward<F>(f));
}

void report_results(int *p) {
  int idx = 0;
  int global_size = p[idx++];

  do {
    auto size = p[idx++];
    if (size > 0) {
      auto *str = reinterpret_cast<char *>(p + idx);
      std::cout << str << std::endl;
      idx += size;
      continue;
    }

    if (size == -2 || size == -3) {
      if (size == -3)
        std::cout << std::hex;
      bool all_same = std::all_of(p + idx, p + idx + global_size,
                                 [&](auto v) { return v == p[idx]; });
      if (all_same) {
        std::cout << "All: " << p[idx] << std::endl;
      } else {
        for (int i = 0; i < global_size; ++i) {
          if (i % 8 == 0)
            std::cout << "|";
          if (size == -3)
            std::cout << "0x" << p[idx + i] << " ";
          else
            std::cout << std::setw(3) << p[idx + i] << " ";
        }
        std::cout << "|" << std::endl;
      }
      std::cout << std::dec;
      idx += global_size;
      continue;
    }

    break;

  } while (true);
}

struct Printer {
  Printer(int *p, nd_item<1> ndi) : p(p), ndi(ndi) {
    p[idx++] = ndi.get_global_range(0);
  }
  void print(const char *str) {
    auto len = ((std::strlen(str) + 1) / sizeof(int)) * sizeof(int);
    p[idx++] = len;
    auto *dst = reinterpret_cast<char *>(p + idx);
    do {
      *dst++ = *str++;
    } while (*str); // Expect non-null string.
    idx += len;
  }
  void print(int i) {
    p[idx++] = -2;
    p[idx + ndi.get_global_id(0)] = i;
    idx += ndi.get_global_range(0);
  }
  void print_ptr(void *ptr) {
    p[idx++] = -3;
    p[idx + ndi.get_global_id(0)] = reinterpret_cast<uintptr_t>(ptr);
    idx += ndi.get_global_range(0);
  }
  void finalize() { p[idx++] = -1; }
  int *p;
  int idx = 0;
  nd_item<1> ndi;
};

template <int BlockSize, bool Blocked, int num_blocks, typename GlobalPtrTy>
void load_bytes(sub_group sg, GlobalPtrTy global_ptr_arg, char *priv_ptr,
                Printer &p) {
  // There is no implicit conversion between decorated pointers.
  auto global_ptr = reinterpret_cast<sycl::detail::DecoratedType<
      char, access::address_space::global_space>::type *>(global_ptr_arg);
#ifdef __SYCL_DEVICE_ONLY__
  // Needs to be 4 bytes aligned (16 for writes).

  // Native is strided!
  // Available native BlockSizes: uchar, ushort, uint, ulong (1, 2, 4, 8).
  // Available native num_blocks:
  //   1, 2, 4, 8, or 16 uchars
  //   1, 2, 4, or 8 ulongs
  //   1, 2, 4, or 8 uints of data
  //   1, 2, 4, or 8 ushorts

  if constexpr (Blocked) {
    //    WI 0                         WI 1
    // | BlockSize x num_blocks | BlockSize x num_blocks | ...

    auto sg_lid = sg.get_local_linear_id();

    auto sg_size = sg.get_max_local_range().size(); // Assume "full" SG.

    constexpr auto type_size = [&]() {
      int size = 8;
      while (BlockSize % size != 0)
        size /= 2;
      return size;
    }();
    using BlockT = cl_unsigned<type_size>;
    using PtrT = typename sycl::detail::DecoratedType<
        BlockT, access::address_space::global_space>::type *;

    // Total size: BlockSize * num_blocks * sg_size;
    // We can load type_size * vec_size * sg_size bytes at most.
    // BlockSize % per_load == 0, so by manipulating vec_size (1/2/4/8/(16)) we
    // can always hit the exact range.
    constexpr const auto blocks_per_wi = BlockSize / type_size * num_blocks;
    const auto total_blocks = blocks_per_wi * sg_size;
    int cur_blocks_start_idx = 0;

    //   SG_SIZE=S+1
    //   vec_size=V+1
    //   s -> SG, w-> WI, b->block
    //
    // memory reads:
    //  | s0.w0.b0 | s0.w1.b0 | s0.w2.b0 | ... | sN.wS-1.bV |  sN.wS.bV |
    //
    // A few examples
    //
    // clang-format off
    //  Example 1
    //  Idx\WI |     0    |     1    |
    //     0   | s0.w0.b0 | s0.w1.b1 |
    //     1   | s0.w1.b0 | Different vec_size for the remainder.
    //     2   | s0.w0.b1 |
    //
    //  Example 2
    //  Idx\WI |     0    |     1    |
    //     0   | s0.w0.b0 | s1.w0.b1 |
    //     1   | s0.w1.b0 | s1.w1.b1 |
    //     2   | s0.w0.b1 | s2.w0.b0 |
    //     3   | s0.w1.b1 | s2.w1.b0 |
    //     4   | s1.w0.b0 | s2.w0.b1 |
    //     5   | s1.w1.b0 | s2.w1.b1 |
    //
    //  Example 3
    //  Idx\WI |     0    |     1    |     2    |     3    |
    //    0    | s0.w0.b0 | s1.w1.b1 | s3.w2.b0 | s4.w3.b1 |
    //    1    | s0.w1.b0 | s1.w2.b1 | s3.w3.b0 | s5.w0.b0 |
    //    2    | s0.w2.b0 | s1.w3.b1 | s3.w0.b1 | s5.w1.b0 |
    //    3    | s0.w3.b0 | s2.w0.b0 | s3.w1.b1 | s5.w2.b0 |
    //    4    | s0.w0.b1 | s2.w1.b0 | s3.w2.b1 | s5.w3.b0 |
    //    5    | s0.w1.b1 | s2.w2.b0 | s3.w3.b1 | s5.w0.b1 |
    //    6    | s0.w2.b1 | s2.w3.b0 | s4.w0.b0 | s5.w1.b1 |
    //    7    | s0.w3.b1 | s2.w0.b1 | s4.w1.b0 | s5.w2.b1 |
    //    8    | s1.w0.b0 | s2.w1.b1 | s4.w2.b0 | s5.w3.b1 |
    //    9    | s1.w1.b0 | s2.w2.b1 | s4.w3.b0 | Remainder
    //   10    | s1.w2.b0 | s2.w3.b1 | s4.w0.b1 |
    //   11    | s1.w3.b0 | s3.w0.b0 | s4.w1.b1 |
    //   12    | s1.w0.b1 | s3.w1.b0 | s4.w2.b1 |
    //
    //  Example 4
    //  Idx\WI |     0    |     1    |     2    |     3    |     4    |     5    |     6    |     7    |
    //    0    | s0.w0.b0 | s0.w4.b0 | s0.w0.b1 | s0.w4.b1 | s1.w0.b0 | s1.w4.b0 | s1.w0.b1 | s1.w4.b1 |
    //    1    | s0.w1.b0 | s0.w5.b0 | s0.w1.b1 | s0.w5.b1 | s1.w1.b0 | s1.w5.b0 | s1.w1.b1 | s1.w5.b1 |
    //    2    | s0.w2.b0 | s0.w6.b0 | s0.w2.b1 | s0.w6.b1 | s1.w2.b0 | s1.w6.b0 | s1.w2.b1 | s1.w6.b1 |
    //    3    | s0.w3.b0 | s0.w7.b0 | s0.w3.b1 | s0.w7.b1 | s1.w3.b0 | s1.w7.b0 | s1.w3.b1 | s1.w7.b1 |
    //
    //  Example 5
    //  Idx\WI |     0    |     1    |     2    |     3    |     4    |     5    |     6    |     7    |
    //    0    | s0.w0.b0 | s0.w5.b0 | s0.w2.b1 | s0.w7.b1 | s1.w4.b0 | s1.w1.b1 | s1.w6.b1 |
    //    1    | s0.w1.b0 | s0.w6.b0 | s0.w3.b1 | s1.w0.b0 | s1.w5.b0 | s1.w2.b1 | s1.w7.b1 |
    //    2    | s0.w2.b0 | s0.w7.b0 | s0.w4.b1 | s1.w1.b0 | s1.w6.b0 | s1.w3.b1 | Remainder
    //    3    | s0.w3.b0 | s0.w0.b1 | s0.w5.b1 | s1.w2.b0 | s1.w7.b0 | s1.w4.b1 |
    //    4    | s0.w4.b0 | s0.w1.b1 | s0.w6.b1 | s1.w3.b0 | s1.w0.b1 | s1.w5.b1 |
    // clang-format on
    //
    // Let's focus on example 5, s1:
    //
    //  Here we have
    //    cur_blocks_start_idx == 16
    //    blocks_per_iter      == 16
    //    blocks_per_wi        == 5
    //

    // select next vec_size for the load.
    // 1 = 2^0,  16 = 2^4
    constexpr auto max_vec_pwr_of_two = type_size == 1 ? 4 : 3;
    loop<max_vec_pwr_of_two + 1>([&](auto idx) {
      constexpr auto i = idx.value;
      // Use bigger sizes first.
      constexpr int vec_size = 1 << (max_vec_pwr_of_two - i);
      const auto blocks_per_iter = sg_size * vec_size;

      auto max_blocks_consumed_per_wi =
          // blocks_per_wi < blocks_per_iter ? blocks_per_wi : blocks_per_iter;
          std::min<int>(blocks_per_wi, blocks_per_iter);

      using LoadT = std::conditional_t<
          vec_size == 1, BlockT,
          sycl::detail::ConvertToOpenCLType_t<vec<BlockT, vec_size>>>;

      auto body = [&]() {
        LoadT load = __spirv_SubgroupBlockReadINTEL<LoadT>(
            reinterpret_cast<PtrT>(global_ptr) + cur_blocks_start_idx);

        auto start_wi = cur_blocks_start_idx / blocks_per_wi;
        auto start_wi_vec_idx = cur_blocks_start_idx % blocks_per_wi;

        for (int i = 0; i < max_blocks_consumed_per_wi; ++i) {
          int write_idx = i;
          if (sg_lid == start_wi)
            write_idx += start_wi_vec_idx;

          int needed_idx;
          if (sg_lid > start_wi) {
            needed_idx =
                (sg_lid - start_wi) * blocks_per_wi + i - start_wi_vec_idx;
          } else if (sg_lid == start_wi) {
            needed_idx = i;
          } else {
            needed_idx = 0; // doesn't matter.
          }

          int vec_idx = needed_idx / sg_size;
          int holding_wi_index = needed_idx % sg_size;

          bool write_needed = sg_lid >= start_wi && write_idx < blocks_per_wi &&
                              needed_idx <= blocks_per_iter;
          if (!write_needed)
            holding_wi_index = 0; // Avoid out-of-range shuffle

          // Shuffle has to be in the convergent control flow.
          auto val = select_from_group(sg, load, holding_wi_index);

          if (write_needed) {
            char *ptr = reinterpret_cast<char *>(&val);
            // p.print("Val before memset:");
            // p.print(42);
            // for (int i = 0; i < 17; ++i)
            //   p.print(caller_val[i]);
            // p.print(43);
            std::memcpy(priv_ptr + write_idx * sizeof(BlockT),
                        reinterpret_cast<BlockT *>(&val) + vec_idx,
                        sizeof(BlockT));
            // p.print("Val after memset:");
            // for (int i = 0; i < 17; ++i)
            //   p.print(caller_val[i]);
          }
        }
        cur_blocks_start_idx += blocks_per_iter;
      };

      // if (cur_blocks_start_idx + blocks_per_iter <= total_blocks)
      //   body();
      // if (cur_blocks_start_idx + blocks_per_iter <= total_blocks)
      //   body();
      while (cur_blocks_start_idx + blocks_per_iter <= total_blocks)
        body();
    });
  } else {
    //    WI 0[0]     WI 1[0]          WI 0[1]
    // | BlockSize | BlockSize | ... | BlockSize |  ...
    if constexpr (BlockSize == 1 || BlockSize == 2 || BlockSize == 4 ||
                  BlockSize == 8) {
      auto sg_size = sg.get_max_local_range().size(); // Assume "full" SG.
      using BlockT = cl_unsigned<BlockSize>;
      using PtrT = typename sycl::detail::DecoratedType<
          BlockT, access::address_space::global_space>::type *;

      unsigned max_num_blocks = BlockSize == 1 ? 16 : 8;

      int written = 0;

      auto body = [&] {
        loop<4>([&](auto i) {
          std::ignore = i;
          if (written + max_num_blocks > num_blocks)
            max_num_blocks /= 2;
        });

        // p.print("max_num_blocks:");
        // p.print(max_num_blocks);
        // p.print("global_ptr:");
        // p.print_ptr(global_ptr);

        // "Select" type matching *run-time* parameters by  looping over all the
        // types and proceeding with the one matching run-time value.
        //
        // 1 = 2^0,  16 = 2^4
        loop<BlockSize == 1 ? 5 : 4>([&](auto idx) {
          constexpr auto i = idx.value;
          constexpr int vec_size = 1 << i;
          if (vec_size != max_num_blocks)
            return;
          using LoadT = std::conditional_t<
              vec_size == 1, BlockT,
              sycl::detail::ConvertToOpenCLType_t<vec<BlockT, vec_size>>>;

          LoadT load = __spirv_SubgroupBlockReadINTEL<LoadT>(
              reinterpret_cast<PtrT>(global_ptr) + written * sg_size);
          std::memcpy(priv_ptr + written * BlockSize, &load,
                      vec_size * BlockSize);
          written += vec_size;
        });
      };


      // if (written < num_blocks)
      //   body();
      // if (written < num_blocks)
      //   body();
      // if (written < num_blocks)
      //   body();
      // if (written < num_blocks)
      //   body();
      // if (written < num_blocks)
      //   body();
      // if (written < num_blocks)
      //   body();
      // if (written < num_blocks)
      //   body();
      // if (written < num_blocks)
      //   body();
      while (written < num_blocks)
        body();

      return;
    } else {
      auto sg_size = sg.get_max_local_range().size(); // Assume "full" SG.
      // load_bytes<BlockSize, true, 1>(sg, global_ptr + 0 * BlockSize * sg_size,
      //                                priv_ptr + 0 * BlockSize, p);
      // load_bytes<BlockSize, true, 1>(sg, global_ptr + 1 * BlockSize * sg_size,
      //                                priv_ptr + 1 * BlockSize, p);
      // return;
      for (int i = 0; i < num_blocks; ++i) {
        load_bytes<BlockSize, true, 1>(sg, global_ptr + i * BlockSize * sg_size,
                                       priv_ptr + i * BlockSize, p);
      }
    }
  }
#endif
}

template <bool blocked, typename T, int ELEMS_PER_WI>
void test() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  queue q;
  static constexpr int SG_SIZE = 16;
  constexpr int N_WGS = 1;
  size_t wg_size = SG_SIZE * 1;
  size_t global_size = wg_size * N_WGS;

  buffer<T, 1> b(global_size * ELEMS_PER_WI);
  buffer<bool, 1> result(global_size);
  int *printer_mem = malloc_shared<int>(1024*1024, q);
  q.submit([&](handler &cgh) {
     accessor acc{b, cgh};
     accessor res{result, cgh};
     local_accessor<T> local{wg_size * ELEMS_PER_WI * 2, cgh};
     cgh.parallel_for(
         nd_range{range<1>{global_size}, range<1>{wg_size}},
         [=](nd_item<1> ndi) [[intel::reqd_sub_group_size(SG_SIZE)]] {
           Printer p(printer_mem, ndi);
           auto g = ndi.get_group();
           auto sg = ndi.get_sub_group();

           auto simd = sg.get_max_local_range().size();
           auto sg_size = sg.get_local_range().size();

           auto gid = ndi.get_global_id(0);
           auto lid = ndi.get_local_id(0);
           auto sg_lid = sg.get_local_linear_id();

           auto *global_mem =
               acc.template get_multi_ptr<decorated::yes>().get_decorated();
           auto *wg_mem =
               global_mem + ndi.get_group(0) * wg_size * ELEMS_PER_WI;
           auto *sg_mem = wg_mem + sg.get_group_id() * simd * ELEMS_PER_WI;

           auto *val = local.get_pointer() + lid * ELEMS_PER_WI;
           auto *res_val = val + wg_size * ELEMS_PER_WI;

           auto base = static_cast<T>(gid) + ELEMS_PER_WI;
           for (int i = 0; i < ELEMS_PER_WI; ++i)
             val[i] = base - i;

           // p.print("Val:");
           // for (int i = 0; i < ELEMS_PER_WI; ++i) {
           //   p.print(val[i].i);
           //   p.print(val[i].j);
           //   p.print(val[i].k);
           // }
           // p.print("End val");
           // for (int i = 0; i < ELEMS_PER_WI; ++i)
           //   p.print(val[i]);

           if constexpr (blocked) {
             for (int i = 0; i < ELEMS_PER_WI; ++i)
               sg_mem[sg_lid * ELEMS_PER_WI + i] = val[i];
           } else {
             for (int i = 0; i < ELEMS_PER_WI; ++i)
               sg_mem[sg_lid + sg_size * i] = val[i];
           }

           group_barrier(g);

           constexpr int num_blocks = ELEMS_PER_WI;
           char priv[sizeof(T) * num_blocks];

           load_bytes<sizeof(T), blocked, num_blocks>(sg, sg_mem, priv, p);

           // XDEPS-6185 from using std::equal/std::memcmp.
           char *val_ptr = reinterpret_cast<char *>(val);
           bool success = true;
           for (int i = 0; i < sizeof(T) * num_blocks; ++i)
             success &= (priv[i] == val_ptr[i]);
           res[gid] = success;

           // T res_val[ELEMS_PER_WI];
           std::memcpy(res_val, priv, sizeof(T));

           // p.print("Result:");
           // for (int i = 0; i < ELEMS_PER_WI; ++i) {
           //   p.print(res_val[i].i);
           //   p.print(res_val[i].j);
           //   p.print(res_val[i].k);
           // }
           // for (int i = 0; i < ELEMS_PER_WI; ++i)
           //   p.print(res_val[i]);

           // p.print("Success:");
           // p.print(success);
         });
   }).wait();

  report_results(printer_mem);
  free(printer_mem, q);

  host_accessor res_acc{result};
  assert(std::all_of(res_acc.begin(), res_acc.end(), [](bool r) { return r; }));
}

struct S1 {
  S1() = default;
  S1(int i) : i(i) {}
  operator int() {
    return i;
  }
  int i = 0;
  int j = 2;
};
static_assert(sizeof(S1) == 8);

struct S2 {
  S2() = default;
  S2(int i) : i(i) {}
  operator int() {
    return i;
  }
  int i = 0;
  int j = 2;
  int k = 3;
};
static_assert(sizeof(S2) == 12);

struct __attribute__((packed)) S3 {
  S3() = default;
  S3(int i) : i(i) {}
  operator int() {
    return i;
  }
  int i = 0;
  int j = 2;
  char k = 3;
};
static_assert(sizeof(S3) == 9);

int main() {
  constexpr int sizes[] = {1, 2, 3, 7, 8, 17, 16, 32, 64, 67};
#ifndef SINGLE
  loop<std::size(sizes)>([&](auto i) {
    constexpr int size = sizes[i.value];
    test<true, char, size>();
    test<true, short, size>();
    test<true, int, size>();
    test<true, long long, size>();
    test<true, float, size>();
    test<true, S1, size>();
    test<true, S2, size>();
    test<true, S3, size>();

    test<false, char, size>();
    test<false, short, size>();
    test<false, int, size>();
    test<false, long long, size>();
    test<false, float, size>();
    test<false, S1, size>();
    test<false, S2, size>();
    test<false, S3, size>();
  });
#else
  test<true, S3, 67>();
#endif

}
