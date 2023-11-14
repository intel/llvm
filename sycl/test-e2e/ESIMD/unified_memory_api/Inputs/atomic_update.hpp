//==---------------- atomic_update.hpp - DPC++ ESIMD on-device test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <iostream>
#include <sycl/ext/intel/esimd.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

#ifdef USE_64_BIT_OFFSET
typedef uint64_t Toffset;
#else
typedef uint32_t Toffset;
#endif

constexpr int Signed = 1;
constexpr int Unsigned = 2;

struct Config {
  int64_t threads_per_group;
  int64_t n_groups;
  int64_t start_ind;
  int64_t use_mask;
  int64_t masked_lane;
  int64_t repeat;
  int64_t stride;
};

// Helper functions

std::ostream &operator<<(std::ostream &out, const Config &cfg) {
  out << "{ thr_per_group=" << cfg.threads_per_group
      << " n_groups=" << cfg.n_groups << " start_ind=" << cfg.start_ind
      << " use_mask= " << cfg.use_mask << " masked_lane=" << cfg.masked_lane
      << " repeat=" << cfg.repeat << " stride=" << cfg.stride << " }";
  return out;
}

const char *to_string(atomic_op op) {
  switch (op) {
  case atomic_op::add:
    return "add";
  case atomic_op::sub:
    return "sub";
  case atomic_op::inc:
    return "inc";
  case atomic_op::dec:
    return "dec";
  case atomic_op::umin:
    return "umin";
  case atomic_op::umax:
    return "umax";
  case atomic_op::xchg:
    return "xchg";
  case atomic_op::cmpxchg:
    return "cmpxchg";
  case atomic_op::bit_and:
    return "bit_and";
  case atomic_op::bit_or:
    return "bit_or";
  case atomic_op::bit_xor:
    return "bit_xor";
  case atomic_op::smin:
    return "smin";
  case atomic_op::smax:
    return "smax";
  case atomic_op::fmax:
    return "fmax";
  case atomic_op::fmin:
    return "fmin";
  case atomic_op::fadd:
    return "fadd";
  case atomic_op::fsub:
    return "fsub";
  case atomic_op::fcmpxchg:
    return "fcmpxchg";
  case atomic_op::load:
    return "load";
  case atomic_op::store:
    return "store";
  case atomic_op::predec:
    return "predec";
  }
  return "<unknown>";
}

template <int N> inline bool any(simd_mask<N> m, simd_mask<N> ignore_mask) {
  simd_mask<N> m1 = 0;
  m.merge(m1, ignore_mask);
  return m.any();
}

// ----------------- The main test function
template <class T, int N, template <class, int> class ImplF, bool UseMask,
          bool UseProperties>
bool test(queue q, const Config &cfg) {
  constexpr auto op = ImplF<T, N>::atomic_op;
  using CurAtomicOpT = decltype(op);
  constexpr int n_args = ImplF<T, N>::n_args;

  std::cout << "USM Testing " << "op=" << to_string(op) << " n_args=" << n_args
            << " T=" << esimd_test::type_name<T>() << " N=" << N
            << " UseMask=" << (UseMask ? "true" : "false") << "\n\t" << cfg
            << "...";

  size_t size = cfg.start_ind + (N - 1) * cfg.stride + 1;
  T *arr = malloc_shared<T>(size, q);
  int n_threads = cfg.threads_per_group * cfg.n_groups;

  for (int i = 0; i < size; ++i) {
    arr[i] = ImplF<T, N>::init(i, cfg);
  }

  range<1> glob_rng(n_threads);
  range<1> loc_rng(cfg.threads_per_group);
  nd_range<1> rng(glob_rng, loc_rng);

  properties props{cache_hint_L1<cache_hint::uncached>,
                   cache_hint_L2<cache_hint::write_back>};

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for(rng, [=](id<1> ii) SYCL_ESIMD_KERNEL {
        int i = ii;
        simd<Toffset, N> offsets(cfg.start_ind * sizeof(T),
                                 cfg.stride * sizeof(T));
        simd_mask<N> m = 1;
        if (cfg.masked_lane < N)
          m[cfg.masked_lane] = 0;
        // barrier to achieve better contention:
        // Intra-work group barrier.
        barrier();

        // the atomic operation itself applied in a loop:
        for (int cnt = 0; cnt < cfg.repeat; ++cnt) {
          if constexpr (n_args == 0) {
            if constexpr (UseMask) {
              if constexpr (UseProperties)
                atomic_update<op>(arr, offsets, m, props);
              else
                atomic_update<op>(arr, offsets, m);
            } else {
              if constexpr (UseProperties)
                atomic_update<op>(arr, offsets, props);
              else
                atomic_update<op>(arr, offsets);
            }
          } else if constexpr (n_args == 1) {
            simd<T, N> v0 = ImplF<T, N>::arg0(i);
            atomic_update<op>(arr, offsets, v0, m);
          } else if constexpr (n_args == 2) {
            simd<T, N> new_val = ImplF<T, N>::arg0(i); // new value
            simd<T, N> exp_val = ImplF<T, N>::arg1(i); // expected value
            // do compare-and-swap in a loop until we get expected value;
            // arg0 and arg1 must provide values which guarantee the loop
            // is not endless:
            for (simd<T, N> old_val =
                     atomic_update<op>(arr, offsets, new_val, exp_val, m);
                 any(old_val < exp_val, !m);
                 old_val = atomic_update<op>(arr, offsets, new_val, exp_val, m))
              ;
          }
        }
      });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(arr, q);
    return false;
  }
  int err_cnt = 0;

  for (int i = 0; i < size; ++i) {
    T gold = ImplF<T, N>::gold(i, cfg);
    T test = arr[i];

    if ((gold != test) && (++err_cnt < 10)) {
      if (err_cnt == 1) {
        std::cout << "\n";
      }
      std::cout << "  failed at index " << i << ": " << test << " != " << gold
                << "(gold)\n";
    }
  }
  if (err_cnt > 0) {
    std::cout << "  FAILED\n  pass rate: "
              << ((float)(size - err_cnt) / (float)size) * 100.0f << "% ("
              << (size - err_cnt) << "/" << size << ")\n";
  } else {
    std::cout << " passed\n";
  }
  free(arr, q);
  return err_cnt == 0;
}

// Functions providing input and golden values for atomic operations.

static int dense_ind(int ind, int VL, const Config &cfg) {
  return (ind - cfg.start_ind) / cfg.stride;
}

static bool is_updated(int ind, int VL, const Config &cfg) {
  if ((ind < cfg.start_ind) || (((ind - cfg.start_ind) % cfg.stride) != 0)) {
    return false;
  }
  int ii = dense_ind(ind, VL, cfg);

  bool res = true;
  if (cfg.use_mask)
    res = (ii % VL) != cfg.masked_lane;
  return res;
}

// Actual "traits" for each operation.

template <class T, int N, class C, C Op> struct ImplIncBase {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 0;

  static T init(int i, const Config &cfg) { return (T)0; }

  static T gold(int i, const Config &cfg) {
    T gold = is_updated(i, N, cfg)
                 ? (T)(cfg.repeat * cfg.threads_per_group * cfg.n_groups)
                 : init(i, cfg);
    return gold;
  }
};

template <class T, int N, class C, C Op> struct ImplDecBase {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 0;
  static constexpr int base = 5;

  static T init(int i, const Config &cfg) {
    return (T)(cfg.repeat * cfg.threads_per_group * cfg.n_groups + base);
  }

  static T gold(int i, const Config &cfg) {
    T gold = is_updated(i, N, cfg) ? (T)base : init(i, cfg);
    return gold;
  }
};

// The purpose of this is to validate that floating point data is correctly
// processed.
constexpr float FPDELTA = 0.5f;

template <class T, int N, class C, C Op> struct ImplLoadBase {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 0;

  static T init(int i, const Config &cfg) { return (T)(i + FPDELTA); }

  static T gold(int i, const Config &cfg) {
    T gold = init(i, cfg);
    return gold;
  }
};

template <class T, int N>
struct ImplLoad : ImplLoadBase<T, N, atomic_op, atomic_op::load> {};
template <class T, int N>
struct ImplInc : ImplIncBase<T, N, atomic_op, atomic_op::inc> {};
template <class T, int N>
struct ImplDec : ImplDecBase<T, N, atomic_op, atomic_op::dec> {};

// Main function and test combinations.

template <int N, template <class, int> class Op, bool UseMask,
          bool UsePVCFeatures, int SignMask = (Signed | Unsigned)>
bool test_int_types(queue q, const Config &cfg) {
  bool passed = true;
  if constexpr (SignMask & Signed) {
    // Supported by LSC atomic:
    if constexpr (UsePVCFeatures)
      passed &= test<int16_t, N, Op, UseMask, UsePVCFeatures>(q, cfg);

    passed &= test<int32_t, N, Op, UseMask, UsePVCFeatures>(q, cfg);
  }

  if constexpr (SignMask & Unsigned) {
    // Supported by LSC atomic:
    if constexpr (UsePVCFeatures)
      passed &= test<uint16_t, N, Op, UseMask, UsePVCFeatures>(q, cfg);

    passed &= test<uint32_t, N, Op, UseMask, UsePVCFeatures>(q, cfg);
  }
  return passed;
}

template <int N, template <class, int> class Op, bool UseMask,
          bool UsePVCFeatures>
bool test_fp_types(queue q, const Config &cfg) {
  bool passed = true;
  passed &= test<float, N, Op, UseMask, UsePVCFeatures>(q, cfg);
  if (q.get_device().has(sycl::aspect::atomic64) &&
      q.get_device().has(sycl::aspect::fp64)) {
    passed &= test<double, N, Op, UseMask, UsePVCFeatures>(q, cfg);
  }
  return passed;
}

template <template <class, int> class Op, bool UseMask, bool UsePVCFeatures,
          int SignMask = (Signed | Unsigned)>
bool test_int_types_and_sizes(queue q, const Config &cfg) {
  bool passed = true;

  passed &= test_int_types<1, Op, UseMask, UsePVCFeatures, SignMask>(q, cfg);
  passed &= test_int_types<2, Op, UseMask, UsePVCFeatures, SignMask>(q, cfg);
  passed &= test_int_types<4, Op, UseMask, UsePVCFeatures, SignMask>(q, cfg);

  passed &= test_int_types<8, Op, UseMask, UsePVCFeatures, SignMask>(q, cfg);

  // Supported by LSC atomic:
  if constexpr (UsePVCFeatures) {
    passed &= test_int_types<16, Op, UseMask, UsePVCFeatures, SignMask>(q, cfg);
    passed &= test_int_types<32, Op, UseMask, UsePVCFeatures, SignMask>(q, cfg);
    // non power of two values are supported only in newer driver.
    // TODO: Enable this when the new driver reaches test infrastructure
    // (v27556).
#if 0
    passed &= test_int_types<12, Op, UseMask, UsePVCFeatures, SignMask>(q, cfg);
    passed &= test_int_types<33, Op, UseMask, UsePVCFeatures, SignMask>(q, cfg);
#endif
  }

  return passed;
}

template <template <class, int> class Op, bool UseMask, bool UsePVCFeatures>
bool test_fp_types_and_sizes(queue q, const Config &cfg) {
  bool passed = true;

  passed &= test_fp_types<1, Op, UseMask, UsePVCFeatures>(q, cfg);
  passed &= test_fp_types<2, Op, UseMask, UsePVCFeatures>(q, cfg);
  passed &= test_fp_types<4, Op, UseMask, UsePVCFeatures>(q, cfg);

  passed &= test_fp_types<8, Op, UseMask, UsePVCFeatures>(q, cfg);
  // Supported by LSC atomic:
  if constexpr (UsePVCFeatures) {
    passed &= test_fp_types<16, Op, UseMask, UsePVCFeatures>(q, cfg);
    passed &= test_fp_types<32, Op, UseMask, UsePVCFeatures>(q, cfg);
    // non power of two values are supported only in newer driver.
    // TODO: Enable this when the new driver reaches test infrastructure
    // (v27556).
#if 0
    passed &= test_fp_types<12, Op, UseMask, UsePVCFeatures>(q, cfg);
    passed &= test_fp_types<35, Op, UseMask, UsePVCFeatures>(q, cfg);
#endif
  }
  return passed;
}

template <bool UseMask, bool UsePVCFeatures> bool test_with_mask(queue q) {
  bool passed = true;

  Config cfg{
      11,      // int threads_per_group;
      11,      // int n_groups;
      5,       // int start_ind;
      UseMask, // int use_mask;
      1,       // int masked_lane;
      100,     // int repeat;
      111      // int stride;
  };

  passed &= test_int_types_and_sizes<ImplInc, UseMask, UsePVCFeatures>(q, cfg);
  passed &= test_int_types_and_sizes<ImplDec, UseMask, UsePVCFeatures>(q, cfg);

  // Can't easily reset input to initial state, so just 1 iteration for CAS.
  cfg.repeat = 1;
  // Decrease number of threads to reduce risk of halting kernel by the driver.
  cfg.n_groups = 7;
  cfg.threads_per_group = 3;
  // Check load/store operations
  passed &= test_int_types_and_sizes<ImplLoad, UseMask, UsePVCFeatures>(q, cfg);
  passed &= test_fp_types_and_sizes<ImplLoad, UseMask, UsePVCFeatures>(q, cfg);

  return passed;
}

template <bool UsePVCFeatures> bool test_main(queue q) {
  bool passed = true;

  constexpr const bool UseMask = true;

  passed &= test_with_mask<UseMask, UsePVCFeatures>(q);
  passed &= test_with_mask<!UseMask, UsePVCFeatures>(q);

  return passed;
}
