//==---local_acessor_atomic_smoke.cpp  - DPC++ ESIMD on-device test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This test checks local accessor atomic operations.
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc || gpu-intel-dg2
// REQUIRES-INTEL-DRIVER: lin: 26690, win: 101.4576
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//

#include "../esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <iostream>
#include <sycl/ext/intel/esimd.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

constexpr int Signed = 1;
constexpr int Unsigned = 2;

constexpr int64_t threads_per_group = 8;
constexpr int64_t n_groups = 1;
constexpr int64_t start_ind = 0;
constexpr int64_t masked_lane = 1;
constexpr int64_t repeat = 1;
constexpr int64_t stride = 1;

// ----------------- Helper functions

using LSCAtomicOp = sycl::ext::intel::esimd::native::lsc::atomic_op;
using DWORDAtomicOp = sycl::ext::intel::esimd::atomic_op;

// This macro selects between DWORD ("legacy") and LSC-based atomics.
#ifdef USE_DWORD_ATOMICS
using AtomicOp = DWORDAtomicOp;
constexpr char MODE[] = "DWORD";
#else
using AtomicOp = LSCAtomicOp;
constexpr char MODE[] = "LSC";
#endif // USE_DWORD_ATOMICS

template <class, int, template <class, int> class> class TestID;

const char *to_string(DWORDAtomicOp op) {
  switch (op) {
  case DWORDAtomicOp::add:
    return "add";
  case DWORDAtomicOp::sub:
    return "sub";
  case DWORDAtomicOp::inc:
    return "inc";
  case DWORDAtomicOp::dec:
    return "dec";
  case DWORDAtomicOp::umin:
    return "umin";
  case DWORDAtomicOp::umax:
    return "umax";
  case DWORDAtomicOp::xchg:
    return "xchg";
  case DWORDAtomicOp::cmpxchg:
    return "cmpxchg";
  case DWORDAtomicOp::bit_and:
    return "bit_and";
  case DWORDAtomicOp::bit_or:
    return "bit_or";
  case DWORDAtomicOp::bit_xor:
    return "bit_xor";
  case DWORDAtomicOp::smin:
    return "smin";
  case DWORDAtomicOp::smax:
    return "smax";
  case DWORDAtomicOp::fmax:
    return "fmax";
  case DWORDAtomicOp::fmin:
    return "fmin";
  case DWORDAtomicOp::fadd:
    return "fadd";
  case DWORDAtomicOp::fsub:
    return "fsub";
  case DWORDAtomicOp::fcmpxchg:
    return "fcmpxchg";
  case DWORDAtomicOp::load:
    return "load";
  case DWORDAtomicOp::store:
    return "store";
  case DWORDAtomicOp::predec:
    return "predec";
  }
  return "<unknown>";
}

const char *to_string(LSCAtomicOp op) {
  switch (op) {
  case LSCAtomicOp::add:
    return "lsc::add";
  case LSCAtomicOp::sub:
    return "lsc::sub";
  case LSCAtomicOp::inc:
    return "lsc::inc";
  case LSCAtomicOp::dec:
    return "lsc::dec";
  case LSCAtomicOp::umin:
    return "lsc::umin";
  case LSCAtomicOp::umax:
    return "lsc::umax";
  case LSCAtomicOp::cmpxchg:
    return "lsc::cmpxchg";
  case LSCAtomicOp::bit_and:
    return "lsc::bit_and";
  case LSCAtomicOp::bit_or:
    return "lsc::bit_or";
  case LSCAtomicOp::bit_xor:
    return "lsc::bit_xor";
  case LSCAtomicOp::smin:
    return "lsc::smin";
  case LSCAtomicOp::smax:
    return "lsc::smax";
  case LSCAtomicOp::fmax:
    return "lsc::fmax";
  case LSCAtomicOp::fmin:
    return "lsc::fmin";
  case LSCAtomicOp::fadd:
    return "lsc::fadd";
  case LSCAtomicOp::fsub:
    return "lsc::fsub";
  case LSCAtomicOp::fcmpxchg:
    return "lsc::fcmpxchg";
  case LSCAtomicOp::load:
    return "lsc::load";
  case LSCAtomicOp::store:
    return "lsc::store";
  }
  return "lsc::<unknown>";
}

template <int N> inline bool any(simd_mask<N> m, simd_mask<N> ignore_mask) {
  simd_mask<N> m1 = 0;
  m.merge(m1, ignore_mask);
  return m.any();
}

// ----------------- The main test function
template <class T, int N, template <class, int> class ImplF>
bool test(queue q) {
  constexpr auto op = ImplF<T, N>::atomic_op;
  using CurAtomicOpT = decltype(op);
  constexpr int n_args = ImplF<T, N>::n_args;

  std::cout << " Local accessor Testing mode=" << MODE
            << " op=" << to_string(op) << " T=" << esimd_test::type_name<T>()
            << " N=" << N << "\n\t"
            << "{ thr_per_group=" << threads_per_group
            << " n_groups=" << n_groups << " start_ind=" << start_ind
            << " masked_lane=" << masked_lane << " repeat=" << repeat
            << " stride=" << stride << " }...";

  constexpr size_t size = start_ind + (N - 1) * stride + 1;
  T *arr = malloc_shared<T>(size, q);

  constexpr int n_threads = threads_per_group * n_groups;

  for (int i = 0; i < size; ++i) {
    arr[i] = ImplF<T, N>::init(i);
  }

  range<1> glob_rng(n_threads);
  range<1> loc_rng(threads_per_group);
  nd_range<1> rng(glob_rng, loc_rng);

  try {
    auto e = q.submit([&](handler &cgh) {
      auto accessor = local_accessor<T, 1>(size, cgh);

      cgh.parallel_for<TestID<T, N, ImplF>>(
          rng, [=](id<1> ii) SYCL_ESIMD_KERNEL {
            int i = ii;
#ifndef USE_SCALAR_OFFSET
            simd<uint32_t, N> offsets(start_ind * sizeof(T),
                                      stride * sizeof(T));
#else
            uint32_t offsets = 0;
#endif
            simd<T, size> data;
            data.copy_from(arr);

            simd<uint32_t, size> LocalOffsets(0, sizeof(T));
            scatter<T, size>(accessor, LocalOffsets, data, 0, 1);
            simd_mask<N> m = 1;
            if (masked_lane < N)
              m[masked_lane] = 0;
            // barrier to achieve better contention:
            // Intra-work group barrier.
            barrier();

            // the atomic operation itself applied in a loop:
            for (int cnt = 0; cnt < repeat; ++cnt) {
              if constexpr (n_args == 0) {
                simd<T, N> res = atomic_update<op, T, N>(accessor, offsets, m);
              } else if constexpr (n_args == 1) {
                simd<T, N> v0 = ImplF<T, N>::arg0(i);
                atomic_update<op, T, N>(accessor, offsets, v0, m);
              } else if constexpr (n_args == 2) {
                simd<T, N> new_val = ImplF<T, N>::arg0(i); // new value
                simd<T, N> exp_val = ImplF<T, N>::arg1(i); // expected value
                // do compare-and-swap in a loop until we get expected value;
                // arg0 and arg1 must provide values which guarantee the loop
                // is not endless:
                for (auto old_val = atomic_update<op, T, N>(
                         accessor, offsets, new_val, exp_val, m);
                     any(old_val < exp_val, !m);
                     old_val = atomic_update<op, T, N>(accessor, offsets,
                                                       new_val, exp_val, m))
                  ;
              }
            }
            auto data0 = gather<T, size>(accessor, LocalOffsets, 0);
            data0.copy_to(arr);
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
    T gold = ImplF<T, N>::gold(i);
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

// ----------------- Functions providing input and golden values for atomic
// ----------------- operations.

static int dense_ind(int ind, int VL) { return (ind - start_ind) / stride; }

static bool is_updated(int ind, int VL) {
  if ((ind < start_ind) || (((ind - start_ind) % stride) != 0)) {
    return false;
  }
  int ii = dense_ind(ind, VL);
  bool res = (ii % VL) != masked_lane;
  return res;
}

// ----------------- Actual "traits" for each operation.

template <class T, int N, class C, C Op> struct ImplIncBase {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 0;

  static T init(int i) { return (T)0; }

  static T gold(int i) {
#ifndef USE_SCALAR_OFFSET
    T gold = is_updated(i, N) ? (T)(repeat * threads_per_group * n_groups)
#else
    int64_t NumLanes = (masked_lane + 1 <= N) ? (N - 1) : N;
    T gold = i == 0 ? (T)(repeat * threads_per_group * n_groups * NumLanes)
#endif
                              : init(i);
    return gold;
  }
};

template <class T, int N, class C, C Op> struct ImplDecBase {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 0;
  static constexpr int base = 5;

  static T init(int i) {
#ifndef USE_SCALAR_OFFSET
    return (T)(repeat * threads_per_group * n_groups + base);
#else
    int64_t NumLanes = (masked_lane + 1 <= N) ? (N - 1) : N;
    return (T)(repeat * threads_per_group * n_groups * NumLanes + base);
#endif
  }

  static T gold(int i) {
#ifndef USE_SCALAR_OFFSET
    T gold = is_updated(i, N) ? (T)base : init(i);
#else
    T gold = i == 0 ? (T)base : init(i);
#endif
    return gold;
  }
};

// The purpose of this is validate that floating point data is correctly
// processed.
constexpr float FPDELTA = 0.5f;

template <class T, int N, class C, C Op> struct ImplLoadBase {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 0;

  static T init(int i) { return (T)(i + FPDELTA); }

  static T gold(int i) {
    T gold = init(i);
    return gold;
  }
};

template <class T, int N, class C, C Op> struct ImplStoreBase {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 1;
  static constexpr T base = (T)(2 + FPDELTA);

  static T init(int i) { return 0; }

  static T gold(int i) {
#ifndef USE_SCALAR_OFFSET
    T gold = is_updated(i, N) ? base : init(i);
#else
    T gold = i == 0 ? base : init(i);
#endif
    return gold;
  }

  static T arg0(int i) { return base; }
};

template <class T, int N, class C, C Op> struct ImplAdd {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 1;

  static T init(int i) { return 0; }

  static T gold(int i) {
#ifndef USE_SCALAR_OFFSET
    T gold = is_updated(i, N)
                 ? (T)(repeat * threads_per_group * n_groups * (T)(1 + FPDELTA))
                 : init(i);
#else
    int64_t NumLanes = (masked_lane + 1 <= N) ? (N - 1) : N;
    T gold = i == 0 ? (T)(repeat * threads_per_group * n_groups * NumLanes *
                          (T)(1 + FPDELTA))
                    : init(i);
#endif
    return gold;
  }

  static T arg0(int i) { return (T)(1 + FPDELTA); }
};

template <class T, int N, class C, C Op> struct ImplSub {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 1;
  static constexpr T base = (T)(5 + FPDELTA);

  static T init(int i) {
#ifndef USE_SCALAR_OFFSET
    return (T)(repeat * threads_per_group * n_groups * (T)(1 + FPDELTA) + base);
#else
    int64_t NumLanes = (masked_lane + 1 <= N) ? (N - 1) : N;
    return (T)(repeat * threads_per_group * n_groups * NumLanes *
                   (T)(1 + FPDELTA) +
               base);
#endif
  }

  static T gold(int i) {
#ifndef USE_SCALAR_OFFSET
    T gold = is_updated(i, N) ? base : init(i);
#else
    T gold = i == 0 ? base : init(i);
#endif
    return gold;
  }

  static T arg0(int i) { return (T)(1 + FPDELTA); }
};

template <class T, int N, class C, C Op> struct ImplMin {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 1;

  static T init(int i) { return std::numeric_limits<T>::max(); }

  static T gold(int i) {
    T ExpectedFoundMin;
    if constexpr (std::is_signed_v<T>)
      ExpectedFoundMin = FPDELTA - (threads_per_group * n_groups - 1);
    else
      ExpectedFoundMin = FPDELTA;
#ifndef USE_SCALAR_OFFSET
    T gold = is_updated(i, N) ? ExpectedFoundMin : init(i);
#else
    T gold = i == 0 ? ExpectedFoundMin : init(i);
#endif
    return gold;
  }

  static T arg0(int i) {
    int64_t sign = std::is_signed_v<T> ? -1 : 1;
    return sign * i + FPDELTA;
  }
};

template <class T, int N, class C, C Op> struct ImplMax {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 1;

  static T init(int i) { return std::numeric_limits<T>::lowest(); }

  static T gold(int i) {
    T ExpectedFoundMax = FPDELTA;
    if constexpr (!std::is_signed_v<T>)
      ExpectedFoundMax += threads_per_group * n_groups - 1;

#ifndef USE_SCALAR_OFFSET
    T gold = is_updated(i, N)
#else
    T gold = i == 0
#endif
                 ? ExpectedFoundMax
                 : init(i);
    return gold;
  }

  static T arg0(int i) {
    int64_t sign = std::is_signed_v<T> ? -1 : 1;
    return sign * i + FPDELTA;
  }
};

#ifndef USE_DWORD_ATOMICS
// These will be redirected by API implementation to LSC ones:
template <class T, int N>
struct ImplStore : ImplStoreBase<T, N, LSCAtomicOp, LSCAtomicOp::store> {};
template <class T, int N>
struct ImplLoad : ImplLoadBase<T, N, LSCAtomicOp, LSCAtomicOp::load> {};
template <class T, int N>
struct ImplInc : ImplIncBase<T, N, LSCAtomicOp, LSCAtomicOp::inc> {};
template <class T, int N>
struct ImplDec : ImplDecBase<T, N, LSCAtomicOp, LSCAtomicOp::dec> {};
template <class T, int N>
struct ImplIntAdd : ImplAdd<T, N, LSCAtomicOp, LSCAtomicOp::add> {};
template <class T, int N>
struct ImplIntSub : ImplSub<T, N, LSCAtomicOp, LSCAtomicOp::sub> {};
template <class T, int N>
struct ImplSMin : ImplMin<T, N, LSCAtomicOp, LSCAtomicOp::smin> {};
template <class T, int N>
struct ImplUMin : ImplMin<T, N, LSCAtomicOp, LSCAtomicOp::umin> {};
template <class T, int N>
struct ImplSMax : ImplMax<T, N, LSCAtomicOp, LSCAtomicOp::smax> {};
template <class T, int N>
struct ImplUMax : ImplMax<T, N, LSCAtomicOp, LSCAtomicOp::umax> {};

template <class T, int N>
struct ImplFmin : ImplMin<T, N, DWORDAtomicOp, DWORDAtomicOp::fmin> {};
template <class T, int N>
struct ImplFmax : ImplMax<T, N, DWORDAtomicOp, DWORDAtomicOp::fmax> {};
// LCS versions:
template <class T, int N>
struct ImplLSCFmin : ImplMin<T, N, LSCAtomicOp, LSCAtomicOp::fmin> {};
template <class T, int N>
struct ImplLSCFmax : ImplMax<T, N, LSCAtomicOp, LSCAtomicOp::fmax> {};
#else
template <class T, int N>
struct ImplIntAdd : ImplAdd<T, N, DWORDAtomicOp, DWORDAtomicOp::add> {};
template <class T, int N>
struct ImplIntSub : ImplSub<T, N, DWORDAtomicOp, DWORDAtomicOp::sub> {};
template <class T, int N>
struct ImplSMin : ImplMin<T, N, DWORDAtomicOp, DWORDAtomicOp::smin> {};
template <class T, int N>
struct ImplUMin : ImplMin<T, N, DWORDAtomicOp, DWORDAtomicOp::umin> {};
template <class T, int N>
struct ImplSMax : ImplMax<T, N, DWORDAtomicOp, DWORDAtomicOp::smax> {};
template <class T, int N>
struct ImplUMax : ImplMax<T, N, DWORDAtomicOp, DWORDAtomicOp::umax> {};
template <class T, int N>
struct ImplStore : ImplStoreBase<T, N, DWORDAtomicOp, DWORDAtomicOp::store> {};
template <class T, int N>
struct ImplLoad : ImplLoadBase<T, N, DWORDAtomicOp, DWORDAtomicOp::load> {};
template <class T, int N>
struct ImplInc : ImplIncBase<T, N, DWORDAtomicOp, DWORDAtomicOp::inc> {};
template <class T, int N>
struct ImplDec : ImplDecBase<T, N, DWORDAtomicOp, DWORDAtomicOp::dec> {};
#endif // USE_DWORD_ATOMICS

template <class T, int N, class C, C Op> struct ImplCmpxchgBase {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 2;
  static constexpr T base = (T)(2 + FPDELTA);

  static T init(int i) { return base - 1; }

  static T gold(int i) {
#ifndef USE_SCALAR_OFFSET
    T gold = is_updated(i, N)
#else
    T gold = i == 0
#endif
                 ? (T)(threads_per_group * n_groups - 1 + base)
                 : init(i);
    return gold;
  }

  // "Replacement value" argument in CAS
  static inline T arg0(int i) { return i + base; }

  // "Expected value" argument in CAS
  static inline T arg1(int i) { return i + base - 1; }
};

#ifndef USE_DWORD_ATOMICS
// This will be redirected by API implementation to LSC one:
template <class T, int N>
struct ImplCmpxchg : ImplCmpxchgBase<T, N, LSCAtomicOp, LSCAtomicOp::cmpxchg> {
};
template <class T, int N>
struct ImplFcmpwr
    : ImplCmpxchgBase<T, N, DWORDAtomicOp, DWORDAtomicOp::fcmpxchg> {};
// LCS versions:
template <class T, int N>
struct ImplLSCFcmpwr
    : ImplCmpxchgBase<T, N, LSCAtomicOp, LSCAtomicOp::fcmpxchg> {};
#else
template <class T, int N>
struct ImplCmpxchg
    : ImplCmpxchgBase<T, N, DWORDAtomicOp, DWORDAtomicOp::cmpxchg> {};
#endif // USE_DWORD_ATOMICS

// ----------------- Main function and test combinations.

template <int N, template <class, int> class Op,
          int SignMask = (Signed | Unsigned)>
bool test_int_types(queue q) {
  bool passed = true;
  if constexpr (SignMask & Signed) {
#ifndef USE_DWORD_ATOMICS
    passed &= test<int16_t, N, Op>(q);
#endif

    // TODO: Enable testing of 8-bit integers is supported in HW.
    // passed &= test<int8_t, N, Op>(q);

    passed &= test<int32_t, N, Op>(q);
  }

  if constexpr (SignMask & Unsigned) {
#ifndef USE_DWORD_ATOMICS
    passed &= test<uint16_t, N, Op>(q);
#endif

    // TODO: Enable testing of 8-bit integers is supported in HW.
    // passed &= test<uint8_t, N, Op>(q);

    passed &= test<uint32_t, N, Op>(q);
  }
  return passed;
}

template <int N, template <class, int> class Op> bool test_fp_types(queue q) {
  bool passed = true;
#ifndef USE_DWORD_ATOMICS
  if constexpr (std::is_same_v<Op<sycl::half, N>, ImplLSCFmax<sycl::half, N>> ||
                std::is_same_v<Op<sycl::half, N>, ImplLSCFmin<sycl::half, N>> ||
                std::is_same_v<Op<sycl::half, N>,
                               ImplLSCFcmpwr<sycl::half, N>>) {
    auto dev = q.get_device();
    if (dev.has(sycl::aspect::fp16)) {
      passed &= test<sycl::half, N, Op>(q);
    }
  }
#endif
  passed &= test<float, N, Op>(q);
  return passed;
}

template <template <class, int> class Op, int SignMask = (Signed | Unsigned)>
bool test_int_types_and_sizes(queue q) {
  bool passed = true;

  passed &= test_int_types<1, Op, SignMask>(q);
  passed &= test_int_types<8, Op, SignMask>(q);

#ifndef USE_DWORD_ATOMICS
  passed &= test_int_types<16, Op, SignMask>(q);
  passed &= test_int_types<32, Op, SignMask>(q);
#endif // !USE_DWORD_ATOMICS

  return passed;
}

template <template <class, int> class Op>
bool test_fp_types_and_sizes(queue q) {
  bool passed = true;

  passed &= test_fp_types<1, Op>(q);
  passed &= test_fp_types<8, Op>(q);
#ifndef USE_DWORD_ATOMICS
  passed &= test_fp_types<16, Op>(q);
  passed &= test_fp_types<32, Op>(q);
#endif // !USE_DWORD_ATOMICS
  return passed;
}

int main(void) {
  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  bool passed = true;
#ifndef CMPXCHG_TEST
  passed &= test_int_types_and_sizes<ImplInc>(q);
  passed &= test_int_types_and_sizes<ImplDec>(q);

  passed &= test_int_types_and_sizes<ImplIntAdd>(q);
  passed &= test_int_types_and_sizes<ImplIntSub>(q);

  passed &= test_int_types_and_sizes<ImplSMax, Signed>(q);
  passed &= test_int_types_and_sizes<ImplSMin, Signed>(q);

  passed &= test_int_types_and_sizes<ImplUMax, Unsigned>(q);
  passed &= test_int_types_and_sizes<ImplUMin, Unsigned>(q);

#ifndef USE_DWORD_ATOMICS
  passed &= test_fp_types_and_sizes<ImplLSCFmax>(q);
  passed &= test_fp_types_and_sizes<ImplLSCFmin>(q);
#endif // USE_DWORD_ATOMICS
  passed &= test_int_types_and_sizes<ImplLoad>(q);
  passed &= test_fp_types_and_sizes<ImplLoad>(q);
#ifndef USE_SCALAR_OFFSET
  passed &= test_int_types_and_sizes<ImplStore>(q);
  passed &= test_fp_types_and_sizes<ImplStore>(q);
#endif
#else // CMPXCHG_TEST
  passed &= test_int_types_and_sizes<ImplCmpxchg>(q);
#ifndef USE_DWORD_ATOMICS
  passed &= test_fp_types_and_sizes<ImplFcmpwr>(q);
  passed &= test_fp_types_and_sizes<ImplLSCFcmpwr>(q);
#endif // USE_DWORD_ATOMICS
#endif // CMPXCHG_TEST

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
