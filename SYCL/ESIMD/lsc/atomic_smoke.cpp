//==---------------- atomic_smoke.cpp  - DPC++ ESIMD on-device test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This test checks LSC atomic operations.
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-pvc
// TODO: esimd_emulator fails due to random timeouts (_XFAIL_: esimd_emulator)
// UNSUPPORTED: esimd_emulator
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "../esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <iostream>
#include <sycl/ext/intel/esimd.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

#ifdef USE_64_BIT_OFFSET
typedef uint64_t Toffset;
#else
typedef uint32_t Toffset;
#endif

struct Config {
  int threads_per_group;
  int n_groups;
  int start_ind;
  int masked_lane;
  int repeat;
  int stride;
};

#ifndef PREFER_FULL_BARRIER
#define PREFER_FULL_BARRIER 0
#endif // PREFER_FULL_BARRIER

#if PREFER_FULL_BARRIER && !defined(USE_DWORD_ATOMICS)
#define USE_FULL_BARRIER 1
#else
#define USE_FULL_BARRIER 0
#endif

// ----------------- Helper functions

std::ostream &operator<<(std::ostream &out, const Config &cfg) {
  out << "{ thr_per_group=" << cfg.threads_per_group
      << " n_groups=" << cfg.n_groups << " start_ind=" << cfg.start_ind
      << " masked_lane=" << cfg.masked_lane << " repeat=" << cfg.repeat
      << " stride=" << cfg.stride << " }";
  return out;
}

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

#ifndef USE_DWORD_ATOMICS
#if USE_FULL_BARRIER
uint32_t atomic_load(uint32_t *addr) {
  auto v = atomic_update<LSCAtomicOp::load, uint32_t, 1>(addr, 0, 1);
  return v[0];
}
#endif // USE_FULL_BARRIER
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
  case DWORDAtomicOp::fcmpxchg:
    return "fcmpxchg";
  case DWORDAtomicOp::fadd:
    return "fadd";
  case DWORDAtomicOp::fsub:
    return "fsub";
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
  case LSCAtomicOp::fcmpxchg:
    return "lsc::fcmpxchg";
  case LSCAtomicOp::fadd:
    return "lsc::fadd";
  case LSCAtomicOp::fsub:
    return "lsc::fsub";
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
bool test(queue q, const Config &cfg) {
  constexpr auto op = ImplF<T, N>::atomic_op;
  using CurAtomicOpT = decltype(op);
  constexpr int n_args = ImplF<T, N>::n_args;

  std::cout << "Testing "
            << "mode=" << MODE << " op=" << to_string(op)
            << " full barrier=" << (USE_FULL_BARRIER ? "yes" : "no")
            << " T=" << typeid(T).name() << " N=" << N
            << "\n"
               "    "
            << cfg << "...";

  size_t size = cfg.start_ind + (N - 1) * cfg.stride + 1;
  T *arr = malloc_shared<T>(size, q);
#if USE_FULL_BARRIER
  uint32_t *flag_ptr = malloc_shared<uint32_t>(1, q);
  *flag_ptr = 0;
#endif // USE_FULL_BARRIER
  int n_threads = cfg.threads_per_group * cfg.n_groups;

  for (int i = 0; i < size; ++i) {
    arr[i] = ImplF<T, N>::init(i, cfg);
  }

  range<1> glob_rng(n_threads);
  range<1> loc_rng(cfg.threads_per_group);
  nd_range<1> rng(glob_rng, loc_rng);

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<TestID<T, N, ImplF>>(
          rng, [=](id<1> ii) SYCL_ESIMD_KERNEL {
            int i = ii;
            simd<Toffset, N> offsets(cfg.start_ind * sizeof(T),
                                     cfg.stride * sizeof(T));
            simd_mask<N> m = 1;
            m[cfg.masked_lane] = 0;
        // barrier to achieve better contention:
#if USE_FULL_BARRIER
            // Full global barrier, works only with LSC atomics
            // (+ ND range should fit into the available h/w threads).
            atomic_update<LSCAtomicOp::inc, uint32_t, 1>(flag_ptr, 0, 1);
            for (uint32_t x = atomic_load(flag_ptr); x < n_threads;
                 x = atomic_load(flag_ptr))
              ;
#else
        // Intra-work group barrier.
        barrier();
#endif // USE_FULL_BARRIER

            // the atomic operation itself applied in a loop:
            for (int cnt = 0; cnt < cfg.repeat; ++cnt) {
              if constexpr (n_args == 0) {
                atomic_update<op>(arr, offsets, m);
              } else if constexpr (n_args == 1) {
                simd<T, N> v0 = ImplF<T, N>::arg0(i);
                atomic_update<op>(arr, offsets, v0, m);
              } else if constexpr (n_args == 2) {
                simd<T, N> new_val = ImplF<T, N>::arg0(i); // new value
                simd<T, N> exp_val = ImplF<T, N>::arg1(i); // expected value
                // do compare-and-swap in a loop until we get expected value;
                // arg0 and arg1 must provide values which guarantee the loop
                // is not endless:
                for (auto old_val =
                         atomic_update<op>(arr, offsets, new_val, exp_val, m);
                     any(old_val < exp_val, !m);
                     old_val =
                         atomic_update<op>(arr, offsets, new_val, exp_val, m))
                  ;
              }
            }
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(arr, q);
#if USE_FULL_BARRIER
    free(flag_ptr, q);
#endif // USE_FULL_BARRIER
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
#if USE_FULL_BARRIER
  free(flag_ptr, q);
#endif // USE_FULL_BARRIER
  return err_cnt == 0;
}

// ----------------- Functions providing input and golden values for atomic
// ----------------- operations.

static int dense_ind(int ind, int VL, const Config &cfg) {
  return (ind - cfg.start_ind) / cfg.stride;
}

static bool is_updated(int ind, int VL, const Config &cfg) {
  if ((ind < cfg.start_ind) || (((ind - cfg.start_ind) % cfg.stride) != 0)) {
    return false;
  }
  int ii = dense_ind(ind, VL, cfg);
  bool res = (ii % VL) != cfg.masked_lane;
  return res;
}

// ----------------- Actual "traits" for each operation.

template <class T, int N> struct ImplInc {
  static constexpr AtomicOp atomic_op = AtomicOp::inc;
  static constexpr int n_args = 0;

  static T init(int i, const Config &cfg) { return (T)0; }

  static T gold(int i, const Config &cfg) {
    T gold = is_updated(i, N, cfg)
                 ? (T)(cfg.repeat * cfg.threads_per_group * cfg.n_groups)
                 : init(i, cfg);
    return gold;
  }
};

template <class T, int N> struct ImplDec {
  static constexpr AtomicOp atomic_op = AtomicOp::dec;
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

// The purpose of this is validate that floating point data is correctly
// processed.
constexpr float FPDELTA = 0.5f;

template <class T, int N> struct ImplLoad {
  static constexpr AtomicOp atomic_op = AtomicOp::load;
  static constexpr int n_args = 0;

  static T init(int i, const Config &cfg) { return (T)(i + FPDELTA); }

  static T gold(int i, const Config &cfg) {
    T gold = init(i, cfg);
    return gold;
  }
};

template <class T, int N> struct ImplStore {
  static constexpr AtomicOp atomic_op = AtomicOp::store;
  static constexpr int n_args = 1;
  static constexpr T base = (T)(2 + FPDELTA);

  static T init(int i, const Config &cfg) { return 0; }

  static T gold(int i, const Config &cfg) {
    T gold = is_updated(i, N, cfg) ? base : init(i, cfg);
    return gold;
  }

  static T arg0(int i) { return base; }
};

template <class T, int N, class C, C Op> struct ImplAdd {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 1;

  static T init(int i, const Config &cfg) { return 0; }

  static T gold(int i, const Config &cfg) {
    T gold = is_updated(i, N, cfg) ? (T)(cfg.repeat * cfg.threads_per_group *
                                         cfg.n_groups * (T)(1 + FPDELTA))
                                   : init(i, cfg);
    return gold;
  }

  static T arg0(int i) { return (T)(1 + FPDELTA); }
};

template <class T, int N, class C, C Op> struct ImplSub {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 1;
  static constexpr T base = (T)(5 + FPDELTA);

  static T init(int i, const Config &cfg) {
    return (T)(cfg.repeat * cfg.threads_per_group * cfg.n_groups *
                   (T)(1 + FPDELTA) +
               base);
  }

  static T gold(int i, const Config &cfg) {
    T gold = is_updated(i, N, cfg) ? base : init(i, cfg);
    return gold;
  }

  static T arg0(int i) { return (T)(1 + FPDELTA); }
};

template <class T, int N, class C, C Op> struct ImplMin {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 1;
  static constexpr T MIN = (T)(1 + FPDELTA);

  static T init(int i, const Config &cfg) {
    return (T)(cfg.threads_per_group * cfg.n_groups + MIN + 1);
  }

  static T gold(int i, const Config &cfg) {
    T gold = is_updated(i, N, cfg) ? (T)MIN : init(i, cfg);
    return gold;
  }

  static T arg0(int i) { return i + MIN; }
};

template <class T, int N, class C, C Op> struct ImplMax {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 1;
  static constexpr T base = (T)(5 + FPDELTA);

  static T init(int i, const Config &cfg) { return (T)FPDELTA; }

  static T gold(int i, const Config &cfg) {
    T gold = is_updated(i, N, cfg)
                 ? (T)(cfg.threads_per_group * cfg.n_groups - 1 + FPDELTA)
                 : init(i, cfg);
    return gold;
  }

  static T arg0(int i) { return (T)(i + FPDELTA); }
};

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

#ifndef USE_DWORD_ATOMICS
// These will be redirected by API implementation to LSC ones:
template <class T, int N>
struct ImplFadd : ImplAdd<T, N, DWORDAtomicOp, DWORDAtomicOp::fadd> {};
template <class T, int N>
struct ImplFsub : ImplSub<T, N, DWORDAtomicOp, DWORDAtomicOp::fsub> {};
template <class T, int N>
struct ImplFmin : ImplMin<T, N, DWORDAtomicOp, DWORDAtomicOp::fmin> {};
template <class T, int N>
struct ImplFmax : ImplMax<T, N, DWORDAtomicOp, DWORDAtomicOp::fmax> {};
// LCS versions:
template <class T, int N>
struct ImplLSCFadd : ImplAdd<T, N, LSCAtomicOp, LSCAtomicOp::fadd> {};
template <class T, int N>
struct ImplLSCFsub : ImplSub<T, N, LSCAtomicOp, LSCAtomicOp::fsub> {};
template <class T, int N>
struct ImplLSCFmin : ImplMin<T, N, LSCAtomicOp, LSCAtomicOp::fmin> {};
template <class T, int N>
struct ImplLSCFmax : ImplMax<T, N, LSCAtomicOp, LSCAtomicOp::fmax> {};
#endif // USE_DWORD_ATOMICS

template <class T, int N, class C, C Op> struct ImplCmpxchgBase {
  static constexpr C atomic_op = Op;
  static constexpr int n_args = 2;
  static constexpr T base = (T)(2 + FPDELTA);

  static T init(int i, const Config &cfg) { return base - 1; }

  static T gold(int i, const Config &cfg) {
    T gold = is_updated(i, N, cfg)
                 ? (T)(cfg.threads_per_group * cfg.n_groups - 1 + base)
                 : init(i, cfg);
    return gold;
  }

  // "Replacement value" argument in CAS
  static inline T arg0(int i) { return i + base; }

  // "Expected value" argument in CAS
  static inline T arg1(int i) { return i + base - 1; }
};

template <class T, int N>
struct ImplCmpxchg
    : ImplCmpxchgBase<T, N, DWORDAtomicOp, DWORDAtomicOp::cmpxchg> {};

#ifndef USE_DWORD_ATOMICS
// This will be redirected by API implementation to LSC one:
template <class T, int N>
struct ImplFcmpwr
    : ImplCmpxchgBase<T, N, DWORDAtomicOp, DWORDAtomicOp::fcmpxchg> {};
// LCS versions:
template <class T, int N>
struct ImplLSCFcmpwr
    : ImplCmpxchgBase<T, N, LSCAtomicOp, LSCAtomicOp::fcmpxchg> {};
#endif // USE_DWORD_ATOMICS

// ----------------- Main function and test combinations.

template <int N, template <class, int> class Op>
bool test_int_types(queue q, const Config &cfg) {
  bool passed = true;
  passed &= test<int32_t, N, Op>(q, cfg);
  passed &= test<uint32_t, N, Op>(q, cfg);
  passed &= test<int64_t, N, Op>(q, cfg);
  passed &= test<uint64_t, N, Op>(q, cfg);
  if constexpr (!std::is_same_v<unsigned long, uint64_t> &&
                !std::is_same_v<unsigned long, uint32_t>) {
    // Test 'long' types if they are not aliased with int types tested above.
    passed &= test<unsigned long, N, Op>(q, cfg);
    passed &= test<signed long, N, Op>(q, cfg);
  }
  return passed;
}

int main(void) {
  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  Config cfg{
      11,  // int threads_per_group;
      11,  // int n_groups;
      5,   // int start_ind;
      5,   // int masked_lane;
      100, // int repeat;
      111  // int stride;
  };

  bool passed = true;
#ifndef CMPXCHG_TEST
  // Template params:
  // - element type, simd size, threads per group, num groups, atomic op,
  //   verification function, argument generation functions...
  // Actual params:
  // - queue, start index in data, masked lane, repeat count
  passed &= test_int_types<8, ImplInc>(q, cfg);
  passed &= test_int_types<8, ImplDec>(q, cfg);

  // TODO: support sizes other than 8 when compiler is fixed
  // passed &= test<uint32_t, 16, ImplDec>(q, cfg);
  // TODO: support 16-bit types when compiler is fixed
  // passed &= test<uint16_t, 8, ImplDec>(q, cfg);

  passed &= test_int_types<8, ImplIntAdd>(q, cfg);
  passed &= test_int_types<8, ImplIntSub>(q, cfg);

  // TODO: this crashes vc-intrinsics
  // passed &= test<int, 8, ImplSMax>(q, cfg);
  // passed &= test<int, 8, ImplSMin>(q, cfg);

  passed &= test<uint32_t, 8, ImplUMax>(q, cfg);
  passed &= test<uint32_t, 8, ImplUMin>(q, cfg);

  // TODO: add other operations

#ifndef USE_DWORD_ATOMICS
  passed &= test<float, 8, ImplFadd>(q, cfg);
  passed &= test<float, 8, ImplFsub>(q, cfg);
  passed &= test<float, 16, ImplFadd>(q, cfg);
  passed &= test<float, 16, ImplFsub>(q, cfg);
  passed &= test<float, 32, ImplFadd>(q, cfg);
  passed &= test<float, 32, ImplFsub>(q, cfg);

  // TODO: support sycl::half when compiler is fixed
  // passed &= test<sycl::half, 8, ImplFmin>(q, cfg);
  passed &= test<float, 8, ImplFmax>(q, cfg);
  passed &= test<float, 16, ImplFmin>(q, cfg);
  passed &= test<float, 16, ImplFmax>(q, cfg);
  passed &= test<float, 32, ImplFmin>(q, cfg);
  // passed &= test<sycl::half, 32, ImplFmax>(q, cfg);

  passed &= test<float, 8, ImplLSCFmax>(q, cfg);
  passed &= test<float, 16, ImplLSCFmin>(q, cfg);
  passed &= test<float, 16, ImplLSCFmax>(q, cfg);
  passed &= test<float, 32, ImplLSCFmin>(q, cfg);
#endif // USE_DWORD_ATOMICS
#else  // CMPXCHG_TEST
  // Can't easily reset input to initial state, so just 1 iteration for CAS.
  cfg.repeat = 1;
  // Decrease number of threads to reduce risk of halting kernel by the driver.
  cfg.n_groups = 7;
  cfg.threads_per_group = 3;
  passed &= test_int_types<8, ImplCmpxchg>(q, cfg);
#ifndef USE_DWORD_ATOMICS
  passed &= test<float, 8, ImplFcmpwr>(q, cfg);
  passed &= test<float, 8, ImplLSCFcmpwr>(q, cfg);
#endif // USE_DWORD_ATOMICS
#endif // CMPXCHG_TEST

  // Check load/store operations
  passed &= test_int_types<8, ImplLoad>(q, cfg);
  if (q.get_backend() != sycl::backend::ext_intel_esimd_emulator)
    passed &= test_int_types<8, ImplStore>(q, cfg);
#ifndef USE_DWORD_ATOMICS
  if (q.get_backend() != sycl::backend::ext_intel_esimd_emulator)
    passed &= test<float, 8, ImplStore>(q, cfg);
#endif // USE_DWORD_ATOMICS
  // TODO: check double other vector lengths in LSC mode.

  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}
