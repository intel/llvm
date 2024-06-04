//
// Windows doesn't yet have full shutdown().
// UNSUPPORTED: ze_debug && windows
// REQUIRES-INTEL-DRIVER: lin: 28454, win: 101.5333
//
// RUN: %{build} -o %t.1.out
// RUN: %{run} %t.1.out
//
// Vary the test case by forcing inlining of the functions with slm_allocator:
// RUN: %{build} -DFORCE_INLINE -o %t.2.out
// RUN: %{run} %t.2.out

// Check if the test sill passes with O0
// RUN: %{build} -O0 -o %t.3.out
// RUN: %{run} %t.3.out

// Checks validity of SLM frame offsets in case of complex call graph with two
// kernels and 2 functions all using SLM, and one of the functions using two
// slm_allocator objects with nested liveness ranges.

// Hierarchy of SLM frames:
//   N1      N2
//   | \     |
//   |  \    |
//   v   \   |
//   X    \  |
//   | \   \ |
//   |  \   \|
//   v   \---v
//   Y       Z
//
// SLM offsets are expected to be:
// --- Kernel0
// X  - N1
// Y  - N1 + X
// Z  - N1 // this is because Z (bar) is inlined into X (foo) and into
//         // N1 (kernel1), and execution of the second inlined scope
//         // allocation and offset recording into the result happens last.
// --- Kernel2
// X  - 0 (not reachable, offset not updated in the result)
// Y  - 0 (not reachable, offset not updated in the result)
// Z  - N2

#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/usm.hpp>

#include <cstring>
#include <iostream>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

using T = uint32_t;

constexpr int SLM_N1 = 7;
constexpr int SLM_N2 = 1;
constexpr int SLM_X = 8;
constexpr int SLM_Y = 16;
constexpr int SLM_Z = 4;

constexpr int LOCAL_SIZE = 2;
constexpr int GLOBAL_SIZE = 2;

template <class T> void scalar_store(T *base, uint32_t off, T val) {
  scatter<T, 1>(base, simd<uint32_t, 1>(off * sizeof(T)), val);
}

// Result array format
// |---- kernel0 ----|  |---- kernel2 ----|
// x_off, y_off, z_off, x_off, y_off, z_off

// Offsets in the result sub-array, to store each checked SLM frame offset at.
enum { x_off_ind, y_off_ind, z_off_ind, num_offs };

// Offsets of sub-arrays
enum { kernel0_base = x_off_ind, kernel1_base = num_offs };

#define STORE_SLM_OFF(ID, off)                                                 \
  if (local_id == 0) {                                                         \
    scalar_store(out, base + ID##_off_ind, off);                               \
  }

#ifdef FORCE_INLINE
constexpr bool force_inline = true;
#define INLINE_CTL inline __attribute__((always_inline))
#else
constexpr bool force_inline = false;
#define INLINE_CTL __attribute__((noinline))
#endif // FORCE_INLINE

INLINE_CTL void bar(int local_id, T *out, unsigned base) {
  slm_allocator<SLM_Z> a;
  unsigned z_off = a.get_offset();
  STORE_SLM_OFF(z, z_off);
}

INLINE_CTL void foo(int local_id, T *out, unsigned base) {
  slm_allocator<SLM_X> a;
  unsigned x_off = a.get_offset();
  STORE_SLM_OFF(x, x_off);
  bar(local_id, out, base);
  {
    slm_allocator<SLM_Y> b;
    unsigned y_off = b.get_offset();
    STORE_SLM_OFF(y, y_off);
  }
}

int main(void) {
  queue q;
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  std::cout << "force_inline=" << force_inline << "\n";
  auto ctxt = q.get_context();

  constexpr int num_kernels = 2;
  T *arr = malloc_shared<T>(num_kernels * num_offs, dev, ctxt);
  std::memset(arr, 0, num_kernels * num_offs * sizeof(T));

  auto e = q.submit([&](handler &cgh) {
    cgh.parallel_for<class Kernel0>(nd_range<1>(GLOBAL_SIZE, LOCAL_SIZE),
                                    [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
                                      slm_init(SLM_N1);
                                      int local_id = ndi.get_local_linear_id();
                                      foo(local_id, arr, kernel0_base);
                                      bar(local_id, arr, kernel0_base);
                                    });
  });
  e.wait();

  e = q.submit([&](handler &cgh) {
    cgh.parallel_for<class Kernel2>(nd_range<1>(GLOBAL_SIZE, LOCAL_SIZE),
                                    [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
                                      slm_init(SLM_N2);
                                      int local_id = ndi.get_local_linear_id();
                                      bar(local_id, arr, kernel1_base);
                                    });
  });
  e.wait();

  T gold_arr[num_kernels * num_offs];
  T *gold_arr0 = &gold_arr[kernel0_base];
  T *gold_arr1 = &gold_arr[kernel1_base];

  // For kernel0 inline/no-inline results are the same for X and Y:
  // X  - N1
  // Y  - N1 + X
  // Z  - max(N1 + X, N2)
  gold_arr0[x_off_ind] = SLM_N1;
  gold_arr0[y_off_ind] = SLM_N1 + SLM_X;
  gold_arr0[z_off_ind] = SLM_N1;

  // For kernel1 inline/no-inline results are the same for X and Y:
  // X  - 0
  // Y  - 0
  gold_arr1[x_off_ind] = 0;
  gold_arr1[y_off_ind] = 0;
  gold_arr1[z_off_ind] = SLM_N2;

  T *test_arr = arr;
  int err_cnt = 0;

  T kernel_bases[num_kernels] = {kernel0_base, kernel1_base};

  for (int k = 0; k < num_kernels; k++) {
    std::cout << "Kernel " << k << "\n";

    for (int i = 0; i < num_offs; i++) {
      T test = test_arr[kernel_bases[k] + i];
      T gold = gold_arr[kernel_bases[k] + i];

      if (test != gold) {
        ++err_cnt;
        std::cerr << "  *** ERROR at [" << i << "]: " << test << " != " << gold
                  << "(gold)\n";
      } else {
        std::cout << "  [" << i << "]: " << test << " == " << gold
                  << "(gold)\n";
      }
    }
  }
  std::cout << (err_cnt ? "FAILED\n" : "Passed\n");
  free(arr, ctxt);
  return err_cnt ? 1 : 0;
}
