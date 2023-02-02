// REQUIRES: gpu
// UNSUPPORTED: cuda || hip || esimd_emulator
//
// RUN: %clangxx -fsycl %s -o %t.1.out
// RUN: %GPU_RUN_PLACEHOLDER %t.1.out
//
// Vary the test case by forcing inlining of the functions with slm_allocator:
// RUN: %clangxx -fsycl -DFORCE_INLINE %s -o %t.2.out
// RUN: %GPU_RUN_PLACEHOLDER %t.2.out

// This is end-to-end test for the slm_allocator API used together with the
// slm_init. The call graph is:
//         Test1(kernel) - uses slm_init(SLM_IN_KERNEL)
//         /    \
//        /      v
//       /       bar - uses slm_allocator(SLM_IN_BAR)
//      v
//      foo - uses slm_allocator(SLM_IN_FOO)
// Test1 kernel SLM usage is SLM_IN_KERNEL + max(SLM_IN_BAR, SLM_IN_FOO).
// SLM offset returned by the slm_allocator in foo and bar is the same and is
// SLM_IN_KERNEL bytes.
// Bar uses slightly bigger SLM frame than foo. It modifies values (adds 10) in
// SLM resulting from foo, plus appends couple more '100's.

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

using T = int;
constexpr int LOCAL_SIZE = 4;
constexpr int GLOBAL_SIZE = 8;

constexpr int NUM_WGS = GLOBAL_SIZE / LOCAL_SIZE;

constexpr int ELEM_SIZE = sizeof(T);

constexpr int SLM_IN_KERNEL = LOCAL_SIZE * ELEM_SIZE;
constexpr int SLM_IN_FOO = LOCAL_SIZE * ELEM_SIZE;
constexpr int BAR_EXTRA_ELEMS = 2;
constexpr int SLM_IN_BAR = SLM_IN_FOO + BAR_EXTRA_ELEMS * ELEM_SIZE;
constexpr int SLM_TOTAL = SLM_IN_KERNEL + std::max(SLM_IN_FOO, SLM_IN_BAR);
constexpr int BAR_MARKER1 = 10;
constexpr int BAR_MARKER2 = 100;

#ifdef FORCE_INLINE
constexpr bool force_inline = true;
inline
    __attribute__((always_inline))
#else
constexpr bool force_inline = false;
__attribute__((noinline))
#endif // FORCE_INLINE
    void
    foo(int local_id) {
  slm_allocator<SLM_IN_FOO> a;
  uint32_t slm_off = a.get_offset();
  // write data chunk "Y":
  slm_scalar_store(slm_off + local_id * ELEM_SIZE, (T)local_id);
}

#ifdef FORCE_INLINE
inline
    __attribute__((always_inline))
#else
__attribute__((noinline))
#endif // FORCE_INLINE
    void
    bar(int local_id) {
  slm_allocator<SLM_IN_BAR> a;
  uint32_t slm_off = a.get_offset();
  uint32_t off = slm_off + local_id * ELEM_SIZE;
  T v = slm_scalar_load<T>(off);
  // update data chunk "Y":
  slm_scalar_store(off, v + BAR_MARKER1);

  if (local_id == 0) {
    for (int i = 0; i < BAR_EXTRA_ELEMS; i++) {
      // write data chunk "Z":
      slm_scalar_store((2 * LOCAL_SIZE + i) * ELEM_SIZE, (T)BAR_MARKER2);
    }
  }
}

int main(void) {
  queue q;
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  std::cout << "force_inline=" << force_inline << "\n";
  auto ctxt = q.get_context();
  uint32_t size = SLM_TOTAL * NUM_WGS / ELEM_SIZE;

  T *arr = malloc_shared<T>(size, dev, ctxt);

  auto e = q.submit([&](handler &cgh) {
    cgh.parallel_for<class Test1>(
        nd_range<1>(GLOBAL_SIZE, LOCAL_SIZE),
        [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
          slm_init(SLM_IN_KERNEL);
          int local_id = ndi.get_local_linear_id();
          int group_id = ndi.get_group_linear_id();

          // write data chunk "X":
          slm_scalar_store(local_id * ELEM_SIZE, local_id);
          barrier();

          foo(local_id);
          barrier();

          bar(local_id);
          barrier();

          // copy data from SLM to the output for further verification
          if (local_id == 0) {
            uint32_t group_off = SLM_TOTAL * group_id;

            for (int i = 0; i < SLM_TOTAL / ELEM_SIZE; i++) {
              uint32_t slm_off = i * ELEM_SIZE;
              uint32_t mem_off = group_off + slm_off;
              scatter(arr, simd<uint32_t, 1>(mem_off),
                      simd<T, 1>(slm_scalar_load<T>(slm_off)));
            }
          }
        });
  });
  e.wait();

  for (int i = 0; i < NUM_WGS * SLM_TOTAL / ELEM_SIZE; i++) {
    std::cout << " " << arr[i];
    if ((i + 1) % 10 == 0) {
      std::cout << "\n";
    }
  }
  std::cout << "\n";
  int err_cnt = 0;

  for (int g = 0; g < NUM_WGS; g++) {
    uint32_t group_off = SLM_TOTAL * g / ELEM_SIZE;
    for (int i = 0; i < LOCAL_SIZE; i++) {
      int ind = group_off + i;

      // check data copied from kernel's SLM frame ("X")
      auto test = arr[ind];
      auto gold = i;

      if (test != gold) {
        if (++err_cnt < 10) {
          std::cerr << "*** ERROR (X) at " << ind << ": " << test
                    << " != " << gold << " (gold)\n";
        }
      }
      // check data copied from the overlapping part of foo's and bar's SLM
      // frames - "Y"
      ind = ind + LOCAL_SIZE; // shift to the foo/bar SLM frame
      test = arr[ind];
      gold = i + BAR_MARKER1;

      if (test != gold) {
        if (++err_cnt < 10) {
          std::cerr << "*** ERROR (Y) at " << ind << ": " << test
                    << " != " << gold << " (gold)\n";
        }
      }
    }
    // now check data written by bar past the overlapping part of foo/bar SLM
    // frame - "Z"
    for (int i = 0; i < BAR_EXTRA_ELEMS; i++) {
      int ind =
          group_off + 2 /*kernel's and foo's SLM frames*/ * LOCAL_SIZE + i;
      auto test = arr[ind];
      auto gold = BAR_MARKER2;

      if (test != gold) {
        if (++err_cnt < 10) {
          std::cerr << "*** ERROR (Z) at " << ind << ": " << test
                    << " != " << gold << " (gold)\n";
        }
      }
    }
  }
  free(arr, ctxt);
  std::cout << (err_cnt ? "FAILED\n" : "Passed\n");
  return err_cnt ? 1 : 0;
}
