// REQUIRES: gpu
// UNSUPPORTED: cuda || hip || esimd_emulator
//
// RUN: %clangxx -fsycl %s -o %t.1.out
// RUN: %GPU_RUN_PLACEHOLDER %t.1.out
//
// Vary the test case by forcing inlining of the functions with slm_allocator:
// RUN: %clangxx -fsycl -DFORCE_INLINE %s -o %t.2.out
// RUN: %GPU_RUN_PLACEHOLDER %t.2.out

// Check that SLM frame offset of a function foo called from two kernels Test1
// and Test2 is the maximum of the SLM size used in both kernels.

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

using T = uint32_t;

constexpr int SLM_IN_KERNEL1 = 16;
constexpr int SLM_IN_KERNEL2 = 32;
constexpr int SLM_IN_FOO = 4;
constexpr int LOCAL_SIZE = 2;
constexpr int GLOBAL_SIZE = 2;

template <class T> void scalar_store(T *addr, T val) {
  scatter<T, 1>(addr, simd<uint32_t, 1>(0), val);
}

#ifdef FORCE_INLINE
constexpr bool force_inline = true;
inline
    __attribute__((always_inline))
#else
constexpr bool force_inline = false;
__attribute__((noinline))
#endif // FORCE_INLINE
    void
    foo(int local_id, T *addr) {
  slm_allocator<SLM_IN_FOO> a;
  uint32_t slm_off = a.get_offset();

  if (local_id == 0) {
    scalar_store(addr, (T)slm_off);
  }
}

int main(void) {
  queue q;
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  std::cout << "force_inline=" << force_inline << "\n";
  auto ctxt = q.get_context();

  T *arr = malloc_shared<T>(1, dev, ctxt);

  auto e = q.submit([&](handler &cgh) {
    cgh.parallel_for<class Test1>(nd_range<1>(GLOBAL_SIZE, LOCAL_SIZE),
                                  [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
                                    slm_init(SLM_IN_KERNEL1);
                                    int local_id = ndi.get_local_linear_id();
                                    foo(local_id, arr);
                                  });
  });
  e.wait();

  e = q.submit([&](handler &cgh) {
    cgh.parallel_for<class Test2>(nd_range<1>(GLOBAL_SIZE, LOCAL_SIZE),
                                  [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
                                    slm_init(SLM_IN_KERNEL2);
                                    int local_id = ndi.get_local_linear_id();
                                    foo(local_id, arr);
                                  });
  });
  e.wait();

  T gold = std::max(SLM_IN_KERNEL1, SLM_IN_KERNEL2);
  T test = *arr;
  int err_cnt = 0;

  if (test != gold) {
    if (++err_cnt < 10) {
      std::cerr << "*** ERROR: " << test << " != " << gold << "(gold)\n";
    }
  } else {
    std::cout << test << " == " << gold << "(gold)\n";
  }
  std::cout << (err_cnt ? "FAILED\n" : "Passed\n");
  return err_cnt ? 1 : 0;
}
