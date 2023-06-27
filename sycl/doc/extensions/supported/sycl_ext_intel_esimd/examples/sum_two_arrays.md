## Sum elements in two arrays: a[i] += b[i].

Compile and run:
```bash
> clang++ -fsycl sum_two_arrays.cpp

> ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./a.out
Running on Intel(R) UHD Graphics 630
Passed
```

Source code:
```C++
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <iostream>

#if !defined(USE_SYCL) && !defined(USE_ESIMD)
#define USE_ESIMD
#endif

using namespace sycl;
using namespace sycl::ext::intel::esimd;

inline auto createExceptionHandler() {
  return [](exception_list l) {
    for (auto ep : l) {
      try {
        std::rethrow_exception(ep);
      } catch (sycl::exception &e0) {
        std::cout << "sycl::exception: " << e0.what() << std::endl;
      } catch (std::exception &e) {
        std::cout << "std::exception: " << e.what() << std::endl;
      } catch (...) {
        std::cout << "generic exception\n";
      }
    }
  };
}

struct usm_deleter {
  queue q;
  void operator()(void *ptr) {
    if (ptr)
      sycl::free(ptr, q);
  }
};

int main() {
  constexpr unsigned Size = 128;
  constexpr unsigned VL = 32;
  int err_cnt = 0;

  try {
    queue q(gpu_selector_v, createExceptionHandler());
    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

    float *a = malloc_shared<float>(Size, q); // USM memory for A
    float *b = new float[Size];               // B uses HOST memory
    buffer<float, 1> buf_b(b, Size);

    std::unique_ptr<float, usm_deleter> guard_a(a, usm_deleter{ q });
    std::unique_ptr<float> guard_b(b);

    for (unsigned i = 0; i != Size; i++)
      a[i] = b[i] = i;

    q.submit([&](handler &cgh) {
      auto acc_b = buf_b.get_access<access::mode::read>(cgh);
#ifdef USE_ESIMD
      cgh.parallel_for(Size / VL, [=](id<1> i) [[intel::sycl_explicit_simd]] {
        auto element_offset = i * VL;
        simd<float, VL> vec_a(a + element_offset); // Pointer arithmetic uses element offset
        simd<float, VL> vec_b(acc_b, element_offset * sizeof(float)); // accessor API uses byte-offset

        vec_a += vec_b;
        vec_a.copy_to(a + element_offset);
      });
#elif defined(USE_SYCL)
      cgh.parallel_for(Size, [=](id<1> i) {
        a[i] += acc_b[i];
      });
#endif
    }).wait_and_throw();

    for (unsigned i = 0; i < Size; ++i) {
      if (a[i] != (float)i + (float)i) {
        err_cnt++;
        std::cout << "failed at" << i << ": " << a[i] << " != " << (float)i
            << " + " << (float)i << std::endl;
      }
    }
  }
  catch (sycl::exception &e) {
    std::cout << "SYCL exception caught: " << e.what() << "\n";
    return 1;
  }
  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}
```
