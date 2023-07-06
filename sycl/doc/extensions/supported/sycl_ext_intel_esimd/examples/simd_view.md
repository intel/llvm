### Using simd_view to construct views of simd objects

In this example we demonstrate using `simd_view` to construct multi-level
view into `simd` objects.

Compile and run:
```bash
> clang++ -fsycl simd_view.cpp

> ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./a.out
```

Source code:
```C++

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

inline auto create_exception_handler() {
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
    queue q(gpu_selector_v, create_exception_handler());
    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

    float *a = malloc_shared<float>(Size, q); // USM memory for A.
    float *b = new float[Size];

    std::unique_ptr<float, usm_deleter> guard_a(a, usm_deleter{q});
    std::unique_ptr<float> guard_b(b);

    // Initialize a and b.
    for (unsigned i = 0; i < Size; i++)
      a[i] = b[i] = i * i;

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

    // Verify on host.
    for (unsigned i = 0; i < Size; ++i) {
      float gold = b[i];
      if (i % 2 == 0)
        gold *= 3;
      if (i % 4 == 0)
        gold *= 2;
      if (a[i] != gold) {
        err_cnt++;
        std::cout << "failed at" << i << ": " << a[i] << " != " << (float)i
                  << " + " << (float)i << std::endl;
      }
    }
  } catch (sycl::exception &e) {
    std::cout << "SYCL exception caught: " << e.what() << "\n";
    return 1;
  }
  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}
```
