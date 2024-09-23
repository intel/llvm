## Computing a binary function of three operands

Compile and run:
```bash
> clang++ -fsycl bnf.cpp

> ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./a.out
```

Illustrates using a special API to compute a binary function of three operands.
The function is computed for every bit in the integral input.

Source code:
```C++
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel;

using bfn_t = esimd::bfn_t;

// Let's compute f(x, y, z) = ~x | y ^ z.
constexpr esimd::bfn_t F = ~bfn_t::x | bfn_t::y ^ bfn_t::z;

constexpr unsigned Size = 128;
constexpr unsigned VL = 32;

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

void init(unsigned *in1, unsigned *in2, unsigned *in3, unsigned *res,
          size_t size) {
  for (int i = 0; i < size; i++) {
    in1[i] = i * 3;
    in2[i] = i * 3 + 1;
    in3[i] = i * 3 + 2;
    res[i] = 0;
  }
}

// Compute reference gold value.
unsigned ref(unsigned x, unsigned y, unsigned z) {
  unsigned res = 0;
  for (unsigned i = 0; i < sizeof(x) * 8; i++) {
    unsigned mask = unsigned(0x1) << i;
    res = (res & ~mask) |
          ((static_cast<uint8_t>(F) >> ((((x >> i) & unsigned(0x1))) +
                                        (((y >> i) & unsigned(0x1)) << 1) +
                                        (((z >> i) & unsigned(0x1)) << 2)) &
            unsigned(0x1))
           << i);
  }
  return res;
}

int main() {
  int err_cnt = 0;

  try {
    queue q(gpu_selector_v, create_exception_handler());
    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
              << "\n";

    unsigned *in1 = malloc_shared<unsigned>(Size, q);
    unsigned *in2 = malloc_shared<unsigned>(Size, q);
    unsigned *in3 = malloc_shared<unsigned>(Size, q);
    unsigned *res = malloc_shared<unsigned>(Size, q);

    std::unique_ptr<unsigned, usm_deleter> guard_in1(in1, usm_deleter{q});
    std::unique_ptr<unsigned, usm_deleter> guard_in2(in2, usm_deleter{q});
    std::unique_ptr<unsigned, usm_deleter> guard_in3(in3, usm_deleter{q});
    std::unique_ptr<unsigned, usm_deleter> guard_res(res, usm_deleter{q});

    init(in1, in2, in3, res, Size);

    q.parallel_for(Size / VL, [=](id<1> i) [[intel::sycl_explicit_simd]] {
       using namespace esimd;
       auto offset = i * VL;
       simd<unsigned, VL> X(in1 + offset);
       simd<unsigned, VL> Y(in2 + offset);
       simd<unsigned, VL> Z(in3 + offset);

       simd<unsigned, VL> val = bfn<F>(X, Y, Z);
       val.copy_to(res + offset);
     }).wait_and_throw();

    for (int i = 0; i < Size; i++) {
      unsigned gold = ref(in1[i], in2[i], in3[i]);
      unsigned test = res[i];
      if (gold != test)
        err_cnt++;
    }
  } catch (sycl::exception &e) {
    std::cout << "SYCL exception caught: " << e.what() << "\n";
    return 1;
  }

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}
```
