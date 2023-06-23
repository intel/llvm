## Call ESIMD from SYCL using invoke_simd

In this example, we will scale the input data by a factor of 2 using `invoke_simd`.

Compile and run:
```bash
> clang++ -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr invoke_simd.cpp

> IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./a.out
Running on Intel(R) UHD Graphics 630
Passed
```
Source code:
```c++
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/sycl.hpp>
#include <iostream>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
namespace esimd = sycl::ext::intel::esimd;

constexpr int Size = 512;
constexpr int VL = 16;

[[intel::device_indirectly_callable]] simd<int, VL> __regcall scale(
    simd<int, VL> x, int n) SYCL_ESIMD_FUNCTION {
  esimd::simd<int, VL> vec = x;
  esimd::simd<int, VL> result = vec * n;
  return result;
}

int main(void) {
  auto q = queue{gpu_selector_v};
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";
  bool passed = true;
  int *In = new int[Size];
  int *Out = new int[Size];

  for (int i = 0; i < Size; ++i) {
    In[i] = i;
    Out[i] = 0;
  }

  // scale factor
  int n = 2;

  try {
    buffer<int, 1> bufin(In, range<1>(Size));
    buffer<int, 1> bufout(Out, range<1>(Size));

    sycl::range<1> GlobalRange{Size};
    sycl::range<1> LocalRange{VL};

    auto e = q.submit([&](handler &cgh) {
      auto accin = bufin.get_access<access::mode::read>(cgh);
      auto accout = bufout.get_access<access::mode::write>(cgh);

      cgh.parallel_for<class Scale>(
          nd_range<1>(GlobalRange, LocalRange), [=](nd_item<1> item) {
            sycl::sub_group sg = item.get_sub_group();
            unsigned int offset = item.get_global_linear_id();

            int in = sg.load(accin.get_pointer() + offset);

            int out = invoke_simd(sg, scale, in, uniform{n});

            sg.store(accout.get_pointer() + offset, out);
          });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    delete[] In;
    delete[] Out;
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return 1;
  }

  for (int i = 0; i < Size; ++i) {
    if (Out[i] != In[i] * n) {
      std::cout << "failed at index " << i << ", " << Out[i] << " != " << In[i]
                << " * " << n << "\n";
      passed = false;
    }
  }
  delete[] In;
  delete[] Out;
  std::cout << (passed ? "Passed\n" : "FAILED\n");
  return passed ? 0 : 1;
}

```
