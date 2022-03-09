#include <CL/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

static constexpr const int VL = 4;

sycl::event add(size_t n, sycl::buffer<int, 1> &buf_a,
                sycl::buffer<int, 1> &buf_b, sycl::buffer<int, 1> &buf_c,
                sycl::queue &Q) {
  auto E = Q.submit([&](sycl::handler &H) {
    sycl::accessor acc_a{buf_a, H, sycl::read_only};
    sycl::accessor acc_b{buf_b, H, sycl::read_only};
    sycl::accessor acc_c{buf_c, H, sycl::write_only};

    H.parallel_for(n, [=](sycl::id<1> i) SYCL_ESIMD_KERNEL {
      using namespace sycl::ext::intel::esimd;
      unsigned int offset = i * VL * sizeof(int);
      simd<int, VL> va(acc_a, offset);
      simd<int, VL> vb(acc_b, offset);
      simd<int, VL> vc = va + vb;
      vc.copy_to(acc_c, offset);
    });
  });
  return E;
}
