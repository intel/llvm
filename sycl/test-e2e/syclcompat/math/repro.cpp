// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}

// RUN: %{build} %{mathflags} -o %t.out
// RUN: %{run} %t.out

#include "sycl/detail/builtins/builtins.hpp"
#include <sycl/usm.hpp>

#include <syclcompat/dims.hpp>
#include <syclcompat/math.hpp>

inline void fmax_nan_kernel(float *a, float *b, sycl::vec<float, 2> *r) {
  *r = syclcompat::fmax_nan(*a, *b);
}

void test_container_syclcompat_fmax_nan() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  sycl::queue q;

  sycl::range global{1};
  sycl::range local{1};
  sycl::nd_range ndr{global, local};

  const sycl::vec<float, 2> op1 = {5.0f, 10.0f};
  const sycl::vec<float, 2> op2 = {10.0f, 5.0f};
  const sycl::vec<float, 2> expected{static_cast<float>(10),
                                     static_cast<float>(10)};
  sycl::vec<float, 2> res;

  sycl::vec<float, 2> *op1_d = sycl::malloc_device<sycl::vec<float, 2>>(1, q);
  sycl::vec<float, 2> *op2_d = sycl::malloc_device<sycl::vec<float, 2>>(1, q);
  sycl::vec<float, 2> *res_d = sycl::malloc_device<sycl::vec<float, 2>>(1, q);

  q.memcpy(op1_d, &op1, sizeof(sycl::vec<float, 2>));
  q.memcpy(op2_d, &op2, sizeof(sycl::vec<float, 2>));
  q.submit([&](sycl::handler &cgh) {
     cgh.parallel_for(ndr, [=](sycl::nd_item<1> nd_item) {
       *res_d = syclcompat::fmax_nan(*op1_d, *op2_d);
     });
   }).wait_and_throw();
  q.memcpy(&res, res_d, sizeof(sycl::vec<float, 2>)).wait();

  constexpr float ERROR_TOLERANCE = 1e-6;
  for (size_t i = 0; i < 2; i++) {
    assert((res[i] - expected[i]) < ERROR_TOLERANCE ||
           !(std::cerr << "-- " << res[i] << " - " << expected[i] << " < "
                       << ERROR_TOLERANCE << " --"));
  }

  const sycl::vec<float, 2> op3 = {sycl::nan(static_cast<unsigned int>(0)),
                                   sycl::nan(static_cast<unsigned int>(0))};

  q.memcpy(op2_d, &op3, sizeof(sycl::vec<float, 2>));
  q.submit([&](sycl::handler &cgh) {
     cgh.parallel_for(ndr, [=](sycl::nd_item<1> nd_item) {
       *res_d = syclcompat::fmax_nan(*op1_d, *op2_d);
     });
   }).wait_and_throw();
  q.memcpy(&res, res_d, sizeof(sycl::vec<float, 2>)).wait();

  for (size_t i = 0; i < 2; i++) {
    assert(sycl::isnan(res[i]));
  }

  sycl::free(op1_d, q);
  sycl::free(op2_d, q);
  sycl::free(res_d, q);
}

int main() {
  test_container_syclcompat_fmax_nan();

  return 0;
}
