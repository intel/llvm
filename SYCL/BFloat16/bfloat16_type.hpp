#include <iostream>
#include <sycl/ext/oneapi/experimental/bfloat16.hpp>
#include <sycl/sycl.hpp>

#include <cmath>

using namespace sycl;

constexpr size_t N = 100;

template <typename T> void assert_close(const T &C, const float ref) {
  for (size_t i = 0; i < N; i++) {
    auto diff = C[i] - ref;
    assert(std::fabs(static_cast<float>(diff)) <
           std::numeric_limits<float>::epsilon());
  }
}

void verify_conv_implicit(queue &q, buffer<float, 1> &a, range<1> &r,
                          const float ref) {
  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read_write>(cgh);
    cgh.parallel_for<class calc_conv>(r, [=](id<1> index) {
      sycl::ext::oneapi::experimental::bfloat16 AVal{A[index]};
      A[index] = AVal;
    });
  });

  assert_close(a.get_access<access::mode::read>(), ref);
}

void verify_conv_explicit(queue &q, buffer<float, 1> &a, range<1> &r,
                          const float ref) {
  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read_write>(cgh);
    cgh.parallel_for<class calc_conv_impl>(r, [=](id<1> index) {
      uint16_t AVal =
          sycl::ext::oneapi::experimental::bfloat16::from_float(A[index]);
      A[index] = sycl::ext::oneapi::experimental::bfloat16::to_float(AVal);
    });
  });

  assert_close(a.get_access<access::mode::read>(), ref);
}

void verify_add(queue &q, buffer<float, 1> &a, buffer<float, 1> &b, range<1> &r,
                const float ref) {
  buffer<float, 1> c{r};

  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read>(cgh);
    auto B = b.get_access<access::mode::read>(cgh);
    auto C = c.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class calc_add_expl>(r, [=](id<1> index) {
      sycl::ext::oneapi::experimental::bfloat16 AVal{A[index]};
      sycl::ext::oneapi::experimental::bfloat16 BVal{B[index]};
      sycl::ext::oneapi::experimental::bfloat16 CVal = AVal + BVal;
      C[index] = CVal;
    });
  });

  assert_close(c.get_access<access::mode::read>(), ref);
}

void verify_sub(queue &q, buffer<float, 1> &a, buffer<float, 1> &b, range<1> &r,
                const float ref) {
  buffer<float, 1> c{r};

  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read>(cgh);
    auto B = b.get_access<access::mode::read>(cgh);
    auto C = c.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class calc_sub>(r, [=](id<1> index) {
      sycl::ext::oneapi::experimental::bfloat16 AVal{A[index]};
      sycl::ext::oneapi::experimental::bfloat16 BVal{B[index]};
      sycl::ext::oneapi::experimental::bfloat16 CVal = AVal - BVal;
      C[index] = CVal;
    });
  });

  assert_close(c.get_access<access::mode::read>(), ref);
}

void verify_minus(queue &q, buffer<float, 1> &a, range<1> &r, const float ref) {
  buffer<float, 1> c{r};

  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read>(cgh);
    auto C = c.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class calc_minus>(r, [=](id<1> index) {
      sycl::ext::oneapi::experimental::bfloat16 AVal{A[index]};
      sycl::ext::oneapi::experimental::bfloat16 CVal = -AVal;
      C[index] = CVal;
    });
  });

  assert_close(c.get_access<access::mode::read>(), ref);
}

void verify_mul(queue &q, buffer<float, 1> &a, buffer<float, 1> &b, range<1> &r,
                const float ref) {
  buffer<float, 1> c{r};

  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read>(cgh);
    auto B = b.get_access<access::mode::read>(cgh);
    auto C = c.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class calc_mul>(r, [=](id<1> index) {
      sycl::ext::oneapi::experimental::bfloat16 AVal{A[index]};
      sycl::ext::oneapi::experimental::bfloat16 BVal{B[index]};
      sycl::ext::oneapi::experimental::bfloat16 CVal = AVal * BVal;
      C[index] = CVal;
    });
  });

  assert_close(c.get_access<access::mode::read>(), ref);
}

void verify_div(queue &q, buffer<float, 1> &a, buffer<float, 1> &b, range<1> &r,
                const float ref) {
  buffer<float, 1> c{r};

  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read>(cgh);
    auto B = b.get_access<access::mode::read>(cgh);
    auto C = c.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class calc_div>(r, [=](id<1> index) {
      sycl::ext::oneapi::experimental::bfloat16 AVal{A[index]};
      sycl::ext::oneapi::experimental::bfloat16 BVal{B[index]};
      sycl::ext::oneapi::experimental::bfloat16 CVal = AVal / BVal;
      C[index] = CVal;
    });
  });

  assert_close(c.get_access<access::mode::read>(), ref);
}

void verify_logic(queue &q, buffer<float, 1> &a, buffer<float, 1> &b,
                  range<1> &r, const float ref) {
  buffer<float, 1> c{r};

  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read>(cgh);
    auto B = b.get_access<access::mode::read>(cgh);
    auto C = c.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class logic>(r, [=](id<1> index) {
      sycl::ext::oneapi::experimental::bfloat16 AVal{A[index]};
      sycl::ext::oneapi::experimental::bfloat16 BVal{B[index]};
      if (AVal) {
        if (AVal > BVal || AVal >= BVal || AVal < BVal || AVal <= BVal ||
            !BVal) {
          sycl::ext::oneapi::experimental::bfloat16 CVal =
              AVal != BVal ? AVal : BVal;
          CVal--;
          CVal++;
          if (AVal == BVal) {
            CVal -= AVal;
            CVal *= 3.0;
            CVal /= 2.0;
          } else
            CVal += BVal;
          C[index] = CVal;
        }
      }
    });
  });

  assert_close(c.get_access<access::mode::read>(), ref);
}

int run_tests() {
  device dev{default_selector_v};

  // TODO: replace is_gpu check with extension check when the appropriate part
  // of implementation ready (aspect)
  if (!dev.is_gpu() && !dev.is_cpu()) {
    std::cout << "This device doesn't support bfloat16 conversion feature"
              << std::endl;
    return 0;
  }

  std::vector<float> vec_a(N, 5.0);
  std::vector<float> vec_b(N, 2.0);
  std::vector<float> vec_b_neg(N, -2.0);

  range<1> r(N);
  buffer<float, 1> a{vec_a.data(), r};
  buffer<float, 1> b{vec_b.data(), r};
  buffer<float, 1> b_neg{vec_b_neg.data(), r};

  queue q{dev};

  verify_conv_implicit(q, a, r, 5.0);
  verify_conv_explicit(q, a, r, 5.0);
  verify_add(q, a, b, r, 7.0);
  verify_sub(q, a, b, r, 3.0);
  verify_mul(q, a, b, r, 10.0);
  verify_div(q, a, b, r, 2.5);
  verify_logic(q, a, b, r, 7.0);
  verify_add(q, a, b_neg, r, 3.0);
  verify_sub(q, a, b_neg, r, 7.0);
  verify_minus(q, a, r, -5.0);
  verify_mul(q, a, b_neg, r, -10.0);
  verify_div(q, a, b_neg, r, -2.5);
  verify_logic(q, a, b_neg, r, 3.0);

  return 0;
}
