// REQUIRES: native_cpu_be
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu %s -o %t -g
// RUN: env ONEAPI_DEVICE_SELECTOR="native_cpu:cpu" %t

#include "sycl/builtins_marray_gen.hpp"
#include "sycl/builtins_vector_gen.hpp"
#include <sycl/sycl.hpp>

#include <array>
#include <iostream>

using namespace sycl;
constexpr sycl::access::mode sycl_read_write = sycl::access::mode::read_write;

template <typename T>
class Test;

template <typename T>
class TestInt;

static constexpr int NumMathBuiltins = 16; 
static constexpr float eps = 0.01;

template <typename T>
using ResultT = std::array<T, NumMathBuiltins>;

template <typename T>
ResultT<T> do_test(T in) {
  ResultT<T> res;
  unsigned i = 0;
  res[i++] = sycl::native::sqrt(in);
  res[i++] = sycl::sqrt(in);
  res[i++] = sycl::fabs(in);
  res[i++] = sycl::fma(in,in,in);
  res[i++] = sycl::trunc(in);
  res[i++] = sycl::rint(in);
  res[i++] = sycl::round(in);
  res[i++] = sycl::ceil(in);
  res[i++] = sycl::floor(in);
  res[i++] = sycl::native::cos(in);
  res[i++] = sycl::native::sin(in);
  res[i++] = sycl::native::exp2(in);
  res[i++] = sycl::native::exp(in);
  res[i++] = sycl::native::log10(in);
  res[i++] = sycl::native::log(in);
  res[i++] = sycl::native::log2(in);
  return res;
}

template <typename T>
bool check(T& res, T& exp) {
  bool correct = std::abs(static_cast<float>(res) - static_cast<float>(exp)) < eps;
  if(!correct) {
    std::cout << "Value mismatch; Expected: " << exp << " actual: " << res << "\n";
    return false;
  }
  return true;
}

template <typename T, int N>
bool check(sycl::vec<T,N>& res, sycl::vec<T, N>& exp) {
  bool correct = true;
  for(int i = 0; i < N; i++) {
    correct &= check(res[i], exp[i]);
  }
  return correct;
}


template<typename T>
bool test(queue deviceQueue) {
  const size_t N = 1;
  const T Init{1};
  std::array<T, N> A = {Init};
  std::array<ResultT<T>, 1> Res;
  sycl::range<1> numOfItems{N};
  {
    sycl::buffer<T, 1> bufferA(A.data(), numOfItems);
    sycl::buffer<ResultT<T>, 1> bufferRes(Res.data(), numOfItems);

    deviceQueue
        .submit([&](sycl::handler &cgh) {
          auto accessorA = bufferA.template get_access<sycl_read_write>(cgh);
          auto accessorRes = bufferRes.template get_access<sycl_read_write>(cgh);

          auto kern = [=]() {
            accessorRes[0] = do_test(accessorA[0]);
          };
          cgh.single_task<Test<T>>(kern);
        })
        .wait();
  }
  ResultT<T> expected = do_test(Init);
  for(int i = 0; i < NumMathBuiltins; i++) {
     if(!check(Res[0][i], expected[i])) {
       return false;
     }
  }
 return true; 
}

template <typename T>
bool test_int(queue deviceQueue) {
  const size_t N = 1;
  const T Init{10};
  std::array<T, N> A = {Init};
  sycl::range<1> numOfItems{N};
  {
    sycl::buffer<T, 1> bufferA(A.data(), numOfItems);

    deviceQueue
        .submit([&](sycl::handler &cgh) {
          auto accessorA = bufferA.template get_access<sycl_read_write>(cgh);

          auto kern = [=]() {
            accessorA[0] = sycl::popcount<T>(accessorA[0]);
          };
          cgh.single_task<TestInt<T>>(kern);
        })
        .wait();
  }
  T expected = sycl::popcount(Init);
  if(!(A[0] == expected)) {
    return false;
  }
 return true; 
}

int main() {
  queue q;
  bool success = true;
  success &= test<float>(q);
  success &= test<double>(q);
  success &= test_int<int>(q);
  success &= test_int<char>(q);

  if(!success) {
    std::cout << "Test failed\n";
    return 1;
  }
  std::cout << "Test passed\n";
  return 0;
}
