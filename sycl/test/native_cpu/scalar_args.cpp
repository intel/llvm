// REQUIRES: native_cpu
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu %s -o %t
// RUN: env ONEAPI_DEVICE_SELECTOR="native_cpu:cpu" %t

#include <sycl/sycl.hpp>

#include <iostream>

using namespace sycl;

const size_t N = 10;

template <typename T> class init_a;

template <typename T, int M>
bool compare(const sycl::vec<T, M> a, const sycl::vec<T, M> truth) {
  bool res = true;
  for (int i = 0; i < M; i++) {
    res &= a[i] == truth[i];
  }
  return res;
}

template <typename T> bool compare(const T a, const T truth) {
  return a == truth;
}

template <typename T> bool check(buffer<T, 1> &result, const T truth) {
  auto A = result.get_host_access();
  for (size_t i = 0; i < N; i++) {
    if (!compare(A[i], truth)) {
      return false;
    }
  }
  return true;
}

template <typename T> bool test(queue myQueue) {
  buffer<T, 1> a(range<1>{N});
  const T test{42};

  myQueue.submit([&](handler &cgh) {
    auto A = a.template get_access<access::mode::write>(cgh);
    cgh.parallel_for<init_a<T>>(range<1>{N},
                                [=](id<1> index) { A[index] = test; });
  });

  return check(a, test);
}

int main() {
  queue q;

  std::vector<bool> res;
  res.push_back(test<int>(q));
  res.push_back(test<unsigned>(q));
  res.push_back(test<float>(q));
  res.push_back(test<double>(q));
  res.push_back(test<sycl::vec<int, 2>>(q));
  res.push_back(test<sycl::vec<int, 3>>(q));
  res.push_back(test<sycl::vec<int, 4>>(q));
  res.push_back(test<sycl::vec<int, 8>>(q));
  res.push_back(test<sycl::vec<int, 16>>(q));
  res.push_back(test<sycl::vec<double, 2>>(q));
  res.push_back(test<sycl::vec<double, 3>>(q));
  res.push_back(test<sycl::vec<double, 4>>(q));
  res.push_back(test<sycl::vec<double, 8>>(q));
  res.push_back(test<sycl::vec<double, 16>>(q));

  if (std::any_of(res.begin(), res.end(), [](bool b) { return !b; })) {
    return 1;
  }
  return 0;
}
