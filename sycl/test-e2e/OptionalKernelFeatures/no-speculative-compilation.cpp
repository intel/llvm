// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// This test intends to check that no speculative compilation is happening,
// i.e. there are no exceptions thrown about aspects fp16 or fp64 being
// unsuppored on device.

#include <sycl/detail/core.hpp>

void foo(sycl::half &value) { value += sycl::half(1.0f); }

void foo(double &value) { value += 2.0; }

void foo(float &value) { value += 3.0f; }

int main() {
  sycl::queue q;

  sycl::half h = 0.0f;
  double d = 0.0;
  float f = 0.0f;

  sycl::buffer<sycl::half> buf_half(&h, sycl::range{1});
  sycl::buffer<double> buf_double(&d, sycl::range{1});
  sycl::buffer<float> buf_float(&f, sycl::range{1});

  if (q.get_device().has(sycl::aspect::fp16)) {
    q.submit([&](sycl::handler &cgh) {
       auto acc = buf_half.get_access(cgh);
       cgh.single_task([=] { foo(acc[0]); });
     }).wait();

    auto host_acc = buf_half.get_host_access();
    assert(host_acc[0] == 1.0f);
  } else if (q.get_device().has(sycl::aspect::fp64)) {
    q.submit([&](sycl::handler &cgh) {
       auto acc = buf_double.get_access(cgh);
       cgh.single_task([=] { foo(acc[0]); });
     }).wait();

    auto host_acc = buf_double.get_host_access();
    assert(host_acc[0] == 2.0f);
  } else {
    q.submit([&](sycl::handler &cgh) {
       auto acc = buf_float.get_access(cgh);
       cgh.single_task([=] { foo(acc[0]); });
     }).wait();

    auto host_acc = buf_float.get_host_access();
    assert(host_acc[0] == 3.0f);
  }

  return 0;
}
