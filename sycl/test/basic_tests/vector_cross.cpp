// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

#include <cassert>
#include <cmath>
#include <sycl/sycl.hpp>

bool isEqualTo(double x, double y, double epsilon = 0.001) {
  return std::fabs(x - y) <= epsilon;
}

int main(int argc, const char **argv) {
  sycl::vec<sycl::opencl::cl_double, 4> r = sycl::cross(
      sycl::vec<sycl::opencl::cl_double, 4>{
          2.5,
          3.0,
          4.0,
          0.0,
      },
      sycl::vec<sycl::opencl::cl_double, 4>{
          5.2,
          6.0,
          7.0,
          0.0,
      });

  sycl::opencl::cl_double r1 = r.x();
  sycl::opencl::cl_double r2 = r.y();
  sycl::opencl::cl_double r3 = r.z();
  sycl::opencl::cl_double r4 = r.w();

  assert(isEqualTo(r1, -3.0));
  assert(isEqualTo(r2, 3.3));
  assert(isEqualTo(r3, -0.6));
  assert(isEqualTo(r4, 0.0));

  return 0;
}
