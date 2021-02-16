// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %RUN_ON_HOST %t.out

#include <CL/sycl.hpp>
#include <cassert>
#include <cmath>

bool isEqualTo(double x, double y, double epsilon = 0.001) {
  return std::fabs(x - y) <= epsilon;
}

int main(int argc, const char **argv) {
  cl::sycl::cl_double4 r{0};
  {
    cl::sycl::buffer<cl::sycl::cl_double4, 1> BufR(&r, cl::sycl::range<1>(1));
    cl::sycl::queue myQueue;
    myQueue.submit([&](cl::sycl::handler &cgh) {
      auto AccR = BufR.get_access<cl::sycl::access::mode::write>(cgh);
      cgh.single_task<class crossD4>([=]() {
        AccR[0] = cl::sycl::cross(
            cl::sycl::cl_double4{
                2.5,
                3.0,
                4.0,
                0.0,
            },
            cl::sycl::cl_double4{
                5.2,
                6.0,
                7.0,
                0.0,
            });
      });
    });
  }
  cl::sycl::cl_double r1 = r.x();
  cl::sycl::cl_double r2 = r.y();
  cl::sycl::cl_double r3 = r.z();
  cl::sycl::cl_double r4 = r.w();

  assert(isEqualTo(r1, -3.0));
  assert(isEqualTo(r2, 3.3));
  assert(isEqualTo(r3, -0.6));
  assert(isEqualTo(r4, 0.0));

  return 0;
}
