// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
#include <cassert>
#include <iostream>

#include <sycl/detail/core.hpp>

#include <sycl/detail/boolean.hpp>

using namespace sycl;
namespace s = sycl;
namespace d = s::detail;

d::Boolean<3> foo() {
  d::Boolean<3> b3{true, false, true};
  return b3;
}

int main() {
  {
    s::long4 r{0};
    {
      buffer<s::long4, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class b4_l4>([=]() {
          d::Boolean<4> b4{false, true, false, false};
          AccR[0] = b4;
        });
      });
    }
    long long r1 = r.s0();
    long long r2 = r.s1();
    long long r3 = r.s2();
    long long r4 = r.s3();

    std::cout << "r1 " << r1 << " r2 " << r2 << " r3 " << r3 << " r4 " << r4
              << std::endl;

    assert(r1 == 0);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == 0);
  }

  {
    s::short3 r{0};
    {
      buffer<s::short3, 1> BufR(&r, range<1>(1));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR = BufR.get_access<access::mode::write>(cgh);
        cgh.single_task<class b3_sh3>([=]() { AccR[0] = foo(); });
      });
    }
    short r1 = r.s0();
    short r2 = r.s1();
    short r3 = r.s2();

    std::cout << "r1 " << r1 << " r2 " << r2 << " r3 " << r3 << std::endl;

    assert(r1 == -1);
    assert(r2 == 0);
    assert(r3 == -1);
  }

  {
    int r1[5];
    int r2[5];
    {
      buffer<int, 1> BufR1(r1, range<1>(6));
      buffer<int, 1> BufR2(r2, range<1>(6));
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto AccR1 = BufR1.get_access<access::mode::write>(cgh);
        auto AccR2 = BufR2.get_access<access::mode::write>(cgh);
        cgh.single_task<class size_align>([=]() {
          AccR1[0] = sizeof(d::Boolean<2>);
          AccR1[1] = sizeof(d::Boolean<3>);
          AccR1[2] = sizeof(d::Boolean<4>);
          AccR1[3] = sizeof(d::Boolean<8>);
          AccR1[4] = sizeof(d::Boolean<16>);

          AccR2[0] = alignof(d::Boolean<2>);
          AccR2[1] = alignof(d::Boolean<3>);
          AccR2[2] = alignof(d::Boolean<4>);
          AccR2[3] = alignof(d::Boolean<8>);
          AccR2[4] = alignof(d::Boolean<16>);
        });
      });
    }

    for (size_t I = 0; I < 5; I++) {
      std::cout << " r1[" << I << "] " << r1[I];
    }
    std::cout << std::endl;

    for (size_t I = 0; I < 5; I++) {
      std::cout << " r2[" << I << "] " << r2[I];
    }
    std::cout << std::endl;
    assert(r1[0] == sizeof(d::Boolean<2>));
    assert(r1[1] == sizeof(d::Boolean<3>));
    assert(r1[2] == sizeof(d::Boolean<4>));
    assert(r1[3] == sizeof(d::Boolean<8>));
    assert(r1[4] == sizeof(d::Boolean<16>));

    assert(r2[0] == alignof(d::Boolean<2>));
    assert(r2[1] == alignof(d::Boolean<3>));
    assert(r2[2] == alignof(d::Boolean<4>));
    assert(r2[3] == alignof(d::Boolean<8>));
    assert(r2[4] == alignof(d::Boolean<16>));
  }

  {
    s::int4 i4 = {1, -2, 0, -3};
    d::Boolean<4> b4(i4);
    i4 = b4;

    int r1 = i4.s0();
    int r2 = i4.s1();
    int r3 = i4.s2();
    int r4 = i4.s3();

    std::cout << "r1 " << r1 << " r2 " << r2 << " r3 " << r3 << " r4 " << r4
              << std::endl;
    assert(r1 == 0);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == -1);
  }

  return 0;
}
