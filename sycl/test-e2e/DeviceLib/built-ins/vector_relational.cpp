// FIXME unsupported on windows (opencl and level-zero) until fix of libdevice
// UNSUPPORTED: windows && (opencl || level_zero)
// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}
// RUN: %{build} -o %t.out %{mathflags}
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <sycl/builtins.hpp>

#include <cassert>
#include <cmath>
#include <iostream>

namespace s = sycl;

int main() {
  // isequal
  {
    s::int4 r{0};
    {
      s::buffer<s::int4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isequalF4F4>([=]() {
          AccR[0] = s::isequal(s::float4{0.5f, 0.6f, NAN, INFINITY},
                               s::float4{0.5f, 0.5f, 0.5f, 0.5f});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    int r3 = r.z();
    int r4 = r.w();

    assert(r1 == -1);
    assert(r2 == 0);
    assert(r3 == 0);
    assert(r4 == 0);
  }

  // isnotequal
  {
    s::int4 r{0};
    {
      s::buffer<s::int4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isnotequalF4F4>([=]() {
          AccR[0] = s::isnotequal(s::float4{0.5f, 0.6f, NAN, INFINITY},
                                  s::float4{0.5f, 0.5f, 0.5f, 0.5f});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    int r3 = r.z();
    int r4 = r.w();

    assert(r1 == 0);
    assert(r2 == -1);
    assert(r3 == -1);
    assert(r4 == -1);
  }

  // isgreater
  {
    s::int4 r{0};
    {
      s::buffer<s::int4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isgreaterF4F4>([=]() {
          AccR[0] = s::isgreater(s::float4{0.5f, 0.6f, NAN, INFINITY},
                                 s::float4{0.5f, 0.5f, 0.5f, 0.5f});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    int r3 = r.z();
    int r4 = r.w();

    assert(r1 == 0);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == -1);
  }

  // isgreaterequal
  {
    s::int4 r{0};
    {
      s::buffer<s::int4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isgreaterequalF4F4>([=]() {
          AccR[0] = s::isgreaterequal(s::float4{0.5f, 0.6f, NAN, INFINITY},
                                      s::float4{0.5f, 0.5f, 0.5f, 0.5f});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    int r3 = r.z();
    int r4 = r.w();

    assert(r1 == -1);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == -1);
  }

  // isless
  {
    s::int4 r{0};
    {
      s::buffer<s::int4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class islessF4F4>([=]() {
          AccR[0] = s::isless(s::float4{0.5f, 0.4f, NAN, INFINITY},
                              s::float4{0.5f, 0.5f, 0.5f, 0.5f});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    int r3 = r.z();
    int r4 = r.w();

    assert(r1 == 0);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == 0);
  }

  // islessequal
  {
    s::int4 r{0};
    {
      s::buffer<s::int4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class islessequalF4F4>([=]() {
          AccR[0] = s::islessequal(s::float4{0.5f, 0.4f, NAN, INFINITY},
                                   s::float4{0.5f, 0.5f, 0.5f, 0.5f});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    int r3 = r.z();
    int r4 = r.w();

    assert(r1 == -1);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == 0);
  }

  // islessgreater
  {
    s::int4 r{0};
    {
      s::buffer<s::int4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class islessgreaterF4F4>([=]() {
          AccR[0] = s::islessgreater(s::float4{0.5f, 0.4f, NAN, INFINITY},
                                     s::float4{0.5f, 0.5f, 0.5f, INFINITY});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    int r3 = r.z();
    int r4 = r.w();

    assert(r1 == 0);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == 0); // Infinity is considered as greater than any
                     // other value except Infinity.
  }

  // isfinite
  {
    s::int4 r{0};
    {
      s::buffer<s::int4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isfiniteF4F4>([=]() {
          AccR[0] = s::isfinite(s::float4{0.5f, 0.4f, NAN, INFINITY});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    int r3 = r.z();
    int r4 = r.w();

    assert(r1 == -1);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == 0);
  }

  // isinf
  {
    s::int4 r{0};
    {
      s::buffer<s::int4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isinfF4F4>([=]() {
          AccR[0] = s::isinf(s::float4{0.5f, 0.4f, NAN, INFINITY});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    int r3 = r.z();
    int r4 = r.w();

    assert(r1 == 0);
    assert(r2 == 0);
    assert(r3 == 0);
    assert(r4 == -1);
  }

  // isnan
  {
    s::int4 r{0};
    {
      s::buffer<s::int4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isnanF4F4>([=]() {
          AccR[0] = s::isnan(s::float4{0.5f, 0.4f, NAN, INFINITY});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    int r3 = r.z();
    int r4 = r.w();

    assert(r1 == 0);
    assert(r2 == 0);
    assert(r3 == -1);
    assert(r4 == 0);
  }

  // isnormal
  {
    s::int4 r{0};
    {
      s::buffer<s::int4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isnormalF4F4>([=]() {
          AccR[0] = s::isnormal(s::float4{0.5f, 0.4f, NAN, INFINITY});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    int r3 = r.z();
    int r4 = r.w();

    assert(r1 == -1);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == 0);
  }

  // isordered
  {
    s::int4 r{0};
    {
      s::buffer<s::int4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isorderedF4F4>([=]() {
          AccR[0] = s::isordered(s::float4{0.5f, 0.6f, NAN, INFINITY},
                                 s::float4{0.5f, 0.5f, 0.5f, 0.5f});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    int r3 = r.z();
    int r4 = r.w();

    assert(r1 == -1);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == -1); // infinity is ordered.
  }

  // isunordered
  {
    s::int4 r{0};
    {
      s::buffer<s::int4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isunorderedF4F4>([=]() {
          AccR[0] = s::isunordered(s::float4{0.5f, 0.6f, NAN, INFINITY},
                                   s::float4{0.5f, 0.5f, 0.5f, 0.5f});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    int r3 = r.z();
    int r4 = r.w();

    assert(r1 == 0);
    assert(r2 == 0);
    assert(r3 == -1);
    assert(r4 == 0);
  }

  // signbit
  {
    s::int3 r{0};
    {
      s::buffer<s::int3, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class signbitF3>([=]() {
          AccR[0] = s::signbit(s::float3{0.5f, -12.0f, INFINITY});
        });
      });
    }
    int r1 = r.x();
    int r2 = r.y();
    int r3 = r.z();

    assert(r1 == 0);
    assert(r2 == -1);
    assert(r3 == 0);
  }

  // any.
  // Call to the device function with vector parameters work. Scalars do not.
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class anyI4>([=]() {
          AccR[0] = s::any(s::int4{-12, -12, 0, 1});
        });
      });
    }
    int r1 = r;

    assert(r1 == 1);
  }

  // any.
  // Call to the device function with vector parameters work. Scalars do not.
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class anyI4negative>([=]() {
          AccR[0] = s::any(s::int4{-12, -12, -12, -12});
        });
      });
    }
    int r1 = r;

    assert(r1 == 1);
  }

  // any.
  // Call to the device function with vector parameters work. Scalars do not.
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class anyI4zero>([=]() {
          AccR[0] = s::any(s::int4{0, 0, 0, 0});
        });
      });
    }
    int r1 = r;

    assert(r1 == 0);
  }

  // any.
  // Call to the device function with vector parameters work. Scalars do not.
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class anyI4positive>([=]() {
          AccR[0] = s::any(s::int4{12, 12, 12, 12});
        });
      });
    }
    int r1 = r;

    assert(r1 == 0);
  }

  // all.
  // Call to the device function with vector parameters work. Scalars do not.
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class allI4>([=]() {
          AccR[0] = s::all(s::int4{-12, -12, -12, -12});
          // Infinity (positive or negative) or Nan are not integers.
          // Passing them creates inconsistent results between host and device
          // execution.
        });
      });
    }
    int r1 = r;

    assert(r1 == 1);
  }

  // all.
  // Call to the device function with vector parameters work. Scalars do not.
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class allI4negative>([=]() {
          AccR[0] = s::all(s::int4{-12, -12, -12, -12});
        });
      });
    }
    int r1 = r;

    assert(r1 == 1);
  }

  // all.
  // Call to the device function with vector parameters work. Scalars do not.
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class allI4zero>([=]() {
          AccR[0] = s::all(s::int4{0, 0, 0, 0});
        });
      });
    }
    int r1 = r;

    assert(r1 == 0);
  }

  // all.
  // Call to the device function with vector parameters work. Scalars do not.
  {
    int r{0};
    {
      s::buffer<int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class allI4positive>([=]() {
          AccR[0] = s::all(s::int4{12, 12, 12, 12});
        });
      });
    }
    int r1 = r;

    assert(r1 == 0);
  }

  // bitselect
  {
    s::float4 r{0};
    {
      s::buffer<s::float4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class bitselectF4F4F4>([=]() {
          AccR[0] = s::bitselect(s::float4{112.112f, 12.12f, 0, 0.0f},
                                 s::float4{34.34f, 23.23f, 1, 0.0f},
                                 s::float4{3.3f, 6.6f, 1, 0.0f});
        }); // Using NAN/INFINITY as any float produced consistent results
            // between host and device.
      });
    }
    float r1 = r.x();
    float r2 = r.y();
    float r3 = r.z();
    float r4 = r.w();

    assert(abs(r1 - 80.5477f) < 0.0001);
    assert(abs(r2 - 18.2322f) < 0.0001);
    assert(abs(r3 - 1.0f) < 0.01);
    assert(abs(r4 - 0.0f) < 0.01);
  }

  // select
  {
    s::float4 r{0};
    {
      s::buffer<s::float4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class selectF4F4I4>([=]() {
          AccR[0] = s::select(s::float4{112.112f, 34.34f, 112.112f, 34.34f},
                              s::float4{34.34f, 112.112f, 34.34f, 112.112f},
                              s::int4{0, -1, 0, 1});
          // Using NAN/infinity as an input, which gets
          // selected by -1, produces a NAN/infinity as expected.
        });
      });
    }
    float r1 = r.x();
    float r2 = r.y();
    float r3 = r.z();
    float r4 = r.w();

    assert(r1 == 112.112f);
    assert(r2 == 112.112f);
    assert(r3 == 112.112f);
    assert(r4 == 34.34f);
  }

  {
    s::vec<int, 4> r(0);
    {
      s::vec<int, 4> a(1, 2, 3, 4);
      s::vec<int, 4> b(5, 6, 7, 8);
      s::vec<unsigned int, 4> m(1u, 0x80000000u, 42u, 0x80001000u);
      s::buffer<s::vec<int, 4>> A(&a, s::range<1>(1));
      s::buffer<s::vec<int, 4>> B(&b, s::range<1>(1));
      s::buffer<s::vec<unsigned int, 4>> M(&m, s::range<1>(1));
      s::buffer<s::vec<int, 4>> R(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccA = A.get_access<s::access::mode::read>(cgh);
        auto AccB = B.get_access<s::access::mode::read>(cgh);
        auto AccM = M.get_access<s::access::mode::read>(cgh);
        auto AccR = R.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class selectI4I4U4>(
            [=]() { AccR[0] = s::select(AccA[0], AccB[0], AccM[0]); });
      });
    }
    if (r.x() != 1 || r.y() != 6 || r.z() != 3 || r.w() != 8) {
      std::cerr << "selectI4I4U4 test case failed!\n";
      std::cerr << "Expected result: 1 6 3 8\n";
      std::cerr << "Got: " << r.x() << " " << r.y() << " " << r.z() << " "
                << r.w() << "\n";
      return 1;
    }
  }

  return 0;
}
