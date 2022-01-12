// FIXME unsupported on windows (opencl and level-zero) until fix of libdevice
// UNSUPPORTED: windows && (opencl || level_zero)
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

#include <cassert>
#include <cmath>
#include <iostream>

namespace s = cl::sycl;

int main() {
  // isequal
  {
    s::cl_int4 r{0};
    {
      s::buffer<s::cl_int4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isequalF4F4>([=]() {
          AccR[0] = s::isequal(s::cl_float4{0.5f, 0.6f, NAN, INFINITY},
                               s::cl_float4{0.5f, 0.5f, 0.5f, 0.5f});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    s::cl_int r3 = r.z();
    s::cl_int r4 = r.w();

    assert(r1 == -1);
    assert(r2 == 0);
    assert(r3 == 0);
    assert(r4 == 0);
  }

  // isnotequal
  {
    s::cl_int4 r{0};
    {
      s::buffer<s::cl_int4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isnotequalF4F4>([=]() {
          AccR[0] = s::isnotequal(s::cl_float4{0.5f, 0.6f, NAN, INFINITY},
                                  s::cl_float4{0.5f, 0.5f, 0.5f, 0.5f});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    s::cl_int r3 = r.z();
    s::cl_int r4 = r.w();

    assert(r1 == 0);
    assert(r2 == -1);
    assert(r3 == -1);
    assert(r4 == -1);
  }

  // isgreater
  {
    s::cl_int4 r{0};
    {
      s::buffer<s::cl_int4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isgreaterF4F4>([=]() {
          AccR[0] = s::isgreater(s::cl_float4{0.5f, 0.6f, NAN, INFINITY},
                                 s::cl_float4{0.5f, 0.5f, 0.5f, 0.5f});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    s::cl_int r3 = r.z();
    s::cl_int r4 = r.w();

    assert(r1 == 0);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == -1);
  }

  // isgreaterequal
  {
    s::cl_int4 r{0};
    {
      s::buffer<s::cl_int4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isgreaterequalF4F4>([=]() {
          AccR[0] = s::isgreaterequal(s::cl_float4{0.5f, 0.6f, NAN, INFINITY},
                                      s::cl_float4{0.5f, 0.5f, 0.5f, 0.5f});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    s::cl_int r3 = r.z();
    s::cl_int r4 = r.w();

    assert(r1 == -1);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == -1);
  }

  // isless
  {
    s::cl_int4 r{0};
    {
      s::buffer<s::cl_int4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class islessF4F4>([=]() {
          AccR[0] = s::isless(s::cl_float4{0.5f, 0.4f, NAN, INFINITY},
                              s::cl_float4{0.5f, 0.5f, 0.5f, 0.5f});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    s::cl_int r3 = r.z();
    s::cl_int r4 = r.w();

    assert(r1 == 0);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == 0);
  }

  // islessequal
  {
    s::cl_int4 r{0};
    {
      s::buffer<s::cl_int4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class islessequalF4F4>([=]() {
          AccR[0] = s::islessequal(s::cl_float4{0.5f, 0.4f, NAN, INFINITY},
                                   s::cl_float4{0.5f, 0.5f, 0.5f, 0.5f});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    s::cl_int r3 = r.z();
    s::cl_int r4 = r.w();

    assert(r1 == -1);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == 0);
  }

  // islessgreater
  {
    s::cl_int4 r{0};
    {
      s::buffer<s::cl_int4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class islessgreaterF4F4>([=]() {
          AccR[0] = s::islessgreater(s::cl_float4{0.5f, 0.4f, NAN, INFINITY},
                                     s::cl_float4{0.5f, 0.5f, 0.5f, INFINITY});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    s::cl_int r3 = r.z();
    s::cl_int r4 = r.w();

    assert(r1 == 0);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == 0); // Infinity is considered as greater than any
                     // other value except Infinity.
  }

  // isfinite
  {
    s::cl_int4 r{0};
    {
      s::buffer<s::cl_int4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isfiniteF4F4>([=]() {
          AccR[0] = s::isfinite(s::cl_float4{0.5f, 0.4f, NAN, INFINITY});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    s::cl_int r3 = r.z();
    s::cl_int r4 = r.w();

    assert(r1 == -1);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == 0);
  }

  // isinf
  {
    s::cl_int4 r{0};
    {
      s::buffer<s::cl_int4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isinfF4F4>([=]() {
          AccR[0] = s::isinf(s::cl_float4{0.5f, 0.4f, NAN, INFINITY});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    s::cl_int r3 = r.z();
    s::cl_int r4 = r.w();

    assert(r1 == 0);
    assert(r2 == 0);
    assert(r3 == 0);
    assert(r4 == -1);
  }

  // isnan
  {
    s::cl_int4 r{0};
    {
      s::buffer<s::cl_int4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isnanF4F4>([=]() {
          AccR[0] = s::isnan(s::cl_float4{0.5f, 0.4f, NAN, INFINITY});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    s::cl_int r3 = r.z();
    s::cl_int r4 = r.w();

    assert(r1 == 0);
    assert(r2 == 0);
    assert(r3 == -1);
    assert(r4 == 0);
  }

  // isnormal
  {
    s::cl_int4 r{0};
    {
      s::buffer<s::cl_int4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isnormalF4F4>([=]() {
          AccR[0] = s::isnormal(s::cl_float4{0.5f, 0.4f, NAN, INFINITY});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    s::cl_int r3 = r.z();
    s::cl_int r4 = r.w();

    assert(r1 == -1);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == 0);
  }

  // isordered
  {
    s::cl_int4 r{0};
    {
      s::buffer<s::cl_int4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isorderedF4F4>([=]() {
          AccR[0] = s::isordered(s::cl_float4{0.5f, 0.6f, NAN, INFINITY},
                                 s::cl_float4{0.5f, 0.5f, 0.5f, 0.5f});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    s::cl_int r3 = r.z();
    s::cl_int r4 = r.w();

    assert(r1 == -1);
    assert(r2 == -1);
    assert(r3 == 0);
    assert(r4 == -1); // infinity is ordered.
  }

  // isunordered
  {
    s::cl_int4 r{0};
    {
      s::buffer<s::cl_int4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class isunorderedF4F4>([=]() {
          AccR[0] = s::isunordered(s::cl_float4{0.5f, 0.6f, NAN, INFINITY},
                                   s::cl_float4{0.5f, 0.5f, 0.5f, 0.5f});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    s::cl_int r3 = r.z();
    s::cl_int r4 = r.w();

    assert(r1 == 0);
    assert(r2 == 0);
    assert(r3 == -1);
    assert(r4 == 0);
  }

  // signbit
  {
    s::cl_int3 r{0};
    {
      s::buffer<s::cl_int3, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class signbitF3>([=]() {
          AccR[0] = s::signbit(s::cl_float3{0.5f, -12.0f, INFINITY});
        });
      });
    }
    s::cl_int r1 = r.x();
    s::cl_int r2 = r.y();
    s::cl_int r3 = r.z();

    assert(r1 == 0);
    assert(r2 == -1);
    assert(r3 == 0);
  }

  // any.
  // Call to the device function with vector parameters work. Scalars do not.
  {
    s::cl_int r{0};
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class anyI4>([=]() {
          AccR[0] = s::any(s::cl_int4{-12, -12, 0, 1});
        });
      });
    }
    s::cl_int r1 = r;

    assert(r1 == 1);
  }

  // any.
  // Call to the device function with vector parameters work. Scalars do not.
  {
    s::cl_int r{0};
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class anyI4negative>([=]() {
          AccR[0] = s::any(s::cl_int4{-12, -12, -12, -12});
        });
      });
    }
    s::cl_int r1 = r;

    assert(r1 == 1);
  }

  // any.
  // Call to the device function with vector parameters work. Scalars do not.
  {
    s::cl_int r{0};
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class anyI4zero>([=]() {
          AccR[0] = s::any(s::cl_int4{0, 0, 0, 0});
        });
      });
    }
    s::cl_int r1 = r;

    assert(r1 == 0);
  }

  // any.
  // Call to the device function with vector parameters work. Scalars do not.
  {
    s::cl_int r{0};
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class anyI4positive>([=]() {
          AccR[0] = s::any(s::cl_int4{12, 12, 12, 12});
        });
      });
    }
    s::cl_int r1 = r;

    assert(r1 == 0);
  }

  // all.
  // Call to the device function with vector parameters work. Scalars do not.
  {
    s::cl_int r{0};
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class allI4>([=]() {
          AccR[0] = s::all(s::cl_int4{-12, -12, -12, -12});
          // Infinity (positive or negative) or Nan are not integers.
          // Passing them creates inconsistent results between host and device
          // execution.
        });
      });
    }
    s::cl_int r1 = r;

    assert(r1 == 1);
  }

  // all.
  // Call to the device function with vector parameters work. Scalars do not.
  {
    s::cl_int r{0};
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class allI4negative>([=]() {
          AccR[0] = s::all(s::cl_int4{-12, -12, -12, -12});
        });
      });
    }
    s::cl_int r1 = r;

    assert(r1 == 1);
  }

  // all.
  // Call to the device function with vector parameters work. Scalars do not.
  {
    s::cl_int r{0};
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class allI4zero>([=]() {
          AccR[0] = s::all(s::cl_int4{0, 0, 0, 0});
        });
      });
    }
    s::cl_int r1 = r;

    assert(r1 == 0);
  }

  // all.
  // Call to the device function with vector parameters work. Scalars do not.
  {
    s::cl_int r{0};
    {
      s::buffer<s::cl_int, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class allI4positive>([=]() {
          AccR[0] = s::all(s::cl_int4{12, 12, 12, 12});
        });
      });
    }
    s::cl_int r1 = r;

    assert(r1 == 0);
  }

  // bitselect
  {
    s::cl_float4 r{0};
    {
      s::buffer<s::cl_float4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class bitselectF4F4F4>([=]() {
          AccR[0] = s::bitselect(s::cl_float4{112.112, 12.12, 0, 0.0},
                                 s::cl_float4{34.34, 23.23, 1, 0.0},
                                 s::cl_float4{3.3, 6.6, 1, 0.0});
        }); // Using NAN/INFINITY as any float produced consistent results
            // between host and device.
      });
    }
    s::cl_float r1 = r.x();
    s::cl_float r2 = r.y();
    s::cl_float r3 = r.z();
    s::cl_float r4 = r.w();

    assert(abs(r1 - 80.5477f) < 0.0001);
    assert(abs(r2 - 18.2322f) < 0.0001);
    assert(abs(r3 - 1.0f) < 0.01);
    assert(abs(r4 - 0.0f) < 0.01);
  }

  // select
  {
    s::cl_float4 r{0};
    {
      s::buffer<s::cl_float4, 1> BufR(&r, s::range<1>(1));
      s::queue myQueue;
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::write>(cgh);
        cgh.single_task<class selectF4F4I4>([=]() {
          AccR[0] = s::select(s::cl_float4{112.112f, 34.34f, 112.112f, 34.34f},
                              s::cl_float4{34.34f, 112.112f, 34.34f, 112.112f},
                              s::cl_int4{0, -1, 0, 1});
          // Using NAN/infinity as an input, which gets
          // selected by -1, produces a NAN/infinity as expected.
        });
      });
    }
    s::cl_float r1 = r.x();
    s::cl_float r2 = r.y();
    s::cl_float r3 = r.z();
    s::cl_float r4 = r.w();

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
