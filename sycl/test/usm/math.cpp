// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out

// REQUIRES: cpu
// TODO: ptxas fatal   : Unresolved extern function '_Z20__spirv_ocl_lgamma_rfPi'
// XFAIL: cuda

#include <CL/sycl.hpp>

#include <array>
#include <cassert>
#include <cmath>

namespace s = cl::sycl;

int main() {
  s::queue myQueue;

  if (myQueue.get_device()
          .get_info<s::info::device::usm_shared_allocations>()) {
    // fract with unified shared memory
    {
      s::cl_float r{0};
      s::cl_float i{999};
      {
        s::cl_float *Buf = (s::cl_float *)s::malloc_shared(
            sizeof(cl_float) * 2, myQueue.get_device(), myQueue.get_context());
        s::malloc_shared(100, myQueue.get_device(), myQueue.get_context());
        myQueue.submit([&](s::handler &cgh) {
          cgh.single_task<class fractF1UF1>(
              [=]() { Buf[0] = s::fract(s::cl_float{1.5f}, &Buf[1]); });
        });
        myQueue.wait();
        r = Buf[0];
        i = Buf[1];
        s::free(Buf, myQueue.get_context());
      }
      assert(r == 0.5f);
      assert(i == 1.0f);
    }

    // vector fract with unified shared memory
    {
      s::cl_float2 *Buf = (s::cl_float2 *)s::malloc_shared(
          sizeof(cl_float2) * 2, myQueue.get_device(), myQueue.get_context());
      myQueue.submit([&](s::handler &cgh) {
        cgh.single_task<class fractF2UF2>([=]() {
          Buf[0] = s::fract(s::cl_float2{1.5f, 2.5f}, &Buf[1]);
        });
      });
      myQueue.wait();

      s::cl_float r1 = Buf[0].x();
      s::cl_float r2 = Buf[0].y();
      s::cl_float i1 = Buf[1].x();
      s::cl_float i2 = Buf[1].y();

      assert(r1 == 0.5f);
      assert(r2 == 0.5f);
      assert(i1 == 1.0f);
      assert(i2 == 2.0f);
    }

    // lgamma_r with unified shared memory
    {
      s::cl_float r{0};
      s::cl_int i{999};
      {
        s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
        s::cl_int *BufI = (s::cl_int *)s::malloc_shared(
            sizeof(cl_int) * 2, myQueue.get_device(), myQueue.get_context());
        myQueue.submit([&](s::handler &cgh) {
          auto AccR = BufR.get_access<s::access::mode::read_write>(cgh);
          cgh.single_task<class lgamma_rF1PI1>(
              [=]() { AccR[0] = s::lgamma_r(s::cl_float{10.f}, BufI); });
        });
        myQueue.wait();
        i = *BufI;
        s::free(BufI, myQueue.get_context());
      }
      assert(r > 12.8017f && r < 12.8019f); // ~12.8018
      assert(i == 1);                       // tgamma of 10 is ~362880.0
    }

    // lgamma_r with unified shared memory
    {
      s::cl_float r{0};
      s::cl_int i{999};
      {
        s::buffer<s::cl_float, 1> BufR(&r, s::range<1>(1));
        s::cl_int *BufI = (s::cl_int *)s::malloc_shared(
            sizeof(cl_int) * 2, myQueue.get_device(), myQueue.get_context());
        myQueue.submit([&](s::handler &cgh) {
          auto AccR = BufR.get_access<s::access::mode::read_write>(cgh);
          cgh.single_task<class lgamma_rF1PI1_neg>(
              [=]() { AccR[0] = s::lgamma_r(s::cl_float{-2.4f}, BufI); });
        });
        myQueue.wait();
        i = *BufI;
        s::free(BufI, myQueue.get_context());
      }
      assert(r > 0.1024f && r < 0.1026f); // ~0.102583
      assert(i == -1); // tgamma of -2.4 is ~-1.1080299470333461
    }

    // vector lgamma_r with unified shared memory
    {
      s::cl_float2 r{0, 0};
      s::cl_int2 i{0, 0};
      s::buffer<s::cl_float2, 1> BufR(&r, s::range<1>(1));
      s::cl_int2 *BufI = (s::cl_int2 *)s::malloc_shared(
          sizeof(cl_int2) * 2, myQueue.get_device(), myQueue.get_context());
      myQueue.submit([&](s::handler &cgh) {
        auto AccR = BufR.get_access<s::access::mode::read_write>(cgh);
        cgh.single_task<class lgamma_rF2PF2>([=]() {
          AccR[0] = s::lgamma_r(s::cl_float2{10.f, -2.4f}, BufI);
        });
      });
      myQueue.wait();

      s::cl_float r1 = r.x();
      s::cl_float r2 = r.y();
      s::cl_int i1 = BufI->x();
      s::cl_int i2 = BufI->y();

      assert(r1 > 12.8017f && r1 < 12.8019f); // ~12.8018
      assert(r2 > 0.1024f && r2 < 0.1026f);   // ~0.102583
      assert(i1 == 1);                        // tgamma of 10 is ~362880.0
      assert(i2 == -1); // tgamma of -2.4 is ~-1.1080299470333461
    }
  }
  return 0;
}
