// RUN: %clangxx -fsycl -D__ENABLE_USM_ADDR_SPACE__ -fsyntax-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=note,warning
// expected-no-diagnostics

// Checks that extended address spaces are allowed when creating multi_ptr from
// accessors.

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  queue Q;

  buffer<int> Buf{1};
  Q.submit([&](handler &CGH) {
     accessor Acc{Buf, CGH, read_write};
     CGH.single_task([=]() {
       device_ptr<int, access::decorated::yes> MPtr1{Acc};
       device_ptr<int, access::decorated::no> MPtr2{Acc};
       device_ptr<int, access::decorated::legacy> MPtr3{Acc};
       device_ptr<const int, access::decorated::yes> MPtr4{Acc};
       device_ptr<const int, access::decorated::no> MPtr5{Acc};
       device_ptr<const int, access::decorated::legacy> MPtr6{Acc};
       device_ptr<void, access::decorated::yes> MPtr7{Acc};
       device_ptr<void, access::decorated::no> MPtr8{Acc};
       device_ptr<void, access::decorated::legacy> MPtr9{Acc};
       device_ptr<const void, access::decorated::yes> MPtr10{Acc};
       device_ptr<const void, access::decorated::no> MPtr11{Acc};
       device_ptr<const void, access::decorated::legacy> MPtr12{Acc};
     });
   }).wait_and_throw();
  return 0;
}
