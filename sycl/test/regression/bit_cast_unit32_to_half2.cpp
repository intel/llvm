// RUN: %clangxx -fsycl -fpreview-breaking-changes  %s -fsyntax-only
#include <sycl/sycl.hpp>

SYCL_EXTERNAL uint32_t test(uint32_t a) {
   sycl::half2 ah=sycl::bit_cast<sycl::half2, uint32_t>(a);
   return sycl::bit_cast<uint32_t, sycl::half2>(ah);
}                                                                    
