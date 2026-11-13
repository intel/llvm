// SYCL_EXTERNAL is only defined if the implementation supports external
// linkage (SYCL 2020 5.6). With relocatable device code (-fsycl-rdc, the
// default) it must be defined; with -fno-sycl-rdc it must never be defined.
// This must hold in every pass of a SYCL compilation, including a host pass
// performed by a third-party compiler (-fsycl-host-compiler).
//
// The check below is evaluated by each compiler invocation (device and host),
// so any pass that disagrees with EXPECT fails the compilation.
//
// RUN: %clangxx -fsycl                -DEXPECT=1 -c %s -o %t.rdc.o
// RUN: %clangxx -fsycl -fno-sycl-rdc  -DEXPECT=0 -c %s -o %t.nordc.o
//
// The DPC++ compiler must get this right on its own, without any SYCL header.
// RUN: %clangxx -fsycl                -DEXPECT=1 -DNO_SYCL_INCLUDE -c %s -o %t.noinc.rdc.o
// RUN: %clangxx -fsycl -fno-sycl-rdc  -DEXPECT=0 -DNO_SYCL_INCLUDE -c %s -o %t.noinc.nordc.o
//
// The external host compiler does not see the driver's -D options; pass EXPECT
// to it through -fsycl-host-compiler-options as well.
// RUN: %if linux %{ %clangxx -fsycl               -fsycl-host-compiler=g++ -fsycl-host-compiler-options="-std=c++17 -DEXPECT=1" -DEXPECT=1 -c %s -o %t.gxx.rdc.o %}
// RUN: %if linux %{ %clangxx -fsycl -fno-sycl-rdc -fsycl-host-compiler=g++ -fsycl-host-compiler-options="-std=c++17 -DEXPECT=0" -DEXPECT=0 -c %s -o %t.gxx.nordc.o %}

#ifndef NO_SYCL_INCLUDE
#include <sycl/sycl.hpp>
// A kernel makes the integration header non-empty, so the
// -fsycl-host-compiler pass sees what a real program sees.
void use_kernel(sycl::queue &q) {
  q.single_task<class K>([] {});
}
#endif

#ifndef EXPECT
#error "EXPECT must be defined to 0 or 1"
#endif

#if defined(SYCL_EXTERNAL) != EXPECT
#error "SYCL_EXTERNAL defined state does not match the RDC mode"
#endif
