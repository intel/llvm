// RUN: %clang -I %S/Inputs -fsycl-device-only -Xclang -fsycl-int-header=%t.h %s -c -o kernel.spv
// RUN: FileCheck -input-file=%t.h %s

// CHECK:template <> struct KernelInfo<class KernelName> {
// CHECK:template <> struct KernelInfo<::nm1::nm2::KernelName0> {
// CHECK:template <> struct KernelInfo<::nm1::KernelName1> {
// CHECK:template <> struct KernelInfo<::nm1::KernelName3<::nm1::nm2::KernelName0>> {
// CHECK:template <> struct KernelInfo<::nm1::KernelName3<::nm1::KernelName1>> {
// CHECK:template <> struct KernelInfo<::nm1::KernelName4<::nm1::nm2::KernelName0>> {
// CHECK:template <> struct KernelInfo<::nm1::KernelName4<::nm1::KernelName1>> {
// CHECK:template <> struct KernelInfo<::nm1::KernelName3<KernelName5>> {
// CHECK:template <> struct KernelInfo<::nm1::KernelName4<KernelName7>> {
// CHECK:template <> struct KernelInfo<::nm1::KernelName8<::nm1::nm2::C>> {
// CHECK:template <> struct KernelInfo<::TmplClassInAnonNS<ClassInAnonNS>> {

// This test checks if the SYCL device compiler is able to generate correct
// integration header when the kernel name class is expressed in different
// forms.

#include "sycl.hpp"

template <typename KernelName, typename KernelType>
__attribute__((sycl_kernel)) void kernel_single_task(KernelType kernelFunc) {
  kernelFunc();
}

namespace nm1 {
  namespace nm2 {
    class C {};
    class KernelName0 : public C {};
  } // namespace nm2

  class KernelName1;

  template <typename T> class KernelName3;
  template <typename T> class KernelName4;
  template <typename... T> class KernelName8;

  template <> class KernelName3<nm1::nm2::KernelName0>;
  template <> class KernelName3<KernelName1>;

  template <> class KernelName4<nm1::nm2::KernelName0> {};
  template <> class KernelName4<KernelName1> {};

} // namespace nm1

namespace {
  class ClassInAnonNS;
  template <typename T> class TmplClassInAnonNS;
}

struct MyWrapper {
  class KN101 {};

  int test() {

    cl::sycl::accessor<char, 1, cl::sycl::access::mode::read> acc;

    // Acronyms used to designate a test combination:
    //   Declaration levels: 'T'-translation unit, 'L'-local scope,
    //                       'C'-containing class, 'P'-"in place", '-'-N/A
    //   Class definition:   'I'-incomplete (not defined), 'D' - defined,
    //                       '-'-N/A
    // Test combination positional parameters:
    // 0: Kernel class declaration level
    // 1: Kernel class definition
    // 2: Declaration level of the template argument class of the kernel class
    // 3: Definition of the template argument class of the kernel class

    // PI--
    // traditional in-place incomplete type
    kernel_single_task<class KernelName>([=]() { acc.use(); });

    // TD--
    // a class completely defined within a namespace at
    // translation unit scope
    kernel_single_task<nm1::nm2::KernelName0>([=]() { acc.use(); });

    // TI--
    // an incomplete class forward-declared in a namespace at
    // translation unit scope
    kernel_single_task<nm1::KernelName1>([=]() { acc.use(); });

    // TITD
    // an incomplete template specialization class with defined class as
    // argument declared in a namespace at translation unit scope
    kernel_single_task<nm1::KernelName3<nm1::nm2::KernelName0>>(
      [=]() { acc.use(); });

    // TITI
    // an incomplete template specialization class with incomplete class as
    // argument forward-declared in a namespace at translation unit scope
    kernel_single_task<nm1::KernelName3<nm1::KernelName1>>(
      [=]() { acc.use(); });

    // TDTD
    // a defined template specialization class with defined class as argument
    // declared in a namespace at translation unit scope
    kernel_single_task<nm1::KernelName4<nm1::nm2::KernelName0>>(
      [=]() { acc.use(); });

    // TDTI
    // a defined template specialization class with incomplete class as
    // argument forward-declared in a namespace at translation unit scope
    kernel_single_task<nm1::KernelName4<nm1::KernelName1>>(
      [=]() { acc.use(); });

    // TIPI
    // an incomplete template specialization class with incomplete class as
    // argument forward-declared "in-place"
    kernel_single_task<nm1::KernelName3<class KernelName5>>(
      [=]() { acc.use(); });

    // TDPI
    // a defined template specialization class with incomplete class as
    // argument forward-declared "in-place"
    kernel_single_task<nm1::KernelName4<class KernelName7>>(
      [=]() { acc.use(); });

    // TPITD
    // a defined template pack specialization class with defined class
    // as argument declared in a namespace at translation unit scope
    kernel_single_task<nm1::KernelName8<nm1::nm2::C>>(
      [=]() { acc.use(); });

    // kernel name type is a templated class, both the top-level class and the
    // template argument are declared in the anonymous namespace
    kernel_single_task<TmplClassInAnonNS<class ClassInAnonNS>>(
      [=]() { acc.use(); });

    return 0;
  }
};

#ifndef __SYCL_DEVICE_ONLY__
using namespace cl::sycl::detail;
#endif // __SYCL_DEVICE_ONLY__

int main() {
  MyWrapper w;
  int a = w.test();
#ifndef __SYCL_DEVICE_ONLY__
  KernelInfo<class KernelName>::getName();
  KernelInfo<class nm1::nm2::KernelName0>::getName();
  KernelInfo<class nm1::KernelName1>::getName();
  KernelInfo<class nm1::KernelName3<nm1::nm2::KernelName0>>::getName();
  KernelInfo<class nm1::KernelName3<class nm1::KernelName1>>::getName();
  KernelInfo<class nm1::KernelName4<nm1::nm2::KernelName0>>::getName();
  KernelInfo<class nm1::KernelName4<class nm1::KernelName1>>::getName();
  KernelInfo<class nm1::KernelName3<class KernelName5>>::getName();
  KernelInfo<class nm1::KernelName4<class KernelName7>>::getName();
  KernelInfo<class nm1::KernelName8<nm1::nm2::C>>::getName();
  KernelInfo<class TmplClassInAnonNS<class ClassInAnonNS>>::getName();
#endif //__SYCL_DEVICE_ONLY__
}
