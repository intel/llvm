// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -fsycl-int-header=%t.h %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s

// Check if forward declarations of kernel names in anonymous namespace are in
// anonymous namespace in the integration header as well.
// CHECK:namespace  {
// CHECK-NEXT:class ClassInAnonNS;
// CHECK-NEXT:}

// CHECK:namespace  { namespace NestedInAnon {
// CHECK-NEXT:struct StructInAnonymousNS;
// CHECK-NEXT:}}

// CHECK:namespace Named { namespace  {
// CHECK-NEXT:struct IsThisValid;
// CHECK-NEXT:}}

// CHECK:template <> struct KernelInfo<KernelName> {
// CHECK:template <> struct KernelInfo<::nm1::nm2::KernelName0> {
// CHECK:template <> struct KernelInfo<::nm1::KernelName1> {
// CHECK:template <> struct KernelInfo<::nm1::KernelName3<::nm1::nm2::KernelName0>> {
// CHECK:template <> struct KernelInfo<::nm1::KernelName3<::nm1::KernelName1>> {
// CHECK:template <> struct KernelInfo<::nm1::KernelName4<::nm1::nm2::KernelName0>> {
// CHECK:template <> struct KernelInfo<::nm1::KernelName4<::nm1::KernelName1>> {
// CHECK:template <> struct KernelInfo<::nm1::KernelName8<::nm1::nm2::C>> {
// CHECK:template <> struct KernelInfo<::TmplClassInAnonNS<ClassInAnonNS>> {
// CHECK:template <> struct KernelInfo<::nm1::KernelName9<char>> {
// CHECK:template <> struct KernelInfo<::nm1::KernelName3<const volatile ::nm1::KernelName3<const volatile char>>> {

// This test checks if the SYCL device compiler is able to generate correct
// integration header when the kernel name class is expressed in different
// forms.

#include "sycl.hpp"

template <typename KernelName, typename KernelType>
__attribute__((sycl_kernel)) void kernel_single_task(const KernelType &kernelFunc) {
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

  template <typename T, typename...>
  class KernelName9;

} // namespace nm1

namespace {
  class ClassInAnonNS;
  template <typename T> class TmplClassInAnonNS;
}

namespace {
namespace NestedInAnon {
struct StructInAnonymousNS {};
} // namespace NestedInAnon
} // namespace

namespace Named {
namespace {
struct IsThisValid {};
} // namespace
} // namespace Named

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

    // TPITD
    // a defined template pack specialization class with defined class
    // as argument declared in a namespace at translation unit scope
    kernel_single_task<nm1::KernelName8<nm1::nm2::C>>(
      [=]() { acc.use(); });

    // kernel name type is a templated class, both the top-level class and the
    // template argument are declared in the anonymous namespace
    kernel_single_task<TmplClassInAnonNS<class ClassInAnonNS>>(
        [=]() { acc.use(); });

    // kernel name type is a class, declared in the anonymous namespace
    kernel_single_task<ClassInAnonNS>(
        [=]() { acc.use(); });

    // kernel name types declared in nested anonymous namespace
    kernel_single_task<NestedInAnon::StructInAnonymousNS>(
        [=]() { acc.use(); });

    kernel_single_task<Named::IsThisValid>(
        [=]() { acc.use(); });

    // Kernel name type is a templated specialization class with empty template pack argument
    kernel_single_task<nm1::KernelName9<char>>(
        [=]() { acc.use(); });

    // Ensure we print template arguments with CVR qualifiers
    kernel_single_task<nm1::KernelName3<
        const volatile nm1::KernelName3<
            const volatile char>>>(
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
  KernelInfo<class nm1::KernelName9<char>>::getName();
#endif //__SYCL_DEVICE_ONLY__
}
