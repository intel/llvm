// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -sycl-std=2020 -fsycl-int-header=%t.h %s
// RUN: FileCheck -input-file=%t.h %s

// This test checks integration header contents for free functions kernels with
// parameter types that have default template arguments.

#include "mock_properties.hpp"
#include "sycl.hpp"

namespace ns {

struct notatuple {
  int a;
};

namespace ns1 {
template <typename A = notatuple>
class hasDefaultArg {

};
}

template <typename T, typename = int, int a = 12, typename = notatuple, typename ...TS> struct Arg {
  T val;
};

[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel",
                                              2)]] void
simple(Arg<char>){
}

}

[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel",
                                              2)]] void
simple1(ns::Arg<ns::ns1::hasDefaultArg<>>){
}


template <typename T>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void
templated(ns::Arg<T, float, 3>, T end) {
}

template void templated(ns::Arg<int, float, 3>, int);

using namespace ns;

template <typename T>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void
templated2(Arg<T, notatuple>, T end) {
}

template void templated2(Arg<int, notatuple>, int);

template <typename T, int a = 3>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void
templated3(Arg<T, notatuple, a, ns1::hasDefaultArg<>, int, int>, T end) {
}

template void templated3(Arg<int, notatuple, 3, ns1::hasDefaultArg<>, int, int>, int);

using AliasType = float;
template void templated3(Arg<AliasType, notatuple, 3, ns1::hasDefaultArg<>, int, int>, AliasType);

namespace sycl {
template <typename T> struct X {};
template <> struct X<int> {};
namespace detail {
struct Y {};
} // namespace detail
template <> struct X<detail::Y> {};
} // namespace sycl
using namespace sycl;
template <typename T, typename = X<detail::Y>> struct Arg1 { T val; };

[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel",
                                              2)]] void
foo(Arg1<int> arg) {
  arg.val = 42;
}

namespace TestNamespace {
template <typename T>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void
templated(Arg<T, float, 3>, T end) {
}

typedef int TypedefInt;

template void templated(Arg<TypedefInt, float, 3>, TypedefInt end);
}

namespace TestNamespace {
inline namespace _V1 {
template <typename T, int a = 3>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void
templated1(Arg<T, float, a>, T end) {
}
template void templated1(Arg<TypedefInt, float, 10>, TypedefInt end);
}
inline namespace _V2 {
template <typename T, int a = 5>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void
templated1(Arg<T, T, a>, T end) {
}
template void templated1(Arg<TypedefInt, TypedefInt>, TypedefInt end);
}
}

namespace {
template <typename T>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 1)]] void
templated(T start, T end) {
}

template void templated(float start, float end);
}

struct TestStruct {
  int a;
  float b;
};

template void templated(ns::Arg<TestStruct, float, 3>, TestStruct);

class BaseClass {
  int base;
public:
  BaseClass() : base(0) {}
};

class ChildOne : public BaseClass {
  int child;
public:
  ChildOne() : child(1) {}
};

class ChildTwo : protected BaseClass {
  int child;
public:
  ChildTwo() : child(1) {}
};

class ChildThree : private BaseClass {
  int child;
public:
  ChildThree() : child(1) {}
};

namespace One::Two::Three {
  struct TestStruct {
    int a;
    float b;
  };

  struct AnotherStruct: public TestStruct {
    int c;
  };
}

namespace {
template void templated(BaseClass, BaseClass);
template void templated(ChildOne, ChildOne);
template void templated(ChildTwo, ChildTwo);
template void templated(ChildThree, ChildThree);
template void templated(sycl::id<2>, sycl::id<2>);
template void templated(sycl::range<3>, sycl::range<3>);
template void templated(int *, int *);
template void templated(sycl::X<ChildTwo>, sycl::X<ChildTwo>);
}

namespace TestNamespace {
  inline namespace _V1 {
    template void templated1(Arg<One::Two::Three::AnotherStruct, float, 10>, One::Two::Three::AnotherStruct end);
  }
}

template <typename... Args>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void
variadic_templated(Args... args) {
}

template void variadic_templated(int, float, char);
template void variadic_templated(int, float, char, int);
template void variadic_templated<float, float>(float, float);

template <typename T, typename... Args>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void
variadic_templated1(T b, Args... args) {
}

template void variadic_templated1<float, char, char>(float, char, char);
template void variadic_templated1(int, float, char);

namespace Testing::Tests {
  template <typename T, typename... Args>
  [[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void
  variadic_templated(T b, Args... args) {
  }

  template void variadic_templated<float, float>(float, float);
  template void variadic_templated(int, int, int, int);
}


// CHECK: namespace sycl {
// CHECK-NEXT:  inline namespace _V1 {
// CHECK-NEXT:  namespace detail {
// CHECK-NEXT:  // names of all kernels defined in the corresponding source
// CHECK-NEXT:  static constexpr
// CHECK-NEXT:  const char* const kernel_names[] = {
// CHECK-NEXT:    "_ZN16__sycl_kernel_ns6simpleENS_3ArgIciLi12ENS_9notatupleEJEEE",
// CHECK-NEXT:    "_Z21__sycl_kernel_simple1N2ns3ArgINS_3ns113hasDefaultArgINS_9notatupleEEEiLi12ES3_JEEE",
// CHECK-NEXT:    "_Z23__sycl_kernel_templatedIiEvN2ns3ArgIT_fLi3ENS0_9notatupleEJEEES2_",
// CHECK-NEXT:    "_Z24__sycl_kernel_templated2IiEvN2ns3ArgIT_NS0_9notatupleELi12ES3_JEEES2_",
// CHECK-NEXT:    "_Z24__sycl_kernel_templated3IiLi3EEvN2ns3ArgIT_NS0_9notatupleEXT0_ENS0_3ns113hasDefaultArgIS3_EEJiiEEES2_",
// CHECK-NEXT:    "_Z24__sycl_kernel_templated3IfLi3EEvN2ns3ArgIT_NS0_9notatupleEXT0_ENS0_3ns113hasDefaultArgIS3_EEJiiEEES2_",
// CHECK-NEXT:    "_Z17__sycl_kernel_foo4Arg1IiN4sycl1XINS0_6detail1YEEEE",
// CHECK-NEXT:    "_ZN27__sycl_kernel_TestNamespace9templatedIiEEvN2ns3ArgIT_fLi3ENS1_9notatupleEJEEES3_",
// CHECK-NEXT:    "_ZN27__sycl_kernel_TestNamespace3_V110templated1IiLi10EEEvN2ns3ArgIT_fXT0_ENS2_9notatupleEJEEES4_",
// CHECK-NEXT:    "_ZN27__sycl_kernel_TestNamespace3_V210templated1IiLi12EEEvN2ns3ArgIT_S4_XT0_ENS2_9notatupleEJEEES4_",
// CHECK-NEXT:    "_ZN26__sycl_kernel__GLOBAL__N_19templatedIfEEvT_S1_",
// CHECK-NEXT:    "_Z23__sycl_kernel_templatedI10TestStructEvN2ns3ArgIT_fLi3ENS1_9notatupleEJEEES3_",
// CHECK-NEXT:    "_ZN26__sycl_kernel__GLOBAL__N_19templatedI9BaseClassEEvT_S2_",
// CHECK-NEXT:    "_ZN26__sycl_kernel__GLOBAL__N_19templatedI8ChildOneEEvT_S2_",
// CHECK-NEXT:    "_ZN26__sycl_kernel__GLOBAL__N_19templatedI8ChildTwoEEvT_S2_",
// CHECK-NEXT:    "_ZN26__sycl_kernel__GLOBAL__N_19templatedI10ChildThreeEEvT_S2_",
// CHECK-NEXT:    "_ZN26__sycl_kernel__GLOBAL__N_19templatedIN4sycl3_V12idILi2EEEEEvT_S5_",
// CHECK-NEXT:    "_ZN26__sycl_kernel__GLOBAL__N_19templatedIN4sycl3_V15rangeILi3EEEEEvT_S5_",
// CHECK-NEXT:    "_ZN26__sycl_kernel__GLOBAL__N_19templatedIPiEEvT_S2_",
// CHECK-NEXT:    "_ZN26__sycl_kernel__GLOBAL__N_19templatedIN4sycl1XI8ChildTwoEEEEvT_S5_",
// CHECK-NEXT:    "_ZN27__sycl_kernel_TestNamespace3_V110templated1IN3One3Two5Three13AnotherStructELi10EEEvN2ns3ArgIT_fXT0_ENS6_9notatupleEJEEES8_",
// CHECK-NEXT:    "_Z32__sycl_kernel_variadic_templatedIJifcEEvDpT_",
// CHECK-NEXT:    "_Z32__sycl_kernel_variadic_templatedIJifciEEvDpT_",
// CHECK-NEXT:    "_Z32__sycl_kernel_variadic_templatedIJffEEvDpT_",
// CHECK-NEXT:    "_Z33__sycl_kernel_variadic_templated1IfJccEEvT_DpT0_", 
// CHECK-NEXT:    "_Z33__sycl_kernel_variadic_templated1IiJfcEEvT_DpT0_",
// CHECK-NEXT:    "_ZN21__sycl_kernel_Testing5Tests18variadic_templatedIfJfEEEvT_DpT0_", 
// CHECK-NEXT:    "_ZN21__sycl_kernel_Testing5Tests18variadic_templatedIiJiiiEEEvT_DpT0_", 
// CHECK-NEXT:    "",
// CHECK-NEXT:  };

// CHECK: Forward declarations of kernel and its argument types:
// CHECK-NEXT: namespace ns { 
// CHECK-NEXT: struct notatuple;
// CHECK-NEXT: }
// CHECK-NEXT: namespace ns { 
// CHECK-NEXT: template <typename T, typename, int a, typename, typename ...TS> struct Arg;
// CHECK-NEXT: }

// CHECK: namespace ns {
// CHECK-NEXT: void simple(ns::Arg<char, int, 12, ns::notatuple> );
// CHECK-NEXT: } // namespace ns
// CHECK: static constexpr auto __sycl_shim1() {
// CHECK-NEXT:   return (void (*)(struct ns::Arg<char, int, 12, struct ns::notatuple>))ns::simple;
// CHECK-NEXT: }

// CHECK: Forward declarations of kernel and its argument types:
// CHECK: namespace ns {
// CHECK: namespace ns1 {
// CHECK-NEXT: template <typename A> class hasDefaultArg;
// CHECK-NEXT: }}

// CHECK: void simple1(ns::Arg<ns::ns1::hasDefaultArg<ns::notatuple>, int, 12, ns::notatuple> );
// CHECK-NEXT: static constexpr auto __sycl_shim2() {
// CHECK-NEXT:   return (void (*)(struct ns::Arg<class ns::ns1::hasDefaultArg<struct ns::notatuple>, int, 12, struct ns::notatuple>))simple1;
// CHECK-NEXT: }

// CHECK: template <typename T> void templated(ns::Arg<T, float, 3, ns::notatuple> , T end);
// CHECK-NEXT: static constexpr auto __sycl_shim3() {
// CHECK-NEXT:   return (void (*)(struct ns::Arg<int, float, 3, struct ns::notatuple>, int))templated<int>;
// CHECK-NEXT: }

// CHECK: template <typename T> void templated2(ns::Arg<T, ns::notatuple, 12, ns::notatuple> , T end);
// CHECK-NEXT: static constexpr auto __sycl_shim4() {
// CHECK-NEXT:   return (void (*)(struct ns::Arg<int, struct ns::notatuple, 12, struct ns::notatuple>, int))templated2<int>;
// CHECK-NEXT: }

// CHECK: template <typename T, int a> void templated3(ns::Arg<T, ns::notatuple, a, ns::ns1::hasDefaultArg<ns::notatuple>, int, int> , T end);
// CHECK-NEXT: static constexpr auto __sycl_shim5() {
// CHECK-NEXT:   return (void (*)(struct ns::Arg<int, struct ns::notatuple, 3, class ns::ns1::hasDefaultArg<struct ns::notatuple>, int, int>, int))templated3<int, 3>;
// CHECK-NEXT: }

// CHECK Forward declarations of kernel and its argument types:
// CHECK: namespace sycl { namespace detail {
// CHECK-NEXT: struct Y;
// CHECK-NEXT: }}
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <typename T> struct X;
// CHECK-NEXT: }
// CHECK-NEXT: template <typename T, typename> struct Arg1;

// CHECK: void foo(Arg1<int, sycl::X<sycl::detail::Y> > arg);
// CHECK-NEXT: static constexpr auto __sycl_shim7() {
// CHECK-NEXT:   return (void (*)(struct Arg1<int, struct sycl::X<struct sycl::detail::Y> >))foo;
// CHECK-NEXT: }

// CHECK: namespace TestNamespace {
// CHECK-NEXT:  template <typename T> void templated(ns::Arg<T, float, 3, ns::notatuple> , T end);
// CHECK-NEXT:  } // namespace TestNamespace
  
// CHECK:  static constexpr auto __sycl_shim8() {
// CHECK-NEXT:    return (void (*)(struct ns::Arg<int, float, 3, struct ns::notatuple>, int))TestNamespace::templated<int>;
// CHECK-NEXT:  }
// CHECK-NEXT:  namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_kernel<__sycl_shim8()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim8(), 2> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  }

// CHECK: namespace TestNamespace {
// CHECK-NEXT:   inline namespace _V1 {
// CHECK-NEXT:   template <typename T, int a> void templated1(ns::Arg<T, float, a, ns::notatuple> , T end);
// CHECK-NEXT:   } // inline namespace _V1
// CHECK-NEXT:   } // namespace TestNamespace
// CHECK:   static constexpr auto __sycl_shim9() {
// CHECK-NEXT:    return (void (*)(struct ns::Arg<int, float, 10, struct ns::notatuple>, int))TestNamespace::_V1::templated1<int, 10>;
// CHECK-NEXT:   }
// CHECK-NEXT:   namespace sycl {
// CHECK-NEXT:   template <>
// CHECK-NEXT:   struct ext::oneapi::experimental::is_kernel<__sycl_shim9()> {
// CHECK-NEXT:     static constexpr bool value = true;
// CHECK-NEXT:   };
// CHECK-NEXT:   template <>
// CHECK-NEXT:   struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim9(), 2> {
// CHECK-NEXT:     static constexpr bool value = true;
// CHECK-NEXT:   };
// CHECK-NEXT:   }

// CHECK: namespace TestNamespace {
// CHECK-NEXT:  inline namespace _V2 {
// CHECK-NEXT:  template <typename T, int a> void templated1(ns::Arg<T, T, a, ns::notatuple> , T end);
// CHECK-NEXT:  } // inline namespace _V2
// CHECK-NEXT:  } // namespace TestNamespace
// CHECK:  static constexpr auto __sycl_shim10() {
// CHECK-NEXT:    return (void (*)(struct ns::Arg<int, int, 12, struct ns::notatuple>, int))TestNamespace::_V2::templated1<int, 12>;
// CHECK-NEXT:  }
// CHECK-NEXT:  namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_kernel<__sycl_shim10()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim10(), 2> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  }

// CHECK: namespace {
// CHECK-NEXT:  template <typename T> void templated(T start, T end);
// CHECK-NEXT:  } // namespace 
// CHECK:  static constexpr auto __sycl_shim11() {
// CHECK-NEXT:    return (void (*)(float, float))templated<float>;
// CHECK-NEXT:  }
// CHECK-NEXT:  namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_kernel<__sycl_shim11()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim11(), 1> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  }

// CHECK: struct TestStruct;
// CHECK: template <typename T> void templated(ns::Arg<T, float, 3, ns::notatuple> , T end);
// CHECK-NEXT: static constexpr auto __sycl_shim12() {
// CHECK-NEXT:  return (void (*)(struct ns::Arg<struct TestStruct, float, 3, struct ns::notatuple>, struct TestStruct))templated<struct TestStruct>;
// CHECK-NEXT:}
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_kernel<__sycl_shim12()> {
// CHECK-NEXT:  static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim12(), 2> {
// CHECK-NEXT:  static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: }

// CHECK: class BaseClass;
// CHECK: namespace {
// CHECK-NEXT: template <typename T> void templated(T start, T end);
// CHECK-NEXT: } // namespace 
// CHECK: static constexpr auto __sycl_shim13() {
// CHECK-NEXT:  return (void (*)(class BaseClass, class BaseClass))templated<class BaseClass>;
// CHECK-NEXT: }
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_kernel<__sycl_shim13()> {
// CHECK-NEXT:  static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim13(), 1> {
// CHECK-NEXT:  static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: }

// CHECK: class ChildOne;
// CHECK: namespace {
// CHECK-NEXT: template <typename T> void templated(T start, T end);
// CHECK-NEXT: } // namespace 
// CHECK: static constexpr auto __sycl_shim14() {
// CHECK-NEXT:  return (void (*)(class ChildOne, class ChildOne))templated<class ChildOne>;
// CHECK-NEXT: }
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_kernel<__sycl_shim14()> {
// CHECK-NEXT:  static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim14(), 1> {
// CHECK-NEXT:  static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: }

// CHECK: class ChildTwo;
// CHECK: namespace {
// CHECK-NEXT: template <typename T> void templated(T start, T end);
// CHECK-NEXT: } // namespace 
// CHECK: static constexpr auto __sycl_shim15() {
// CHECK-NEXT:  return (void (*)(class ChildTwo, class ChildTwo))templated<class ChildTwo>;
// CHECK-NEXT: }
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_kernel<__sycl_shim15()> {
// CHECK-NEXT:  static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim15(), 1> {
// CHECK-NEXT:  static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: }

// CHECK: class ChildThree;
// CHECK: namespace {
// CHECK-NEXT: template <typename T> void templated(T start, T end);
// CHECK-NEXT: } // namespace 
// CHECK: static constexpr auto __sycl_shim16() {
// CHECK-NEXT:  return (void (*)(class ChildThree, class ChildThree))templated<class ChildThree>;
// CHECK-NEXT: }
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_kernel<__sycl_shim16()> {
// CHECK-NEXT:  static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim16(), 1> {
// CHECK-NEXT:  static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: }

// CHECK: namespace sycl { inline namespace _V1 { 
// CHECK-NEXT:  template <int dim> struct id;
// CHECK-NEXT:  }}
// CHECK:  namespace {
// CHECK-NEXT:  template <typename T> void templated(T start, T end);
// CHECK-NEXT:  } // namespace 
// CHECK:  static constexpr auto __sycl_shim17() {
// CHECK-NEXT:    return (void (*)(struct sycl::id<2>, struct sycl::id<2>))templated<struct sycl::id<2>>;
// CHECK-NEXT:  }
// CHECK-NEXT:  namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_kernel<__sycl_shim17()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim17(), 1> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  }

// CHECK: namespace sycl { inline namespace _V1 { 
// CHECK-NEXT:  template <int dim> struct range;
// CHECK-NEXT:  }}
// CHECK:  namespace {
// CHECK-NEXT:  template <typename T> void templated(T start, T end);
// CHECK-NEXT:  } // namespace 
// CHECK:  static constexpr auto __sycl_shim18() {
// CHECK-NEXT:    return (void (*)(struct sycl::range<3>, struct sycl::range<3>))templated<struct sycl::range<3>>;
// CHECK-NEXT:  }
// CHECK-NEXT:  namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_kernel<__sycl_shim18()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim18(), 1> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  }

// CHECK: namespace {
// CHECK-NEXT:  template <typename T> void templated(T start, T end);
// CHECK-NEXT:  } // namespace 
// CHECK:  static constexpr auto __sycl_shim19() {
// CHECK-NEXT:    return (void (*)(int *, int *))templated<int *>;
// CHECK-NEXT:  }
// CHECK-NEXT:  namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_kernel<__sycl_shim19()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim19(), 1> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  }

// CHECK: namespace {
// CHECK-NEXT:  template <typename T> void templated(T start, T end);
// CHECK-NEXT:  } // namespace 
// CHECK:  static constexpr auto __sycl_shim20() {
// CHECK-NEXT:    return (void (*)(struct sycl::X<class ChildTwo>, struct sycl::X<class ChildTwo>))templated<struct sycl::X<class ChildTwo>>;
// CHECK-NEXT:  }
// CHECK-NEXT:  namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_kernel<__sycl_shim20()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim20(), 1> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  }

// CHECK: namespace One { namespace Two { namespace Three { 
// CHECK-NEXT:  struct AnotherStruct;
// CHECK-NEXT:  }}}
// CHECK:  namespace TestNamespace {
// CHECK-NEXT:  inline namespace _V1 {
// CHECK-NEXT:  template <typename T, int a> void templated1(ns::Arg<T, float, a, ns::notatuple> , T end);
// CHECK-NEXT:  } // inline namespace _V1
// CHECK-NEXT:  } // namespace TestNamespace
// CHECK:  static constexpr auto __sycl_shim21() {
// CHECK-NEXT:    return (void (*)(struct ns::Arg<struct One::Two::Three::AnotherStruct, float, 10, struct ns::notatuple>, struct One::Two::Three::AnotherStruct))TestNamespace::_V1::templated1<struct One::Two::Three::AnotherStruct, 10>;
// CHECK-NEXT:  }
// CHECK-NEXT:  namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_kernel<__sycl_shim21()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim21(), 2> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  }

// CHECK: template <typename ... Args> void variadic_templated(Args... args);
// CHECK-NEXT: static constexpr auto __sycl_shim22() {
// CHECK-NEXT:  return (void (*)(int, float, char))variadic_templated<int, float, char>;
// CHECK-NEXT: }
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_kernel<__sycl_shim22()> {
// CHECK-NEXT:  static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim22(), 2> {
// CHECK-NEXT:  static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: }

// CHECK: template <typename ... Args> void variadic_templated(Args... args);
// CHECK-NEXT: static constexpr auto __sycl_shim23() {
// CHECK-NEXT:  return (void (*)(int, float, char, int))variadic_templated<int, float, char, int>;
// CHECK-NEXT: }
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_kernel<__sycl_shim23()> {
// CHECK-NEXT:  static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim23(), 2> {
// CHECK-NEXT:  static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: }

// CHECK: template <typename ... Args> void variadic_templated(Args... args);
// CHECK-NEXT: static constexpr auto __sycl_shim24() {
// CHECK-NEXT:  return (void (*)(float, float))variadic_templated<float, float>;
// CHECK-NEXT: }
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_kernel<__sycl_shim24()> {
// CHECK-NEXT:  static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim24(), 2> {
// CHECK-NEXT:  static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: }

// CHECK: template <typename T, typename ... Args> void variadic_templated1(T b, Args... args);
// CHECK-NEXT: static constexpr auto __sycl_shim25() {
// CHECK-NEXT:  return (void (*)(float, char, char))variadic_templated1<float, char, char>;
// CHECK-NEXT: }
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_kernel<__sycl_shim25()> {
// CHECK-NEXT:  static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim25(), 2> {
// CHECK-NEXT:  static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: }

// CHECK: template <typename T, typename ... Args> void variadic_templated1(T b, Args... args);
// CHECK-NEXT: static constexpr auto __sycl_shim26() {
// CHECK-NEXT:  return (void (*)(int, float, char))variadic_templated1<int, float, char>;
// CHECK-NEXT: }
// CHECK-NEXT: namespace sycl {
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_kernel<__sycl_shim26()> {
// CHECK-NEXT:  static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: template <>
// CHECK-NEXT: struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim26(), 2> {
// CHECK-NEXT:  static constexpr bool value = true;
// CHECK-NEXT: };
// CHECK-NEXT: }

// CHECK: namespace Testing {
// CHECK-NEXT:  namespace Tests {
// CHECK-NEXT:  template <typename T, typename ... Args> void variadic_templated(T b, Args... args);
// CHECK-NEXT:  } // namespace Tests
// CHECK-NEXT:  } // namespace Testing
// CHECK:  static constexpr auto __sycl_shim27() {
// CHECK-NEXT:    return (void (*)(float, float))Testing::Tests::variadic_templated<float, float>;
// CHECK-NEXT:  }
// CHECK-NEXT:  namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_kernel<__sycl_shim27()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim27(), 2> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  }

// CHECK:  namespace Testing {
// CHECK-NEXT:  namespace Tests {
// CHECK-NEXT:  template <typename T, typename ... Args> void variadic_templated(T b, Args... args);
// CHECK-NEXT:  } // namespace Tests
// CHECK-NEXT:  } // namespace Testing
// CHECK:  static constexpr auto __sycl_shim28() {
// CHECK-NEXT:    return (void (*)(int, int, int, int))Testing::Tests::variadic_templated<int, int, int, int>;
// CHECK-NEXT:  }
// CHECK-NEXT:  namespace sycl {
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_kernel<__sycl_shim28()> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  template <>
// CHECK-NEXT:  struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim28(), 2> {
// CHECK-NEXT:    static constexpr bool value = true;
// CHECK-NEXT:  };
// CHECK-NEXT:  }
