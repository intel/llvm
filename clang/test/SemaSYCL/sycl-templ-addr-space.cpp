// RUN: %clang_cc1 -fsycl-is-device -triple spir64 -verify %s

// Test that the compiler no longer asserts while processing this test case.

template <int N>
struct integral_constant {
  static constexpr int val = N;
};
template <typename T>
T &&convert(int);
template <typename T>
auto declval(convert<T>(0));
template <typename Op1, typename Op2>
class sub_group {
  template <typename T, decltype(declval<T>)>
  // expected-note@+1 {{possible target for call}}
  static integral_constant<true> binary_op();
  // expected-error@+1 {{reference to overloaded function could not be resolved; did you mean to call it?}}
  decltype(binary_op<Op1, Op2>) i;
};
// expected-note-re@+2 {{in instantiation {{.*}} requested here}}
template <typename n>
struct group : sub_group<n, int> {
};
template <typename T>
struct wrapper {
  typedef T val;
};
class element_type {
};
struct Container {
  using __element_type = __attribute__((opencl_global)) element_type;
};
template <bool, class Base, class> using BaseImpl = Base;
// expected-note-re@+1 2{{candidate constructor {{.*}} not viable: {{.*}}}}
template <int> class id_type {
  template <class...> struct Base;
  // expected-note-re@+1 {{in instantiation {{.*}} requested here}}
  template <class Der> struct Base<Der> : BaseImpl<Der::val, Base<>, Der> {};
  // expected-note-re@+1 {{in instantiation {{.*}} requested here}}
  template <typename T> using Base2 = wrapper<typename Base<group<T>>::val>;
  // expected-note-re@+3 {{in instantiation {{.*}} requested here}}
  // expected-note-re@+2 {{in instantiation {{.*}} required here}}
  // expected-note-re@+1 {{candidate template ignored: {{.*}}}}
  template <typename T, typename = Base2<T>> id_type(T &);
};
id_type<1> get() {
  Container::__element_type *ElemPtr;
  // expected-error@+2 {{no viable conversion from returned value of type 'Container::__element_type' (aka '__global element_type') to function return type 'id_type<1>'}}
  // expected-note-re@+1 {{while substituting {{.*}}}}
  return ElemPtr[0];
}
