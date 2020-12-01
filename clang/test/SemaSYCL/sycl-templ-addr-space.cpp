// RUN: %clang_cc1 -fsycl -fsycl-is-device -triple spir64 -verify %s

// Test that an error is issued instead of an assertion for this test case.

template <int b>
struct c {
  static constexpr int d = b;
};
template <typename ab>
ab &&l(int);
template <typename ab>
auto ac(l<ab>(0));
template <typename n, typename s>
class o {
  template <typename ad, decltype(ac<ad>)>
  // expected-note@+1 {{possible target for call}}
  static c<true> p();
  // expected-error@+1 {{reference to overloaded function could not be resolved; did you mean to call it?}}
  decltype(p<n, s>) i;
};
// expected-note-re@+2 {{in instantiation {{.*}} requested here}}
template <typename n>
struct q : o<n, int> {
};
template <typename ab>
struct r {
  typedef ab i;
};
class ah {
};
struct ai {
  using i = __attribute__((opencl_global)) ah;
};
template <bool, class ak, class> using am = ak;
// expected-note-re@+1 2{{candidate constructor {{.*}} not viable: {{.*}}}}
template <int> class v {
  template <class...> struct as;
  // expected-note-re@+1 {{in instantiation {{.*}} requested here}}
  template <class at> struct as<at> : am<at::d, as<>, at> {};
  // expected-note-re@+1 {{in instantiation {{.*}} requested here}}
  template <typename au> using av = r<typename as<q<au>>::d>;
  // expected-note-re@+3 {{in instantiation {{.*}} requested here}}
  // expected-note-re@+2 {{in instantiation {{.*}} required here}}
  // expected-note-re@+1 {{candidate template ignored: {{.*}}}}
  template <typename au, typename = av<au>> v(au &);
};
struct x {
  v<1> ay() {
    ai::i *pi;
    // expected-error@+2 {{no viable conversion from returned value of type 'ai::i' (aka '__global ah') to function return type 'v<1>'}}
    // expected-note-re@+1 {{while substituting {{.*}}}}
    return pi[0];
  }
};
