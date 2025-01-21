// Verify the use of wg_scope is correctly diagnosed.
// RUN: %clang_cc1 -fsycl-is-device -verify %s

class [[__sycl_detail__::wg_scope]] G1 {};
class [[__sycl_detail__::wg_scope]] G2 {
  G2() = default;
  G2(int i) : i(i) {}
  int i;
};

class [[__sycl_detail__::wg_scope]] G3 {
  ~G3() = default;
};

class [[__sycl_detail__::wg_scope]] B4 { // expected-error {{SYCL work group scope only applies to class with a trivial default constructor}}
  B4() {}
};

class [[__sycl_detail__::wg_scope]] B5 { // expected-error {{SYCL work group scope only applies to class with a trivial destructor}}
  ~B5() {}
};

class [[__sycl_detail__::wg_scope]] B6 { // expected-error {{SYCL work group scope only applies to class with a trivial default constructor}}
  B6() {}
  ~B6() {}
};

template <typename T> class [[__sycl_detail__::wg_scope]] B7 { // #B7
public:
  T obj;
};

struct Valid {};
struct InvalidCtor {
  InvalidCtor() {}
};
struct InvalidDtor {
  ~InvalidDtor() {}
};
struct InvalidCDtor {
  InvalidCDtor() {}
  ~InvalidCDtor() {}
};

B7<Valid> b7;
// expected-error@#B7 {{SYCL work group scope only applies to class with a trivial default constructor}}
// expected-note@+1 {{in instantiation of template class 'B7<InvalidCtor>' requested here}}
B7<InvalidCtor> b9;
// expected-error@#B7 {{SYCL work group scope only applies to class with a trivial destructor}}
// expected-note@+1 {{in instantiation of template class 'B7<InvalidDtor>' requested here}}
B7<InvalidDtor> b10;
// expected-error@#B7 {{SYCL work group scope only applies to class with a trivial default constructor}}
// expected-note@+1 {{in instantiation of template class 'B7<InvalidCDtor>' requested here}}
B7<InvalidCDtor> b11;

template <typename T> class [[__sycl_detail__::wg_scope]] B12 { // #B12
public:
  B12() = default;
  ~B12() = default;
  T obj;
};

B12<Valid> b12;
// expected-error@#B12 {{SYCL work group scope only applies to class with a trivial default constructor}}
// expected-note@+1 {{in instantiation of template class 'B12<InvalidCtor>' requested here}}
B12<InvalidCtor> b13;

class B14 {
  G1 field; // expected-error {{non-static data member is of a type with a SYCL work group scope attribute applied to it}}
};

template <typename T> class B15 {
  T field; // #B15-field
};

// expected-error@#B15-field {{non-static data member is of a type with a SYCL work group scope attribute applied to it}}
// expected-note@+1 {{in instantiation of template class 'B15<G1>' requested here}}
B15<G1> b15;

G1 g16;
static G1 g17;

struct Wrap {
  static G1 g18;
};

__attribute__((sycl_device)) void ref_func() {
  G1 g19;
  static G1 g20;

  (void)g16;
  (void)g17;
  (void)Wrap::g18;
}
