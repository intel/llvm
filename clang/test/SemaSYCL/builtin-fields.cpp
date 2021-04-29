// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s

template<class T, T v>
struct integral_constant {
    static constexpr T value = v;
    using value_type = T;
    using type = integral_constant;
    constexpr operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; }
};

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template<class T, class U>
struct is_same : false_type {};

template<class T>
struct is_same<T, T> : true_type {};

struct A { int i; };
struct B { float f; int i; A a; };
struct C { struct { int a, b; }; };
struct D { union { int a; float f; }; };
union E { int i; float f; double d; };
template <typename Ty> struct F { Ty a; int b; }; // expected-error {{field has incomplete type 'S'}}
template <int N> struct G { int i = N; };

// This class has two fields and one base. We don't treat base classes
// as a field so that the SYCL library can generate more sensible
// diagnostics.
struct H : B {
  double d;
  const char *str;
};

struct I {};
struct J { int &derp; const int &herp; int &&berp; };
struct K { int a[10]; int b[]; };

void easy() {
  static_assert(__builtin_num_fields(A) == 1, "expected one field");
  static_assert(__builtin_num_fields(B) == 3, "expected three fields");
  static_assert(__builtin_num_fields(E) == 3, "expected three fields");
  static_assert(__builtin_num_fields(H) == 2, "expected two fields");
  static_assert(__builtin_num_fields(I) == 0, "expected no fields");
  static_assert(__builtin_num_fields(J) == 3, "expected three fields");

  static_assert(is_same<decltype(__builtin_field_type(A, 0)), int>::value, "expected an int");

  static_assert(is_same<decltype(__builtin_field_type(B, 0)), float>::value, "expected a float");
  static_assert(is_same<decltype(__builtin_field_type(B, 1)), int>::value, "expected an int");
  static_assert(is_same<decltype(__builtin_field_type(B, 2)), A>::value, "expected an A");

  static_assert(is_same<decltype(__builtin_field_type(E, 0)), int>::value, "expected an int");
  static_assert(is_same<decltype(__builtin_field_type(E, 1)), float>::value, "expected a float");
  static_assert(is_same<decltype(__builtin_field_type(E, 2)), double>::value, "expected a double");

  static_assert(is_same<decltype(__builtin_field_type(H, 0)), double>::value, "expected a double");
  static_assert(is_same<decltype(__builtin_field_type(H, 1)), const char *>::value, "expected a const char *");

  static_assert(is_same<decltype(__builtin_field_type(J, 0)), int&>::value, "expected an int&");
  static_assert(is_same<decltype(__builtin_field_type(J, 1)), const int&>::value, "expected a const int&");
  static_assert(is_same<decltype(__builtin_field_type(J, 2)), int&&>::value, "expected an int&&");

  static_assert(is_same<decltype(__builtin_field_type(K, 0)), int[10]>::value, "expected an int[10]");
  static_assert(is_same<decltype(__builtin_field_type(K, 1)), int[]>::value, "expected an int[]");
}

void odd() {
  // struct C only has one field because anonymous structures are a GNU
  // extension that model the inner struct as a single field containing two
  // fields.
  static_assert(__builtin_num_fields(C) == 1, "expected one field");
  static_assert(__builtin_num_fields(D) == 1, "expected one field");
}

template <typename Ty>
void templates1() {
  static_assert(__builtin_num_fields(F<Ty>) == 2, "expected two fields"); // expected-note {{in instantiation of template class 'F<S>' requested here}}
  static_assert(__builtin_num_fields(G<2>) == 1, "expected one field");
}

template <typename Ty>
void templates2() {
  static_assert(__builtin_num_fields(Ty) == 3, "expected three fields");
}

template <typename FieldTy>
void templates3() {
  static_assert(__builtin_num_fields(F<FieldTy>) == 2, "expected two fields");
  static_assert(is_same<decltype(__builtin_field_type(F<FieldTy>, 0)), FieldTy>::value, "expected a FieldTy");
  static_assert(is_same<decltype(__builtin_field_type(F<FieldTy>, 1)), int>::value, "expected an int");
}

void instantiate() {
  templates1<int>();
  templates1<struct S>(); // expected-note {{in instantiation of function template specialization 'templates1<S>' requested here}} \
                             expected-note {{forward declaration of 'S'}}
  templates2<B>();
  templates2<E>();

  templates3<int>();
  templates3<float>();
  templates3<A>();
}

void lambdas() {
  auto l1 = [](){};
  static_assert(__builtin_num_fields(decltype(l1)) == 0, "expected no fields");

  int i;
  auto l2 = [i](){};
  static_assert(__builtin_num_fields(decltype(l2)) == 1, "expected one field");
  static_assert(is_same<decltype(__builtin_field_type(decltype(l2), 0)), int>::value, "expected an int");

  auto l3 = [&](){};
  static_assert(__builtin_num_fields(decltype(l3)) == 0, "expected no fields");

  auto l4 = [&]() { (void)i; };
  static_assert(__builtin_num_fields(decltype(l4)) == 1, "expected one field");
  static_assert(is_same<decltype(__builtin_field_type(decltype(l4), 0)), int&>::value, "expected an int&");

  auto l5 = [=]() { (void)i; };
  static_assert(__builtin_num_fields(decltype(l5)) == 1, "expected one field");
  static_assert(is_same<decltype(__builtin_field_type(decltype(l5), 0)), int>::value, "expected an int");
}

struct Z {
  int x, y;
  void lambdas() {
    int z;
    auto l1 = [this](){(void)(x + y); };
    static_assert(__builtin_num_fields(decltype(l1)) == 1, "expected one field");
    static_assert(is_same<decltype(__builtin_field_type(decltype(l1), 0)), Z*>::value, "expected a Z*");
    auto l2 = [&](){(void)(x + y); };
    static_assert(__builtin_num_fields(decltype(l2)) == 1, "expected one field");
    static_assert(is_same<decltype(__builtin_field_type(decltype(l2), 0)), Z*>::value, "expected a Z*");
    auto l3 = [=](){(void)(x + y + z); };
    static_assert(__builtin_num_fields(decltype(l3)) == 2, "expected two fields");
    static_assert(is_same<decltype(__builtin_field_type(decltype(l3), 0)), Z*>::value, "expected a Z*");
    static_assert(is_same<decltype(__builtin_field_type(decltype(l3), 1)), int>::value, "expected an int");
  }
};

void errors() {
  __builtin_num_fields(struct S); // expected-error {{'__builtin_num_fields' requires a complete type}} \
                                     expected-note {{forward declaration of 'S'}}
  __builtin_num_fields(easy); // expected-error {{unknown type name 'easy'}}
  __builtin_num_fields(decltype(easy)); // expected-error {{'__builtin_num_fields' requires a structure or lambda type}}

  __builtin_field_type(A, 1); // expected-error {{index 1 is greater than the number of fields in type 'A'}}
  __builtin_field_type(I, 0); // expected-error {{index 0 is greater than the number of fields in type 'I'}}
}
