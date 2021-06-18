// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -std=c++17 -verify %s

// Tests the __builtin_num_fields, __builtin_num_bases, __builtin_field_type,
// and __builtin_base_type intrinsics used by SYCL. These are used to implement
// the is_device_copyable trait in the SYCL runtime library.

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
struct C { struct { int a; float b; }; };
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
struct L { int a; int b : 1; int : 0; int c; };
struct M : virtual A {};

template <typename Ty>
struct N : A, D, Ty {};

void easy() {
  static_assert(__builtin_num_fields(A) == 1, "expected one field");
  static_assert(__builtin_num_fields(B) == 3, "expected three fields");
  static_assert(__builtin_num_fields(E) == 3, "expected three fields");
  static_assert(__builtin_num_fields(H) == 2, "expected two fields");
  static_assert(__builtin_num_fields(I) == 0, "expected no fields");
  static_assert(__builtin_num_fields(J) == 3, "expected three fields");

  static_assert(__builtin_num_bases(A) == 0, "expected no bases");
  static_assert(__builtin_num_bases(H) == 1, "expected one base");
  static_assert(is_same<decltype(__builtin_base_type(H, 0)), B>::value, "expected a B");

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

  // See if we can peer into the anonymous type to get to the two fields
  // declared within it.
  static_assert(__builtin_num_fields(decltype(__builtin_field_type(C, 0))) == 2, "expected two fields");
  static_assert(is_same<decltype(__builtin_field_type(decltype(__builtin_field_type(C, 0)), 0)), int>::value, "expected an int");
  static_assert(is_same<decltype(__builtin_field_type(decltype(__builtin_field_type(C, 0)), 1)), float>::value, "expected an float");

  // struct L has four fields despite containing an anonymous bit-field which
  // is only sort of a field. All four fields are of type int despite some of
  // the fields being bit-fields.
  static_assert(__builtin_num_fields(L) == 4, "expected four fields");
  static_assert(is_same<decltype(__builtin_field_type(L, 0)), int>::value, "expected an int");
  static_assert(is_same<decltype(__builtin_field_type(L, 1)), int>::value, "expected an int");
  static_assert(is_same<decltype(__builtin_field_type(L, 2)), int>::value, "expected an int");
  static_assert(is_same<decltype(__builtin_field_type(L, 3)), int>::value, "expected an int");

  // Virtual bases are still bases.
  static_assert(__builtin_num_bases(M) == 1, "expected one base");
  static_assert(is_same<decltype(__builtin_base_type(M, 0)), A>::value, "expected an A");
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

template <typename Ty>
void templates4() {
  static_assert(__builtin_num_bases(N<Ty>) == 3, "expected three bases");
  static_assert(is_same<decltype(__builtin_base_type(N<Ty>, 0)), A>::value, "expected an A");
  static_assert(is_same<decltype(__builtin_base_type(N<Ty>, 1)), D>::value, "expected a D");
  static_assert(is_same<decltype(__builtin_base_type(N<Ty>, 2)), Ty>::value, "expected a Ty");
}

template <typename Ty, int N>
void templates5() {
  (void)sizeof(__builtin_field_type(Ty, N)); // expected-error {{'__builtin_field_type' requires a non-negative index}}
  (void)sizeof(__builtin_base_type(Ty, N));  // expected-error {{'__builtin_base_type' requires a non-negative index}}
  // expected-note@#neg-instantiation {{in instantiation of function template specialization}}
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

  templates4<B>();
}

template <typename Func>
void invoker(Func F) {
  static_assert(__builtin_num_fields(Func) == 2, "expected two fields");
  static_assert(is_same<decltype(__builtin_field_type(Func, 0)), int>::value, "expected an int");
  static_assert(is_same<decltype(__builtin_field_type(Func, 1)), double>::value, "expected a double");

  // Ensure that the generic lambda is instantiated twice, with different
  // parameter types. This is important for the call to invoker() below which
  // uses a constexpr if that checks the parameter types of the lambda.
  F(1.0);
  F(2);
}

void lambdas() {
  auto l1 = [](){};
  static_assert(__builtin_num_fields(decltype(l1)) == 0, "expected no fields");
  static_assert(__builtin_num_bases(decltype(l1)) == 0, "expected no bases");

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

  // Test that invoking a generic lambda works with the correct number of
  // captures.
  double j;
  invoker([=](auto X) { return i + j + X; });

  // Test it still works even if there's odd stuff happening like using an
  // if constexpr. Note: the order of the implicit captures is important to
  // the static_assert in invoker(). Captures are converted into fields in the
  // order they are implicitly captured.
  invoker([=](auto X) {
    if constexpr (is_same<decltype(X), int>::value) {
      return (double)(X + i);
    }
    return (double)(X + j);
  });
}

struct Z {
  int x, y;
  void lambdas() {
    int z;
    auto l1 = [this](){(void)(x + y); };
    static_assert(__builtin_num_fields(decltype(l1)) == 1, "expected one field");
    static_assert(__builtin_num_bases(decltype(l1)) == 0, "expected no bases");
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
  __builtin_num_bases(struct T); // expected-error {{'__builtin_num_bases' requires a complete type}} \
                                    expected-note {{forward declaration of 'T'}}

  __builtin_num_fields(easy); // expected-error {{unknown type name 'easy'}}
  __builtin_num_fields(decltype(easy)); // expected-error {{'__builtin_num_fields' requires a structure or lambda type}}

  // You can't use __builtin_field_type in an evaluated context because it
  // doesn't return a value that's usable (it's like std::declval).
  int i = __builtin_field_type(A, 0); // expected-error {{'__builtin_field_type' cannot be used in an evaluated context}}
  int j = __builtin_base_type(A, 0); // expected-error {{'__builtin_base_type' cannot be used in an evaluated context}}

  // Using sizeof() to put the expression into an unevaluated context so that
  // the error reporting is focused on the real problem.
  sizeof(__builtin_field_type(A, 1)); // expected-error {{index 1 is greater than the number of fields in type 'A'}}
  sizeof(__builtin_field_type(I, 0)); // expected-error {{index 0 is greater than the number of fields in type 'I'}}
  sizeof(__builtin_base_type(A, 0)); // expected-error {{index 0 is greater than the number of bases in type 'A'}}

  // Test that we reject negative index values, even in templates.
  sizeof(__builtin_field_type(A, -1)); // expected-error {{'__builtin_field_type' requires a non-negative index}}
  sizeof(__builtin_base_type(A, -1)); // expected-error {{'__builtin_base_type' requires a non-negative index}}
  templates5<A, -1>(); // #neg-instantiation
}
