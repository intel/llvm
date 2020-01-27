// RUN: %clang_cc1 -fsyntax-only -verify %s

template<int Bounds>
struct HasExtInt {
  _ExtInt(Bounds) b;
  unsigned _ExtInt(Bounds) b2;
};

// Delcaring variables:
_ExtInt(33) Declarations(_ExtInt(48) &Param) { // Useable in params and returns.
  short _ExtInt(43) a; // expected-error {{'short _ExtInt' is invalid}}
  _ExtInt(43) long b;  // expected-error {{'long _ExtInt' is invalid}}

  // These should all be fine:
  const _ExtInt(5) c = 3;
  const unsigned _ExtInt(5) d; // expected-error {{default initialization of an object of const type 'const unsigned _ExtInt(5)'}}
  unsigned _ExtInt(5) e = 5;
  _ExtInt(5) unsigned f;

  _ExtInt(-3) g; // expected-error{{signed _ExtInt must have a size of at least 2}}
  _ExtInt(0) h; // expected-error{{signed _ExtInt must have a size of at least 2}}
  _ExtInt(1) i; // expected-error{{signed _ExtInt must have a size of at least 2}}
  _ExtInt(2) j;;
  unsigned _ExtInt(0) k;// expected-error{{unsigned _ExtInt must have a size of at least 1}}
  unsigned _ExtInt(1) l;
  signed _ExtInt(1) m; // expected-error{{signed _ExtInt must have a size of at least 2}}

  constexpr _ExtInt(6) n = 33; // expected-warning{{implicit conversion from 'int' to 'const _ExtInt(6)' changes value from 33 to -31}}
  constexpr _ExtInt(7) o = 33;

  // Check LLVM imposed max size.
  _ExtInt(0xFFFFFFFFFF) p; // expected-error {{signed _ExtInt of sizes greater than 16777215 not supported}}
  unsigned _ExtInt(0xFFFFFFFFFF) q; // expected-error {{unsigned _ExtInt of sizes greater than 16777215 not supported}}

// Ensure template params are instantiated correctly.
  // expected-error@5{{signed _ExtInt must have a size of at least 2}}
  // expected-error@6{{unsigned _ExtInt must have a size of at least 1}}
  // expected-note@+1{{in instantiation of template class }}
  HasExtInt<-1> r;
  // expected-error@5{{signed _ExtInt must have a size of at least 2}}
  // expected-error@6{{unsigned _ExtInt must have a size of at least 1}}
  // expected-note@+1{{in instantiation of template class }}
  HasExtInt<0> s;
  // expected-error@5{{signed _ExtInt must have a size of at least 2}}
  // expected-note@+1{{in instantiation of template class }}
  HasExtInt<1> t;
  HasExtInt<2> u;

  _ExtInt(-3.0) v; // expected-error {{integral constant expression must have integral or unscoped enumeration type, not 'double'}}
  _ExtInt(3.0) x; // expected-error {{integral constant expression must have integral or unscoped enumeration type, not 'double'}}

  return 0;
}

template <_ExtInt(5) I> // TODO: CodeGen test for mangling.
struct ExtIntTemplParam {
  static constexpr _ExtInt(5) Var = I;
};

template<typename T>
void deduced_whole_type(T){}
template<int I>
void deduced_bound(_ExtInt(I)){}

// Ensure ext-int can be used in template places.
void Templates() {
  ExtIntTemplParam<13> a;
  constexpr _ExtInt(3) b = 1;
  ExtIntTemplParam<b> c;
  constexpr _ExtInt(9) d = 1;
  ExtIntTemplParam<b> e;

  deduced_whole_type(b);
  deduced_bound(b);
}

template <typename T, typename U>
struct is_same {
  static constexpr bool value = false;
};
template <typename T>
struct is_same<T,T> {
  static constexpr bool value = true;
};

// Test result types of Unary/Bitwise/Binary Operations:
void Ops() {
  _ExtInt(43) x43_s = 1, y43_s = 1;
  _ExtInt(sizeof(int) * 8) x32_s = 1, y32_s = 1;
  unsigned _ExtInt(sizeof(unsigned) * 8) x32_u = 1, y32_u = 1;
  _ExtInt(4) x4_s = 1, y4_s = 1;
  unsigned _ExtInt(43) x43_u = 1, y43_u = 1;
  unsigned _ExtInt(4) x4_u = 1, y4_u = 1;
  int x_int = 1, y_int = 1;
  unsigned x_uint = 1, y_uint = 1;
  bool b;

  // Same size/sign ops don't change type.
  static_assert(is_same<decltype(x43_s + y43_s), _ExtInt(43)>::value,"");
  static_assert(is_same<decltype(x4_s - y4_s), _ExtInt(4)>::value,"");
  static_assert(is_same<decltype(x43_u * y43_u), unsigned _ExtInt(43)>::value,"");
  static_assert(is_same<decltype(x4_u / y4_u), unsigned _ExtInt(4)>::value,"");

  // Same size/diff sign goes to unsigned.
  static_assert(is_same<decltype(x43_u + y43_s), unsigned _ExtInt(43)>::value,"");
  static_assert(is_same<decltype(x4_s - y4_u), unsigned _ExtInt(4)>::value,"");
  static_assert(is_same<decltype(x43_s * y43_u), unsigned _ExtInt(43)>::value,"");
  static_assert(is_same<decltype(x4_u / y4_s), unsigned _ExtInt(4)>::value,"");

  // Different size prefers largest.
  static_assert(is_same<decltype(x43_s + y4_s), _ExtInt(43)>::value,"");
  static_assert(is_same<decltype(x43_s - y4_u), _ExtInt(43)>::value,"");
  static_assert(is_same<decltype(x43_u * y4_u), unsigned _ExtInt(43)>::value,"");
  static_assert(is_same<decltype(x4_u / y43_u), unsigned _ExtInt(43)>::value,"");

  // When mixed with standard types, largest prefered, then standard types 
  // > ext_int.
  static_assert(is_same<decltype(x43_s + x_int), _ExtInt(43)>::value,"");
  static_assert(is_same<decltype(x43_u - x_int), unsigned _ExtInt(43)>::value,"");
  static_assert(is_same<decltype(x32_s * x_int), int>::value,"");
  static_assert(is_same<decltype(x32_u / x_int), unsigned int>::value,"");
  static_assert(is_same<decltype(x32_s * x_uint), unsigned int>::value,"");
  static_assert(is_same<decltype(x32_u / x_uint), unsigned int>::value,"");

  static_assert(is_same<decltype(x4_s + x_int), int>::value,"");
  static_assert(is_same<decltype(x4_u - x_int), int>::value,"");

  // bool was promoted to int, so rules apply same as above.
  static_assert(is_same<decltype(x4_s + b), int>::value,"");
  static_assert(is_same<decltype(x4_u - b), int>::value,"");
  static_assert(is_same<decltype(x43_s + b), _ExtInt(43)>::value,"");
  static_assert(is_same<decltype(x43_u - b), unsigned _ExtInt(43)>::value,"");

  // More spot checking with bitwise ops.
  static_assert(is_same<decltype(x43_s % y4_u), _ExtInt(43)>::value,"");
  static_assert(is_same<decltype(x43_u % y4_s), unsigned _ExtInt(43)>::value,"");
  static_assert(is_same<decltype(x4_s | y43_u), unsigned _ExtInt(43)>::value,"");
  static_assert(is_same<decltype(x4_u | y43_s), _ExtInt(43)>::value,"");
  static_assert(is_same<decltype(x4_s >> 1), _ExtInt(4)>::value,"");
  static_assert(is_same<decltype(x4_u << 1), unsigned _ExtInt(4)>::value,"");

  // Unary ops shouldn't go through integer promotions.
  static_assert(is_same<decltype(~x43_s), _ExtInt(43)>::value,"");
  static_assert(is_same<decltype(~x4_s), _ExtInt(4)>::value,"");
  static_assert(is_same<decltype(+x43_s), _ExtInt(43)>::value,"");
  static_assert(is_same<decltype(+x4_s), _ExtInt(4)>::value,"");
  static_assert(is_same<decltype(-x43_u), unsigned _ExtInt(43)>::value,"");
  static_assert(is_same<decltype(-x4_u), unsigned _ExtInt(4)>::value,"");
  // expected-warning@+1{{expression with side effects has no effect in an unevaluated context}}
  static_assert(is_same<decltype(++x43_s), _ExtInt(43)&>::value,"");
  // expected-warning@+1{{expression with side effects has no effect in an unevaluated context}}
  static_assert(is_same<decltype(--x4_s), _ExtInt(4)&>::value,"");
  // expected-warning@+1{{expression with side effects has no effect in an unevaluated context}}
  static_assert(is_same<decltype(x43_s--), _ExtInt(43)>::value,"");
  // expected-warning@+1{{expression with side effects has no effect in an unevaluated context}}
  static_assert(is_same<decltype(x4_s++), _ExtInt(4)>::value,"");

  static_assert(sizeof(x43_s) == 8, "");
  static_assert(sizeof(x4_s) == 1, "");

  static_assert(sizeof(_ExtInt(3340)) == 424, ""); // 424 * 8 == 3392.
  static_assert(sizeof(_ExtInt(1049)) == 136, ""); // 136  *  8 == 1088.

  // compassign works too.
  // expected-warning@+1{{expression with side effects has no effect in an unevaluated context}}
  static_assert(is_same<decltype(x43_s += 33), _ExtInt(43)&>::value, "");

  // Comparisons work the same as ints.
  static_assert(is_same<decltype(x43_s > 33), bool>::value, "");
  static_assert(is_same<decltype(x4_s > 33), bool>::value, "");
}

// Useable as an underlying type.
enum AsEnumUnderlyingType : _ExtInt(33) {
};

void overloaded(int);
void overloaded(_ExtInt(32));
void overloaded(_ExtInt(33));
void overloaded(short);
//expected-note@+1{{candidate function}}
void overloaded2(_ExtInt(32));
//expected-note@+1{{candidate function}}
void overloaded2(_ExtInt(33));
//expected-note@+1{{candidate function}}
void overloaded2(short);

void overload_use() {
  int i;
  _ExtInt(32) i32;
  _ExtInt(33) i33;
  short s;

  // All of these get their corresponding exact matches.
  overloaded(i);
  overloaded(i32);
  overloaded(i33);
  overloaded(s);

  overloaded2(i); // expected-error{{call to 'overloaded2' is ambiguous}}

  overloaded2(i32);

  overloaded2(s);
}

// no errors expected, this should 'just work'.
struct UsedAsBitField {
  _ExtInt(3) F : 3;
  _ExtInt(3) G : 3;
  _ExtInt(3) H : 3;
};

// Note, the extra closing brackets confuses the parser, so this test should likely be last.
// expected-error@+5{{expected ')'}}
// expected-note@+4{{to match this '('}}
// expected-error@+3{{expected unqualified-id}}
// expected-error@+2{{extraneous closing brace}}
// expected-error@+1{{C++ requires a type specifier for all declarations}}
_ExtInt(32} a;
// expected-error@+2{{expected expression}}
// expected-error@+1{{C++ requires a type specifier for all declarations}}
_ExtInt(32* ) b;
// expected-error@+3{{expected '('}}
// expected-error@+2{{expected unqualified-id}}
// expected-error@+1{{C++ requires a type specifier for all declarations}}
_ExtInt{32} c;

