// RUN: %check_clang_tidy %s bugprone-narrowing-conversions %t \
// RUN: -- -- -target x86_64-unknown-linux -fsigned-char

namespace floats {

void narrow_constant_floating_point_to_int_not_ok(double d) {
  int i = 0;
  i += 0.5;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: narrowing conversion from constant 'double' to 'int' [bugprone-narrowing-conversions]
  i += 0.5f;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: narrowing conversion from constant 'float' to 'int' [bugprone-narrowing-conversions]
  i *= 0.5f;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: narrowing conversion from constant 'float' to 'int' [bugprone-narrowing-conversions]
  i /= 0.5f;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: narrowing conversion from constant 'float' to 'int' [bugprone-narrowing-conversions]
  i += (double)0.5f;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: narrowing conversion from constant 'double' to 'int' [bugprone-narrowing-conversions]
  i += 2.0;
  i += 2.0f;
}

double operator"" _double(unsigned long long);

float narrow_double_to_float_return() {
  return 0.5;
}

void narrow_double_to_float_not_ok(double d) {
  float f;
  f = d;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions]
  f = 15_double;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions]
  f += d;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions]
  f = narrow_double_to_float_return();
}

float narrow_float16_to_float_return(_Float16 f) {
  return f;
}

_Float16 narrow_float_to_float16_return(float f) {
  return f;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: narrowing conversion from 'float' to '_Float16' [bugprone-narrowing-conversions]
}

void narrow_fp_constants() {
  float f;
  f = 0.5; // [dcl.init.list] 7.2 : in-range fp constant to narrower float is not a narrowing.

  f = __builtin_huge_valf();  // max float is not narrowing.
  f = -__builtin_huge_valf(); // -max float is not narrowing.
  f = __builtin_inff();       // float infinity is not narrowing.
  f = __builtin_nanf("0");    // float NaN is not narrowing.

  f = __builtin_huge_val(); // max double is not within-range of float.
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from constant 'double' to 'float' [bugprone-narrowing-conversions]
  f = -__builtin_huge_val(); // -max double is not within-range of float.
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from constant 'double' to 'float' [bugprone-narrowing-conversions]
  f = __builtin_inf(); // double infinity is not within-range of float.
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from constant 'double' to 'float' [bugprone-narrowing-conversions]
  f = __builtin_nan("0"); // double NaN is not narrowing.
}

double false_positive_const_qualified_cast(bool t) {
  double b = 1.0;
  constexpr double a = __builtin_huge_val();
  // PR49498 The constness difference of 'a' and 'b' results in an implicit cast.
  return t ? b : a;
}

} // namespace floats
