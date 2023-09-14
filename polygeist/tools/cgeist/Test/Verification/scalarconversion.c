// RUN: cgeist %s --function=* -S -O0 -w | FileCheck %s

// CHECK-LABEL:   func.func @unsigned2float(
// CHECK-SAME:                              %[[VAL_0:.*]]: i32) -> f32
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.uitofp %[[VAL_0]] : i32 to f32
// CHECK-NEXT:      return %[[VAL_1]] : f32
// CHECK-NEXT:    }
float unsigned2float(unsigned i) { return (float)i; }

// CHECK-LABEL:   func.func @char2double(
// CHECK-SAME:                           %[[VAL_0:.*]]: i8) -> f64
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.sitofp %[[VAL_0]] : i8 to f64
// CHECK-NEXT:      return %[[VAL_1]] : f64
// CHECK-NEXT:    }
double char2double(char i) { return (double)i; }

// CHECK-LABEL:   func.func @float2unsigned(
// CHECK-SAME:                              %[[VAL_0:.*]]: f32) -> i32
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.fptoui %[[VAL_0]] : f32 to i32
// CHECK-NEXT:      return %[[VAL_1]] : i32
// CHECK-NEXT:    }
unsigned float2unsigned(float i) { return (unsigned)i; }

// CHECK-LABEL:   func.func @double2longlong(
// CHECK-SAME:                               %[[VAL_0:.*]]: f64) -> i64
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.fptosi %[[VAL_0]] : f64 to i64
// CHECK-NEXT:      return %[[VAL_1]] : i64
// CHECK-NEXT:    }
long long double2longlong(double i) { return (long long)i; }

// CHECK-LABEL:   func.func @float2double(
// CHECK-SAME:                            %[[VAL_0:.*]]: f32) -> f64
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.extf %[[VAL_0]] : f32 to f64
// CHECK-NEXT:      return %[[VAL_1]] : f64
// CHECK-NEXT:    }
double float2double(float i) { return (double)i; }

// CHECK-LABEL:   func.func @double2float(
// CHECK-SAME:                            %[[VAL_0:.*]]: f64) -> f32
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.truncf %[[VAL_0]] : f64 to f32
// CHECK-NEXT:      return %[[VAL_1]] : f32
// CHECK-NEXT:    }
float double2float(double i) { return (float)i; }

// CHECK-LABEL:   func.func @unsigned2int(
// CHECK-SAME:                            %[[VAL_0:.*]]: i32) -> i32
// CHECK-NEXT:      return %[[VAL_0]] : i32
// CHECK-NEXT:    }
int unsigned2int(unsigned i) { return (int)i; }

// CHECK-LABEL:   func.func @long2int(
// CHECK-SAME:                        %[[VAL_0:.*]]: i64) -> i32
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.trunci %[[VAL_0]] : i64 to i32
// CHECK-NEXT:      return %[[VAL_1]] : i32
// CHECK-NEXT:    }
int long2int(long i) { return (int)i; }

// CHECK-LABEL:   func.func @int2long(
// CHECK-SAME:                        %[[VAL_0:.*]]: i32) -> i64
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.extsi %[[VAL_0]] : i32 to i64
// CHECK-NEXT:      return %[[VAL_1]] : i64
// CHECK-NEXT:    }
long int2long(int i) { return (long)i; }

// CHECK-LABEL:   func.func @unsigned2long(
// CHECK-SAME:                             %[[VAL_0:.*]]: i32) -> i64
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.extui %[[VAL_0]] : i32 to i64
// CHECK-NEXT:      return %[[VAL_1]] : i64
// CHECK-NEXT:    }
long unsigned2long(unsigned i) { return (long)i; }
