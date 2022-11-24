// RUN: cgeist -O0 %s --function=* -S | FileCheck %s

typedef char char_vec __attribute__((ext_vector_type(3)));
typedef short short_vec __attribute__((ext_vector_type(3)));
typedef int int_vec __attribute__((ext_vector_type(3)));
typedef long long_vec __attribute__((ext_vector_type(3)));
typedef unsigned char unsigned_char_vec __attribute__((ext_vector_type(3)));
typedef unsigned short unsigned_short_vec __attribute__((ext_vector_type(3)));
typedef unsigned int unsigned_int_vec __attribute__((ext_vector_type(3)));
typedef unsigned long unsigned_long_vec __attribute__((ext_vector_type(3)));


// CHECK-LABEL:   func.func @and_i8(
// CHECK-SAME:                      %[[VAL_0:.*]]: i8,
// CHECK-SAME:                      %[[VAL_1:.*]]: i8) -> i8
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.andi %[[VAL_0]], %[[VAL_1]] : i8
// CHECK-NEXT:      return %[[VAL_2]] : i8
// CHECK-NEXT:    }
char and_i8(char a, char b) {
  return a & b;
}


// CHECK-LABEL:   func.func @and_i16(
// CHECK-SAME:                       %[[VAL_0:.*]]: i16,
// CHECK-SAME:                       %[[VAL_1:.*]]: i16) -> i16
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.andi %[[VAL_0]], %[[VAL_1]] : i16
// CHECK-NEXT:      return %[[VAL_2]] : i16
// CHECK-NEXT:    }
short and_i16(short a, short b) {
  return a & b;
}


// CHECK-LABEL:   func.func @and_i32(
// CHECK-SAME:                       %[[VAL_0:.*]]: i32,
// CHECK-SAME:                       %[[VAL_1:.*]]: i32) -> i32
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.andi %[[VAL_0]], %[[VAL_1]] : i32
// CHECK-NEXT:      return %[[VAL_2]] : i32
// CHECK-NEXT:    }
int and_i32(int a, int b) {
  return a & b;
}


// CHECK-LABEL:   func.func @and_i64(
// CHECK-SAME:                       %[[VAL_0:.*]]: i64,
// CHECK-SAME:                       %[[VAL_1:.*]]: i64) -> i64
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.andi %[[VAL_0]], %[[VAL_1]] : i64
// CHECK-NEXT:      return %[[VAL_2]] : i64
// CHECK-NEXT:    }
long and_i64(long a, long b) {
  return a & b;
}


// CHECK-LABEL:   func.func @and_ui8(
// CHECK-SAME:                       %[[VAL_0:.*]]: i8,
// CHECK-SAME:                       %[[VAL_1:.*]]: i8) -> i8
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.andi %[[VAL_0]], %[[VAL_1]] : i8
// CHECK-NEXT:      return %[[VAL_2]] : i8
// CHECK-NEXT:    }
unsigned char and_ui8(unsigned char a, unsigned char b) {
  return a & b;
}


// CHECK-LABEL:   func.func @and_ui16(
// CHECK-SAME:                        %[[VAL_0:.*]]: i16,
// CHECK-SAME:                        %[[VAL_1:.*]]: i16) -> i16
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.andi %[[VAL_0]], %[[VAL_1]] : i16
// CHECK-NEXT:      return %[[VAL_2]] : i16
// CHECK-NEXT:    }
unsigned short and_ui16(unsigned short a, unsigned short b) {
  return a & b;
}


// CHECK-LABEL:   func.func @and_ui32(
// CHECK-SAME:                        %[[VAL_0:.*]]: i32,
// CHECK-SAME:                        %[[VAL_1:.*]]: i32) -> i32
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.andi %[[VAL_0]], %[[VAL_1]] : i32
// CHECK-NEXT:      return %[[VAL_2]] : i32
// CHECK-NEXT:    }
unsigned int and_ui32(unsigned int a, unsigned int b) {
  return a & b;
}


// CHECK-LABEL:   func.func @and_ui64(
// CHECK-SAME:                        %[[VAL_0:.*]]: i64,
// CHECK-SAME:                        %[[VAL_1:.*]]: i64) -> i64
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.andi %[[VAL_0]], %[[VAL_1]] : i64
// CHECK-NEXT:      return %[[VAL_2]] : i64
// CHECK-NEXT:    }
unsigned long and_ui64(unsigned long a, unsigned long b) {
  return a & b;
}


// CHECK-LABEL:   func.func @and_vi8(
// CHECK-SAME:                       %[[VAL_0:.*]]: vector<3xi8>,
// CHECK-SAME:                       %[[VAL_1:.*]]: vector<3xi8>) -> vector<3xi8>
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.andi %[[VAL_0]], %[[VAL_1]] : vector<3xi8>
// CHECK-NEXT:      return %[[VAL_2]] : vector<3xi8>
// CHECK-NEXT:    }
char_vec and_vi8(char_vec a, char_vec b) {
  return a & b;
}


// CHECK-LABEL:   func.func @and_vi16(
// CHECK-SAME:                        %[[VAL_0:.*]]: vector<3xi16>,
// CHECK-SAME:                        %[[VAL_1:.*]]: vector<3xi16>) -> vector<3xi16>
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.andi %[[VAL_0]], %[[VAL_1]] : vector<3xi16>
// CHECK-NEXT:      return %[[VAL_2]] : vector<3xi16>
// CHECK-NEXT:    }
short_vec and_vi16(short_vec a, short_vec b) {
  return a & b;
}


// CHECK-LABEL:   func.func @and_vi32(
// CHECK-SAME:                        %[[VAL_0:.*]]: vector<3xi32>,
// CHECK-SAME:                        %[[VAL_1:.*]]: vector<3xi32>) -> vector<3xi32>
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.andi %[[VAL_0]], %[[VAL_1]] : vector<3xi32>
// CHECK-NEXT:      return %[[VAL_2]] : vector<3xi32>
// CHECK-NEXT:    }
int_vec and_vi32(int_vec a, int_vec b) {
  return a & b;
}


// CHECK-LABEL:   func.func @and_vi64(
// CHECK-SAME:                        %[[VAL_0:.*]]: memref<?xvector<3xi64>>,
// CHECK-SAME:                        %[[VAL_1:.*]]: memref<?xvector<3xi64>>) -> vector<3xi64>
// CHECK-NEXT:      %[[VAL_2:.*]] = affine.load %[[VAL_0]][0] : memref<?xvector<3xi64>>
// CHECK-NEXT:      %[[VAL_3:.*]] = affine.load %[[VAL_1]][0] : memref<?xvector<3xi64>>
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.andi %[[VAL_2]], %[[VAL_3]] : vector<3xi64>
// CHECK-NEXT:      return %[[VAL_4]] : vector<3xi64>
// CHECK-NEXT:    }
long_vec and_vi64(long_vec a, long_vec b) {
  return a & b;
}


// CHECK-LABEL:   func.func @and_vui8(
// CHECK-SAME:                        %[[VAL_0:.*]]: vector<3xi8>,
// CHECK-SAME:                        %[[VAL_1:.*]]: vector<3xi8>) -> vector<3xi8>
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.andi %[[VAL_0]], %[[VAL_1]] : vector<3xi8>
// CHECK-NEXT:      return %[[VAL_2]] : vector<3xi8>
// CHECK-NEXT:    }
unsigned_char_vec and_vui8(unsigned_char_vec a, unsigned_char_vec b) {
  return a & b;
}


// CHECK-LABEL:   func.func @and_vui16(
// CHECK-SAME:                         %[[VAL_0:.*]]: vector<3xi16>,
// CHECK-SAME:                         %[[VAL_1:.*]]: vector<3xi16>) -> vector<3xi16>
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.andi %[[VAL_0]], %[[VAL_1]] : vector<3xi16>
// CHECK-NEXT:      return %[[VAL_2]] : vector<3xi16>
// CHECK-NEXT:    }
unsigned_short_vec and_vui16(unsigned_short_vec a, unsigned_short_vec b) {
  return a & b;
}


// CHECK-LABEL:   func.func @and_vui32(
// CHECK-SAME:                         %[[VAL_0:.*]]: vector<3xi32>,
// CHECK-SAME:                         %[[VAL_1:.*]]: vector<3xi32>) -> vector<3xi32>
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.andi %[[VAL_0]], %[[VAL_1]] : vector<3xi32>
// CHECK-NEXT:      return %[[VAL_2]] : vector<3xi32>
// CHECK-NEXT:    }
unsigned_int_vec and_vui32(unsigned_int_vec a, unsigned_int_vec b) {
  return a & b;
}


// CHECK-LABEL:   func.func @and_vui64(
// CHECK-SAME:                         %[[VAL_0:.*]]: memref<?xvector<3xi64>>,
// CHECK-SAME:                         %[[VAL_1:.*]]: memref<?xvector<3xi64>>) -> vector<3xi64>
// CHECK-NEXT:      %[[VAL_2:.*]] = affine.load %[[VAL_0]][0] : memref<?xvector<3xi64>>
// CHECK-NEXT:      %[[VAL_3:.*]] = affine.load %[[VAL_1]][0] : memref<?xvector<3xi64>>
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.andi %[[VAL_2]], %[[VAL_3]] : vector<3xi64>
// CHECK-NEXT:      return %[[VAL_4]] : vector<3xi64>
// CHECK-NEXT:    }
unsigned_long_vec and_vui64(unsigned_long_vec a, unsigned_long_vec b) {
  return a & b;
}
