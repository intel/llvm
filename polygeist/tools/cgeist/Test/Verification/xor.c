// RUN: cgeist -O0 %s --function=* -S | FileCheck %s

typedef char char_vec __attribute__((ext_vector_type(3)));
typedef short short_vec __attribute__((ext_vector_type(3)));
typedef int int_vec __attribute__((ext_vector_type(3)));
typedef long long_vec __attribute__((ext_vector_type(3)));
typedef unsigned char unsigned_char_vec __attribute__((ext_vector_type(3)));
typedef unsigned short unsigned_short_vec __attribute__((ext_vector_type(3)));
typedef unsigned int unsigned_int_vec __attribute__((ext_vector_type(3)));
typedef unsigned long unsigned_long_vec __attribute__((ext_vector_type(3)));

// CHECK-LABEL:   func.func @foo(
// CHECK-SAME:                   %[[VAL_0:.*]]: memref<?xi32>,
// CHECK-SAME:                   %[[VAL_1:.*]]: i32)
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : i32
// CHECK-NEXT:      scf.for %[[VAL_6:.*]] = %[[VAL_3]] to %[[VAL_2]] step %[[VAL_4]] {
// CHECK-NEXT:        %[[VAL_7:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_6]]] : memref<?xi32>
// CHECK-NEXT:        %[[VAL_8:.*]] = arith.addi %[[VAL_7]], %[[VAL_5]] : i32
// CHECK-NEXT:        %[[VAL_9:.*]] = arith.xori %[[VAL_1]], %[[VAL_8]] : i32
// CHECK-NEXT:        %[[VAL_10:.*]] = arith.xori %[[VAL_7]], %[[VAL_9]] : i32
// CHECK-NEXT:        memref.store %[[VAL_10]], %[[VAL_0]]{{\[}}%[[VAL_6]]] : memref<?xi32>
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }
void foo(int A[10], int a) {
  for (int i = 0; i < 10; ++i)
    A[i] ^= (a ^ (A[i] + 1));
}


// CHECK-LABEL:   func.func @xor_i8(
// CHECK-SAME:                      %[[VAL_0:.*]]: i8,
// CHECK-SAME:                      %[[VAL_1:.*]]: i8) -> i8
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.xori %[[VAL_0]], %[[VAL_1]] : i8
// CHECK-NEXT:      return %[[VAL_2]] : i8
// CHECK-NEXT:    }
char xor_i8(char a, char b) {
  return a ^ b;
}

// CHECK-LABEL:   func.func @xor_i16(
// CHECK-SAME:                       %[[VAL_0:.*]]: i16,
// CHECK-SAME:                       %[[VAL_1:.*]]: i16) -> i16
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.xori %[[VAL_0]], %[[VAL_1]] : i16
// CHECK-NEXT:      return %[[VAL_2]] : i16
// CHECK-NEXT:    }
short xor_i16(short a, short b) {
  return a ^ b;
}


// CHECK-LABEL:   func.func @xor_i32(
// CHECK-SAME:                       %[[VAL_0:.*]]: i32,
// CHECK-SAME:                       %[[VAL_1:.*]]: i32) -> i32
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.xori %[[VAL_0]], %[[VAL_1]] : i32
// CHECK-NEXT:      return %[[VAL_2]] : i32
// CHECK-NEXT:    }
int xor_i32(int a, int b) {
  return a ^ b;
}


// CHECK-LABEL:   func.func @xor_i64(
// CHECK-SAME:                       %[[VAL_0:.*]]: i64,
// CHECK-SAME:                       %[[VAL_1:.*]]: i64) -> i64
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.xori %[[VAL_0]], %[[VAL_1]] : i64
// CHECK-NEXT:      return %[[VAL_2]] : i64
// CHECK-NEXT:    }
long xor_i64(long a, long b) {
  return a ^ b;
}


// CHECK-LABEL:   func.func @xor_ui8(
// CHECK-SAME:                       %[[VAL_0:.*]]: i8,
// CHECK-SAME:                       %[[VAL_1:.*]]: i8) -> i8
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.xori %[[VAL_0]], %[[VAL_1]] : i8
// CHECK-NEXT:      return %[[VAL_2]] : i8
// CHECK-NEXT:    }
unsigned char xor_ui8(unsigned char a, unsigned char b) {
  return a ^ b;
}


// CHECK-LABEL:   func.func @xor_ui16(
// CHECK-SAME:                        %[[VAL_0:.*]]: i16,
// CHECK-SAME:                        %[[VAL_1:.*]]: i16) -> i16
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.xori %[[VAL_0]], %[[VAL_1]] : i16
// CHECK-NEXT:      return %[[VAL_2]] : i16
// CHECK-NEXT:    }
unsigned short xor_ui16(unsigned short a, unsigned short b) {
  return a ^ b;
}


// CHECK-LABEL:   func.func @xor_ui32(
// CHECK-SAME:                        %[[VAL_0:.*]]: i32,
// CHECK-SAME:                        %[[VAL_1:.*]]: i32) -> i32
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.xori %[[VAL_0]], %[[VAL_1]] : i32
// CHECK-NEXT:      return %[[VAL_2]] : i32
// CHECK-NEXT:    }
unsigned int xor_ui32(unsigned int a, unsigned int b) {
  return a ^ b;
}


// CHECK-LABEL:   func.func @xor_ui64(
// CHECK-SAME:                        %[[VAL_0:.*]]: i64,
// CHECK-SAME:                        %[[VAL_1:.*]]: i64) -> i64
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.xori %[[VAL_0]], %[[VAL_1]] : i64
// CHECK-NEXT:      return %[[VAL_2]] : i64
// CHECK-NEXT:    }
unsigned long xor_ui64(unsigned long a, unsigned long b) {
  return a ^ b;
}


// CHECK-LABEL:   func.func @xor_vi8(
// CHECK-SAME:                       %[[VAL_0:.*]]: vector<3xi8>,
// CHECK-SAME:                       %[[VAL_1:.*]]: vector<3xi8>) -> vector<3xi8>
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.xori %[[VAL_0]], %[[VAL_1]] : vector<3xi8>
// CHECK-NEXT:      return %[[VAL_2]] : vector<3xi8>
// CHECK-NEXT:    }
char_vec xor_vi8(char_vec a, char_vec b) {
  return a ^ b;
}


// CHECK-LABEL:   func.func @xor_vi16(
// CHECK-SAME:                        %[[VAL_0:.*]]: vector<3xi16>,
// CHECK-SAME:                        %[[VAL_1:.*]]: vector<3xi16>) -> vector<3xi16>
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.xori %[[VAL_0]], %[[VAL_1]] : vector<3xi16>
// CHECK-NEXT:      return %[[VAL_2]] : vector<3xi16>
// CHECK-NEXT:    }
short_vec xor_vi16(short_vec a, short_vec b) {
  return a ^ b;
}


// CHECK-LABEL:   func.func @xor_vi32(
// CHECK-SAME:                        %[[VAL_0:.*]]: vector<3xi32>,
// CHECK-SAME:                        %[[VAL_1:.*]]: vector<3xi32>) -> vector<3xi32>
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.xori %[[VAL_0]], %[[VAL_1]] : vector<3xi32>
// CHECK-NEXT:      return %[[VAL_2]] : vector<3xi32>
// CHECK-NEXT:    }
int_vec xor_vi32(int_vec a, int_vec b) {
  return a ^ b;
}


// CHECK-LABEL:   func.func @xor_vi64(
// CHECK-SAME:                        %[[VAL_0:.*]]: memref<?xvector<3xi64>>,
// CHECK-SAME:                        %[[VAL_1:.*]]: memref<?xvector<3xi64>>) -> vector<3xi64>
// CHECK-NEXT:      %[[VAL_2:.*]] = affine.load %[[VAL_0]][0] : memref<?xvector<3xi64>>
// CHECK-NEXT:      %[[VAL_3:.*]] = affine.load %[[VAL_1]][0] : memref<?xvector<3xi64>>
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.xori %[[VAL_2]], %[[VAL_3]] : vector<3xi64>
// CHECK-NEXT:      return %[[VAL_4]] : vector<3xi64>
// CHECK-NEXT:    }
long_vec xor_vi64(long_vec a, long_vec b) {
  return a ^ b;
}


// CHECK-LABEL:   func.func @xor_vui8(
// CHECK-SAME:                        %[[VAL_0:.*]]: vector<3xi8>,
// CHECK-SAME:                        %[[VAL_1:.*]]: vector<3xi8>) -> vector<3xi8>
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.xori %[[VAL_0]], %[[VAL_1]] : vector<3xi8>
// CHECK-NEXT:      return %[[VAL_2]] : vector<3xi8>
// CHECK-NEXT:    }
unsigned_char_vec xor_vui8(unsigned_char_vec a, unsigned_char_vec b) {
  return a ^ b;
}


// CHECK-LABEL:   func.func @xor_vui16(
// CHECK-SAME:                         %[[VAL_0:.*]]: vector<3xi16>,
// CHECK-SAME:                         %[[VAL_1:.*]]: vector<3xi16>) -> vector<3xi16>
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.xori %[[VAL_0]], %[[VAL_1]] : vector<3xi16>
// CHECK-NEXT:      return %[[VAL_2]] : vector<3xi16>
// CHECK-NEXT:    }
unsigned_short_vec xor_vui16(unsigned_short_vec a, unsigned_short_vec b) {
  return a ^ b;
}


// CHECK-LABEL:   func.func @xor_vui32(
// CHECK-SAME:                         %[[VAL_0:.*]]: vector<3xi32>,
// CHECK-SAME:                         %[[VAL_1:.*]]: vector<3xi32>) -> vector<3xi32>
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.xori %[[VAL_0]], %[[VAL_1]] : vector<3xi32>
// CHECK-NEXT:      return %[[VAL_2]] : vector<3xi32>
// CHECK-NEXT:    }
unsigned_int_vec xor_vui32(unsigned_int_vec a, unsigned_int_vec b) {
  return a ^ b;
}


// CHECK-LABEL:   func.func @xor_vui64(
// CHECK-SAME:                         %[[VAL_0:.*]]: memref<?xvector<3xi64>>,
// CHECK-SAME:                         %[[VAL_1:.*]]: memref<?xvector<3xi64>>) -> vector<3xi64>
// CHECK-NEXT:      %[[VAL_2:.*]] = affine.load %[[VAL_0]][0] : memref<?xvector<3xi64>>
// CHECK-NEXT:      %[[VAL_3:.*]] = affine.load %[[VAL_1]][0] : memref<?xvector<3xi64>>
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.xori %[[VAL_2]], %[[VAL_3]] : vector<3xi64>
// CHECK-NEXT:      return %[[VAL_4]] : vector<3xi64>
// CHECK-NEXT:    }
unsigned_long_vec xor_vui64(unsigned_long_vec a, unsigned_long_vec b) {
  return a ^ b;
}
