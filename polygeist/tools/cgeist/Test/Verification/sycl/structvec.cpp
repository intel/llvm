// RUN: clang++  -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s

#include <initializer_list>
#include <sycl/sycl.hpp>

struct structvec {
  using char2 = char __attribute__((ext_vector_type(2)));
  char2 v;

  structvec(std::initializer_list<char> l) {
    for (unsigned I = 0; I < 2; ++I) {
      v[I] = *(l.begin() + I) ? -1 : 0;
    }
  }
};

// CHECK-LABEL:     func.func @_Z10test_store9structvecic(
// CHECK-SAME:          %[[VAL_151:.*]]: !llvm.ptr {llvm.align = 2 : i64, llvm.byval = !llvm.struct<(vector<2xi8>)>, llvm.noundef}, 
// CHECK-SAME:          %[[VAL_152:.*]]: i32 {llvm.noundef}, %[[VAL_153:.*]]: i8 {llvm.noundef, llvm.signext}) -> !llvm.struct<(vector<2xi8>)>
// CHECK-NEXT:        %[[VAL_154:.*]] = arith.constant 1 : i64
// CHECK-NEXT:        %[[VAL_155:.*]] = llvm.alloca %[[VAL_154]] x !llvm.struct<(vector<2xi8>)> : (i64) -> !llvm.ptr
// CHECK-NEXT:        %[[VAL_156:.*]] = llvm.getelementptr inbounds %[[VAL_151]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(vector<2xi8>)>
// CHECK-NEXT:        %[[VAL_157:.*]] = llvm.load %[[VAL_156]] : !llvm.ptr -> vector<2xi8>
// CHECK-NEXT:        %[[VAL_158:.*]] = vector.insertelement %[[VAL_153]], %[[VAL_157]]{{\[}}%[[VAL_152]] : i32] : vector<2xi8>
// CHECK-NEXT:        llvm.store %[[VAL_158]], %[[VAL_156]] : vector<2xi8>, !llvm.ptr
// CHECK-NEXT:        %[[VAL_159:.*]] = llvm.addrspacecast %[[VAL_155]] : !llvm.ptr to !llvm.ptr<4>
// CHECK-NEXT:        %[[VAL_160:.*]] = llvm.addrspacecast %[[VAL_151]] : !llvm.ptr to !llvm.ptr<4>
// CHECK-NEXT:        call @_ZN9structvecC1EOS_(%[[VAL_159]], %[[VAL_160]]) : (!llvm.ptr<4>, !llvm.ptr<4>) -> ()
// CHECK-NEXT:        %[[VAL_161:.*]] = llvm.load %[[VAL_155]] : !llvm.ptr -> !llvm.struct<(vector<2xi8>)>
// CHECK-NEXT:        return %[[VAL_161]] : !llvm.struct<(vector<2xi8>)>
// CHECK-NEXT:      }


// CHECK-LABEL:     func.func @_ZN9structvecC1EOS_(
// CHECK-SAME:          %[[VAL_162:.*]]: !llvm.ptr<4> {llvm.align = 2 : i64, llvm.dereferenceable_or_null = 2 : i64, llvm.noundef}
// CHECK-SAME:          %[[VAL_163:.*]]: !llvm.ptr<4> {llvm.align = 2 : i64, llvm.dereferenceable = 2 : i64, llvm.noundef})
// CHECK-NEXT:        %[[VAL_164:.*]] = llvm.getelementptr inbounds %[[VAL_163]][0, 0] : (!llvm.ptr<4>) -> !llvm.ptr<4>, !llvm.struct<(vector<2xi8>)>
// CHECK-NEXT:        %[[VAL_165:.*]] = llvm.load %[[VAL_164]] : !llvm.ptr<4> -> vector<2xi8>
// CHECK-NEXT:        %[[VAL_166:.*]] = llvm.getelementptr inbounds %[[VAL_162]][0, 0] : (!llvm.ptr<4>) -> !llvm.ptr<4>, !llvm.struct<(vector<2xi8>)>
// CHECK-NEXT:        llvm.store %[[VAL_165]], %[[VAL_166]] : vector<2xi8>, !llvm.ptr<4>
// CHECK-NEXT:        return
// CHECK-NEXT:      }

SYCL_EXTERNAL structvec test_store(structvec sv, int idx, char el) {
  sv.v[idx] = el;
  return sv;
}

// CHECK-LABEL:     func.func @_Z9test_initv() -> !llvm.struct<(vector<2xi8>)>
// CHECK-DAG:         %[[VAL_167:.*]] = arith.constant 2 : i64
// CHECK-DAG:         %[[VAL_168:.*]] = arith.constant 1 : i8
// CHECK-DAG:         %[[VAL_169:.*]] = arith.constant 0 : i8
// CHECK-DAG:         %[[VAL_170:.*]] = arith.constant 1 : i64
// CHECK-DAG:         %[[VAL_171:.*]] = llvm.alloca %[[VAL_170]] x !llvm.struct<(vector<2xi8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:         %[[VAL_172:.*]] = llvm.alloca %[[VAL_170]] x !llvm.struct<(vector<2xi8>)> : (i64) -> !llvm.ptr
// CHECK-DAG:         %[[VAL_173:.*]] = llvm.alloca %[[VAL_170]] x !llvm.struct<(memref<?xi8, 4>, i64)> : (i64) -> !llvm.ptr
// CHECK-DAG:         %[[VAL_174:.*]] = llvm.alloca %[[VAL_170]] x !llvm.struct<(memref<?xi8, 4>, i64)> : (i64) -> !llvm.ptr
// CHECK-DAG:         %[[VAL_175:.*]] = llvm.alloca %[[VAL_170]] x !llvm.array<2 x i8> : (i64) -> !llvm.ptr
// CHECK-DAG:         %[[VAL_176:.*]] = llvm.alloca %[[VAL_170]] x !llvm.array<2 x i8> : (i64) -> !llvm.ptr
// CHECK-DAG:         %[[VAL_177:.*]] = llvm.alloca %[[VAL_170]] x !llvm.struct<(vector<2xi8>)> : (i64) -> !llvm.ptr
// CHECK-NEXT:        %[[VAL_178:.*]] = llvm.getelementptr inbounds %[[VAL_176]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i8>
// CHECK-NEXT:        llvm.store %[[VAL_169]], %[[VAL_178]] : i8, !llvm.ptr
// CHECK-NEXT:        %[[VAL_179:.*]] = llvm.getelementptr inbounds %[[VAL_176]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i8>
// CHECK-NEXT:        llvm.store %[[VAL_168]], %[[VAL_179]] : i8, !llvm.ptr
// CHECK-NEXT:        %[[VAL_180:.*]] = llvm.addrspacecast %[[VAL_175]] : !llvm.ptr to !llvm.ptr<4>
// CHECK-NEXT:        %[[VAL_181:.*]] = llvm.load %[[VAL_176]] : !llvm.ptr -> !llvm.array<2 x i8>
// CHECK-NEXT:        llvm.store %[[VAL_181]], %[[VAL_180]] : !llvm.array<2 x i8>, !llvm.ptr<4>
// CHECK-NEXT:        %[[VAL_182:.*]] = "polygeist.pointer2memref"(%[[VAL_180]]) : (!llvm.ptr<4>) -> memref<?xi8, 4 : i32>
// CHECK-NEXT:        %[[VAL_183:.*]] = llvm.getelementptr inbounds %[[VAL_174]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8, 4>, i64)>
// CHECK-NEXT:        llvm.store %[[VAL_182]], %[[VAL_183]] : memref<?xi8, 4 : i32>, !llvm.ptr
// CHECK-NEXT:        %[[VAL_184:.*]] = llvm.getelementptr inbounds %[[VAL_174]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi8, 4>, i64)>
// CHECK-NEXT:        llvm.store %[[VAL_167]], %[[VAL_184]] : i64, !llvm.ptr
// CHECK-NEXT:        %[[VAL_185:.*]] = llvm.load %[[VAL_174]] : !llvm.ptr -> !llvm.struct<(memref<?xi8, 4>, i64)>
// CHECK-NEXT:        %[[VAL_186:.*]] = llvm.addrspacecast %[[VAL_177]] : !llvm.ptr to !llvm.ptr<4>
// CHECK-NEXT:        llvm.store %[[VAL_185]], %[[VAL_173]] : !llvm.struct<(memref<?xi8, 4>, i64)>, !llvm.ptr
// CHECK-NEXT:        call @_ZN9structvecC1ESt16initializer_listIcE(%[[VAL_186]], %[[VAL_173]]) : (!llvm.ptr<4>, !llvm.ptr) -> ()
// CHECK-NEXT:        %[[VAL_187:.*]] = llvm.load %[[VAL_177]] : !llvm.ptr -> !llvm.struct<(vector<2xi8>)>
// CHECK-NEXT:        llvm.store %[[VAL_187]], %[[VAL_172]] : !llvm.struct<(vector<2xi8>)>, !llvm.ptr
// CHECK-NEXT:        %[[VAL_188:.*]] = llvm.addrspacecast %[[VAL_171]] : !llvm.ptr to !llvm.ptr<4>
// CHECK-NEXT:        %[[VAL_189:.*]] = llvm.addrspacecast %[[VAL_172]] : !llvm.ptr to !llvm.ptr<4>
// CHECK-NEXT:        call @_ZN9structvecC1EOS_(%[[VAL_188]], %[[VAL_189]]) : (!llvm.ptr<4>, !llvm.ptr<4>) -> ()
// CHECK-NEXT:        %[[VAL_190:.*]] = llvm.load %[[VAL_171]] : !llvm.ptr -> !llvm.struct<(vector<2xi8>)>
// CHECK-NEXT:        return %[[VAL_190]] : !llvm.struct<(vector<2xi8>)>
// CHECK-NEXT:      }


// CHECK-LABEL:     func.func @_ZN9structvecC1ESt16initializer_listIcE(
// CHECK-SAME:          %[[VAL_191:.*]]: !llvm.ptr<4> {llvm.align = 2 : i64, llvm.dereferenceable_or_null = 2 : i64, llvm.noundef}
// CHECK-SAME:          %[[VAL_192:.*]]: !llvm.ptr {llvm.align = 8 : i64, llvm.byval = !llvm.struct<(memref<?xi8, 4>, i64)>, llvm.noundef})
// CHECK-DAG:         %[[VAL_193:.*]] = arith.constant -1 : i32
// CHECK-DAG:         %[[VAL_194:.*]] = arith.constant 0 : i32
// CHECK-DAG:         %[[VAL_195:.*]] = arith.constant 0 : i8
// CHECK-NEXT:        %[[VAL_196:.*]] = llvm.addrspacecast %[[VAL_192]] : !llvm.ptr to !llvm.ptr<4>
// CHECK-NEXT:        %[[VAL_197:.*]] = llvm.getelementptr inbounds %[[VAL_191]][0, 0] : (!llvm.ptr<4>) -> !llvm.ptr<4>, !llvm.struct<(vector<2xi8>)>
// CHECK-NEXT:        affine.for %[[VAL_198:.*]] = 0 to 2 {
// CHECK-NEXT:          %[[VAL_199:.*]] = arith.index_cast %[[VAL_198]] : index to i32
// CHECK-NEXT:          %[[VAL_200:.*]] = func.call @_ZNKSt16initializer_listIcE5beginEv(%[[VAL_196]]) : (!llvm.ptr<4>) -> memref<?xi8, 4>
// CHECK-NEXT:          %[[VAL_201:.*]] = arith.index_castui %[[VAL_199]] : i32 to index
// CHECK-NEXT:          %[[VAL_202:.*]] = memref.load %[[VAL_200]]{{\[}}%[[VAL_201]]] : memref<?xi8, 4>
// CHECK-NEXT:          %[[VAL_203:.*]] = arith.cmpi ne, %[[VAL_202]], %[[VAL_195]] : i8
// CHECK-NEXT:          %[[VAL_204:.*]] = arith.select %[[VAL_203]], %[[VAL_193]], %[[VAL_194]] : i32
// CHECK-NEXT:          %[[VAL_205:.*]] = arith.trunci %[[VAL_204]] : i32 to i8
// CHECK-NEXT:          %[[VAL_206:.*]] = llvm.load %[[VAL_197]] : !llvm.ptr<4> -> vector<2xi8>
// CHECK-NEXT:          %[[VAL_207:.*]] = vector.insertelement %[[VAL_205]], %[[VAL_206]]{{\[}}%[[VAL_199]] : i32] : vector<2xi8>
// CHECK-NEXT:          llvm.store %[[VAL_207]], %[[VAL_197]] : vector<2xi8>, !llvm.ptr<4>
// CHECK-NEXT:        }
// CHECK-NEXT:        return
// CHECK-NEXT:      }

SYCL_EXTERNAL structvec test_init() {
  structvec sv{0, 1};
  return sv;
}
