// Test that llvm.bitreverse is lowered correctly by llvm-spirv

// UNSUPPORTED: hip || cuda

// RUN: %{build} -o %t.out -O2
// RUN: %{run} %t.out

#include <string.h>
#include <sycl/sycl.hpp>
#include <iostream>

using namespace sycl;

template <typename TYPE>
__attribute__((optnone, noinline)) TYPE reference_reverse(TYPE a, const int bitlength) {
  TYPE ret = 0;
  for (auto i = 0; i<bitlength; i++) {
    ret<<=1;
    ret |= a & 0x1;
    a>>=1;
  }
  return ret;
}

template <typename TYPE>
__attribute__((noinline)) TYPE reverse(TYPE a) {
  return __builtin_elementwise_bitreverse(a);
}

////////////////////////////////////////////////////////////////////////////////////
// 8-bit
////////////////////////////////////////////////////////////////////////////////////
template <typename UINT8>
__attribute__((noinline)) UINT8 reverse8(UINT8 a) {
    a = ((0x55 & a) << 1) | (0x55 & (a >> 1));
    a = ((0x33 & a) << 2) | (0x33 & (a >> 2));
    return (a << 4) | (a >> 4);
}
template <typename UINT8>
void do_bitreverse_test8(int *result, int bytesize) {
  for (uint8_t u=0 ; u<250; u++) {
    UINT8 data=u;    
    UINT8 ref = reference_reverse(data,8);
    UINT8 opt = reverse8(data); // avoid bug with __builtin_elementwise_bitreverse(a) on scalar 8-bit types

    if (memcmp(&ref,&opt,bytesize)) {
      *result=1;
      return;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////
// 16-bit
////////////////////////////////////////////////////////////////////////////////////
template <typename UINT16>
__attribute__((noinline)) UINT16 reverse16(UINT16 a) {
    a = ((0x5555 & a) << 1) | (0x5555 & (a >> 1));
    a = ((0x3333 & a) << 2) | (0x3333 & (a >> 2));
    a = ((0x0F0F & a) << 4) | (0x0F0F & (a >> 4));    
    return (a << 8) | (a >> 8);
}

template <typename UINT16>
void do_bitreverse_test16(int *result, int bytesize) {
  for (uint16_t u=0 ; u<0xFF00; u+=0x13) {
    UINT16 data=u;
    UINT16 ref = reference_reverse(data,16);
    UINT16 opt = reverse16(data); // avoid bug with __builtin_elementwise_bitreverse(a) on scalar 16-bit types

    if (memcmp(&ref,&opt,bytesize)) {
      *result=1;
      return;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////
// 32-bit
////////////////////////////////////////////////////////////////////////////////////
template <typename UINT32>
void do_bitreverse_test32(int *result, int bytesize) {
  for (uint32_t u=0 ; u<(0xFF<<24); u+=0xABCD13) {
    UINT32 data=u;
    UINT32 ref = reference_reverse(data,32);
    UINT32 opt = reverse(data);

    if (memcmp(&ref,&opt,bytesize)) {
      *result=1;
      return;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////
// 64-bit
////////////////////////////////////////////////////////////////////////////////////
template <typename UINT64>
void do_bitreverse_test64(int *result, int bytesize) {
  for (uint64_t u=0 ; u<(0xFFUL<<56); u+=0xABCDABCDABCD13UL) {
    UINT64 data=u;
    UINT64 ref = reference_reverse(data,64);
    UINT64 opt = reverse(data);

    if (memcmp(&ref,&opt,bytesize)) {
      *result=1;
      return;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> class BitreverseTest;

template <typename UINT8, typename UINT16, typename UINT32, typename UINT64>
void testTypes(int elements) {
  queue q;

  int *result = (int *) malloc_host(sizeof(int),q);
  *result=0;

  q.submit([=](handler &cgh) { cgh.single_task<BitreverseTest<UINT8>> ([=]() { do_bitreverse_test8 <UINT8> (result,elements*sizeof(uint8_t)); }); }); q.wait();
  if (*result) {
    std::cerr << "Failed bitreverse 8-bit, #elements=" <<  sizeof(UINT8)/sizeof(uint8_t) <<"\n";
    exit(1);
  }
  q.submit([=](handler &cgh) { cgh.single_task<BitreverseTest<UINT16>>([=]() { do_bitreverse_test16<UINT16>(result,elements*sizeof(uint16_t)); }); }); q.wait();
  if (*result) {
    std::cerr << "Failed bitreverse 16-bit, #elements=" <<  sizeof(UINT16)/sizeof(uint16_t) <<"\n";
    exit(1);
  }
  q.submit([=](handler &cgh) { cgh.single_task<BitreverseTest<UINT32>>([=]() { do_bitreverse_test32<UINT32>(result,elements*sizeof(uint32_t)); }); }); q.wait();
  if (*result) {
    std::cerr << "Failed bitreverse 32-bit, #elements=" <<  sizeof(UINT32)/sizeof(uint32_t) <<"\n";
    exit(1);
  }
  q.submit([=](handler &cgh) { cgh.single_task<BitreverseTest<UINT64>>([=]() { do_bitreverse_test64<UINT64>(result,elements*sizeof(uint64_t)); }); }); q.wait();
  if (*result) {
    std::cerr << "Failed bitreverse 64-bit, #elements=" <<  sizeof(UINT64)/sizeof(uint64_t) <<"\n";
    exit(1);
  }
}

using  uint8_t2 =  uint8_t __attribute__((ext_vector_type(2)));
using uint16_t2 = uint16_t __attribute__((ext_vector_type(2)));
using uint32_t2 = uint32_t __attribute__((ext_vector_type(2)));
using uint64_t2 = uint64_t __attribute__((ext_vector_type(2)));

using  uint8_t3 =  uint8_t __attribute__((ext_vector_type(3)));
using uint16_t3 = uint16_t __attribute__((ext_vector_type(3)));
using uint32_t3 = uint32_t __attribute__((ext_vector_type(3)));
using uint64_t3 = uint64_t __attribute__((ext_vector_type(3)));

using  uint8_t4 =  uint8_t __attribute__((ext_vector_type(4)));
using uint16_t4 = uint16_t __attribute__((ext_vector_type(4)));
using uint32_t4 = uint32_t __attribute__((ext_vector_type(4)));
using uint64_t4 = uint64_t __attribute__((ext_vector_type(4)));

using  uint8_t8 =  uint8_t __attribute__((ext_vector_type(8)));
using uint16_t8 = uint16_t __attribute__((ext_vector_type(8)));
using uint32_t8 = uint32_t __attribute__((ext_vector_type(8)));
using uint64_t8 = uint64_t __attribute__((ext_vector_type(8)));

using  uint8_t16 =  uint8_t __attribute__((ext_vector_type(16)));
using uint16_t16 = uint16_t __attribute__((ext_vector_type(16)));
using uint32_t16 = uint32_t __attribute__((ext_vector_type(16)));
using uint64_t16 = uint64_t __attribute__((ext_vector_type(16)));

int main() {
  queue q;

  testTypes<uint8_t,   uint16_t,   uint32_t,   uint64_t>   (1);
  testTypes<uint8_t2,  uint16_t2,  uint32_t2,  uint64_t2>  (2);
  testTypes<uint8_t3,  uint16_t3,  uint32_t3,  uint64_t3>  (3);
  testTypes<uint8_t4,  uint16_t4,  uint32_t4,  uint64_t4>  (4);
  testTypes<uint8_t8,  uint16_t8,  uint32_t8,  uint64_t8>  (8);
  testTypes<uint8_t16, uint16_t16, uint32_t16, uint64_t16> (16);
  
  return 0;
}

