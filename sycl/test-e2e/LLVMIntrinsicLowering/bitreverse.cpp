// Test that llvm.bitreverse is lowered correctly by llvm-spirv

// RUN: %{build} -o %t.out -O2
// RUN: %{run} %t.out

#include <string.h>
#include <sycl/sycl.hpp>
#include <iostream>

using namespace sycl;

////////////////////////////////////////////////////////////////////////////////////
// 8-bit
////////////////////////////////////////////////////////////////////////////////////
template <typename UINT8>
__attribute__((optnone, noinline)) UINT8 reference_reverse8(UINT8 a) {
    a = ((0x55 & a) << 1) | (0x55 & (a >> 1));
    a = ((0x33 & a) << 2) | (0x33 & (a >> 2));
    return (a << 4) | (a >> 4);
}
template <typename UINT8>
__attribute__((noinline)) UINT8 reverse8(UINT8 a) {
    a = ((0x55 & a) << 1) | (0x55 & (a >> 1));
    a = ((0x33 & a) << 2) | (0x33 & (a >> 2));
    return (a << 4) | (a >> 4);
}
template <typename UINT8>
void do_bitreverse_test8(int *result) {
  for (uint8_t u=0 ; u<250; u++) {
    UINT8 data=u;    
    UINT8 ref = reference_reverse8(data);
    UINT8 opt = reverse8(data);

    if (memcmp(&ref,&opt,sizeof(UINT8))) {
      *result=1;
      return;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////
// 16-bit
////////////////////////////////////////////////////////////////////////////////////
template <typename UINT16>
__attribute__((optnone, noinline)) UINT16 reference_reverse16(UINT16 a) {
    a = ((0x5555 & a) << 1) | (0x5555 & (a >> 1));
    a = ((0x3333 & a) << 2) | (0x3333 & (a >> 2));
    a = ((0x0F0F & a) << 4) | (0x0F0F & (a >> 4));    
    return (a << 8) | (a >> 8);
}
template <typename UINT16>
__attribute__((noinline)) UINT16 reverse16(UINT16 a) {
    a = ((0x5555 & a) << 1) | (0x5555 & (a >> 1));
    a = ((0x3333 & a) << 2) | (0x3333 & (a >> 2));
    a = ((0x0F0F & a) << 4) | (0x0F0F & (a >> 4));    
    return (a << 8) | (a >> 8);
}

template <typename UINT16>
void do_bitreverse_test16(int *result) {
  for (uint16_t u=0 ; u<0xFF00; u+=0x13) {
    UINT16 data=u;
    UINT16 ref = reference_reverse16(data);
    UINT16 opt = reverse16(data);

    if (memcmp(&ref,&opt,sizeof(UINT16))) {
      *result=1;
      return;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////
// 32-bit
////////////////////////////////////////////////////////////////////////////////////
template <typename UINT32>
__attribute__((optnone, noinline)) UINT32 reference_reverse32(UINT32 a) {
    a = ((0x55555555 & a) << 1) | (0x55555555 & (a >> 1));
    a = ((0x33333333 & a) << 2) | (0x33333333 & (a >> 2));
    a = ((0x0F0F0F0F & a) << 4) | (0x0F0F0F0F & (a >> 4));
    a = ((0x00FF00FF & a) << 8) | (0x00FF00FF & (a >> 8));            
    return (a << 16) | (a >> 16);
}
template <typename UINT32>
__attribute__((noinline)) UINT32 reverse32(UINT32 a) {
    a = ((0x55555555 & a) << 1) | (0x55555555 & (a >> 1));
    a = ((0x33333333 & a) << 2) | (0x33333333 & (a >> 2));
    a = ((0x0F0F0F0F & a) << 4) | (0x0F0F0F0F & (a >> 4));
    a = ((0x00FF00FF & a) << 8) | (0x00FF00FF & (a >> 8));            
    return (a << 16) | (a >> 16);
}
template <typename UINT32>
void do_bitreverse_test32(int *result) {
  for (uint32_t u=0 ; u<(0xFF<<24); u+=0xABCD13) {
    UINT32 data=u;
    UINT32 ref = reference_reverse32(data);
    UINT32 opt = reverse32(data);

    if (memcmp(&ref,&opt,sizeof(UINT32))) {
      *result=1;
      return;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////
// 64-bit
////////////////////////////////////////////////////////////////////////////////////
template <typename UINT64>
__attribute__((optnone, noinline)) UINT64 reference_reverse64(UINT64 a) {
    a = ((0x5555555555555555UL & a) << 1) | (0x5555555555555555UL & (a >> 1));
    a = ((0x3333333333333333UL & a) << 2) | (0x3333333333333333UL & (a >> 2));
    a = ((0x0F0F0F0F0F0F0F0FUL & a) << 4) | (0x0F0F0F0F0F0F0F0FUL & (a >> 4));
    a = ((0x00FF00FF00FF00FFUL & a) << 8) | (0x00FF00FF00FF00FFUL & (a >> 8));
    a = ((0x0000FFFF0000FFFFUL & a) <<16) | (0x0000FFFF0000FFFFUL & (a >>16));    
    return (a << 32) | (a >> 32);
}
template <typename UINT64>
__attribute__((noinline)) UINT64 reverse64(UINT64 a) {
    a = ((0x5555555555555555UL & a) << 1) | (0x5555555555555555UL & (a >> 1));
    a = ((0x3333333333333333UL & a) << 2) | (0x3333333333333333UL & (a >> 2));
    a = ((0x0F0F0F0F0F0F0F0FUL & a) << 4) | (0x0F0F0F0F0F0F0F0FUL & (a >> 4));
    a = ((0x00FF00FF00FF00FFUL & a) << 8) | (0x00FF00FF00FF00FFUL & (a >> 8));
    a = ((0x0000FFFF0000FFFFUL & a) <<16) | (0x0000FFFF0000FFFFUL & (a >>16));
    return (a << 32) | (a >> 32);
}
template <typename UINT64>
void do_bitreverse_test64(int *result) {
  for (uint64_t u=0 ; u<(0xFFUL<<56); u+=0xABCDABCDABCD13UL) {
    UINT64 data=u;
    UINT64 ref = reference_reverse64(data);
    UINT64 opt = reverse64(data);

    if (memcmp(&ref,&opt,sizeof(UINT64))) {
      *result=1;
      return;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> class BitreverseTest;

template <typename UINT8, typename UINT16, typename UINT32, typename UINT64>
void testTypes() {
  queue q;

  int *result = (int *) malloc_host(sizeof(int),q);
  *result=0;

  q.submit([=](handler &cgh) { cgh.single_task<BitreverseTest<UINT8>> ([=]() { do_bitreverse_test8 <UINT8> (result); }); }); q.wait();
  if (*result) {
    std::cerr << "Failed bitreverse 8-bit, #elements=" <<  sizeof(UINT8)/sizeof(uint8_t) <<"\n";
    exit(1);
  }
  q.submit([=](handler &cgh) { cgh.single_task<BitreverseTest<UINT16>>([=]() { do_bitreverse_test16<UINT16>(result); }); }); q.wait();
  if (*result) {
    std::cerr << "Failed bitreverse 16-bit, #elements=" <<  sizeof(UINT16)/sizeof(uint16_t) <<"\n";
    exit(1);
  }
  q.submit([=](handler &cgh) { cgh.single_task<BitreverseTest<UINT32>>([=]() { do_bitreverse_test32<UINT32>(result); }); }); q.wait();
  if (*result) {
    std::cerr << "Failed bitreverse 32-bit, #elements=" <<  sizeof(UINT32)/sizeof(uint32_t) <<"\n";
    exit(1);
  }
  q.submit([=](handler &cgh) { cgh.single_task<BitreverseTest<UINT64>>([=]() { do_bitreverse_test64<UINT64>(result); }); }); q.wait();
  if (*result) {
    std::cerr << "Failed bitreverse 64-bit, #elements=" <<  sizeof(UINT64)/sizeof(uint64_t) <<"\n";
    exit(1);
  }
}

typedef  uint8_t  uint8_t2  __attribute__((ext_vector_type(2)));
typedef uint16_t uint16_t2  __attribute__((ext_vector_type(2)));
typedef uint32_t uint32_t2  __attribute__((ext_vector_type(2)));
typedef uint64_t uint64_t2  __attribute__((ext_vector_type(2)));

typedef  uint8_t  uint8_t4  __attribute__((ext_vector_type(4)));
typedef uint16_t uint16_t4  __attribute__((ext_vector_type(4)));
typedef uint32_t uint32_t4  __attribute__((ext_vector_type(4)));
typedef uint64_t uint64_t4  __attribute__((ext_vector_type(4)));

int main() {
  queue q;

  testTypes<uint8_t,  uint16_t,  uint32_t,  uint64_t>  ();
  testTypes<uint8_t2, uint16_t2, uint32_t2, uint64_t2> ();
  testTypes<uint8_t4, uint16_t4, uint32_t4, uint64_t4> (); 
  
  return 0;
}

