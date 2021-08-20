// RUN: %clangxx -fsycl -fno-builtin %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>

class KernelTestMemcpy;

bool kernel_test_memcpy(sycl::queue &deviceQueue) {
  bool success = true;
  char src[20] = "abcdefg012345xyzvvv";
  char dst[20];
  {
    sycl::buffer<char, 1> buffer1(src, sycl::range<1>(20));
    sycl::buffer<char, 1> buffer2(dst, sycl::range<1>(20));
    deviceQueue.submit([&](sycl::handler &cgh) {
      auto dst_acc = buffer2.get_access<sycl::access::mode::write>(cgh);
      auto src_acc = buffer1.get_access<sycl::access::mode::read>(cgh);
      cgh.single_task<class KernelTestMemcpy>(
          [=]() { memcpy(dst_acc.get_pointer(), src_acc.get_pointer(), 20); });
    });
  }

  for (size_t idx = 0; idx < 20; ++idx) {
    if (dst[idx] != src[idx]) {
      success = false;
      break;
    }
  }

  return success;
}

class KernelTestMemset;
bool kernel_test_memset(sycl::queue &deviceQueue) {
  bool success = true;
  unsigned char dst[20] = {
      0,
  };
  {
    sycl::buffer<unsigned char, 1> buffer1(dst, sycl::range<1>(20));
    deviceQueue.submit([&](sycl::handler &cgh) {
      auto dst_acc = buffer1.get_access<sycl::access::mode::write>(cgh);
      cgh.single_task<class KernelTestMemset>(
          [=]() { memset(dst_acc.get_pointer(), 'P', 18); });
    });
  }

  for (size_t idx = 0; idx < 18; ++idx) {
    if (dst[idx] != 'P') {
      success = false;
      break;
    }
  }

  if (dst[18] != 0 || dst[19] != 0)
    success = false;

  return success;
}

class KernelTestMemcmp;
bool kernel_test_memcmp(sycl::queue &deviceQueue) {
  bool success = true;
  int results[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  int refer[8] = {0, 0, 0, 0, 1, 1, 0, -1};

  const char str1[] = "a";
  const char str2[] = "ab";
  const char str3[] = "ab12";
  const char str4[] = "ab123";
  const char str5[] = "aB123";
  const char str6[] = "udqw34";
  const char str7[] = "asdfewkfmikewmfi";
  const char str8[] = "asdfewkfnikewmfi";
  {
    sycl::buffer<int, 1> buffer1(results, sycl::range<1>(8));
    sycl::buffer<char, 1> buffer_str1(str1, sycl::range<1>(sizeof(str1)));
    sycl::buffer<char, 1> buffer_str2(str2, sycl::range<1>(sizeof(str2)));
    sycl::buffer<char, 1> buffer_str3(str3, sycl::range<1>(sizeof(str3)));
    sycl::buffer<char, 1> buffer_str4(str4, sycl::range<1>(sizeof(str4)));
    sycl::buffer<char, 1> buffer_str5(str5, sycl::range<1>(sizeof(str5)));
    sycl::buffer<char, 1> buffer_str6(str6, sycl::range<1>(sizeof(str6)));
    sycl::buffer<char, 1> buffer_str7(str7, sycl::range<1>(sizeof(str7)));
    sycl::buffer<char, 1> buffer_str8(str8, sycl::range<1>(sizeof(str8)));
    deviceQueue.submit([&](sycl::handler &cgh) {
      auto results_acc = buffer1.get_access<sycl::access::mode::write>(cgh);
      auto str1_acc = buffer_str1.get_access<sycl::access::mode::read>(cgh);
      auto str2_acc = buffer_str2.get_access<sycl::access::mode::read>(cgh);
      auto str3_acc = buffer_str3.get_access<sycl::access::mode::read>(cgh);
      auto str4_acc = buffer_str4.get_access<sycl::access::mode::read>(cgh);
      auto str5_acc = buffer_str5.get_access<sycl::access::mode::read>(cgh);
      auto str6_acc = buffer_str6.get_access<sycl::access::mode::read>(cgh);
      auto str7_acc = buffer_str7.get_access<sycl::access::mode::read>(cgh);
      auto str8_acc = buffer_str8.get_access<sycl::access::mode::read>(cgh);
      cgh.single_task<class KernelTestMemcmp>([=]() {
        results_acc[0] =
            memcmp(str1_acc.get_pointer(), str1_acc.get_pointer(), 1);
        results_acc[1] =
            memcmp(str2_acc.get_pointer(), str1_acc.get_pointer(), 1);
        results_acc[2] =
            memcmp(str3_acc.get_pointer(), str2_acc.get_pointer(), 2);
        results_acc[3] =
            memcmp(str3_acc.get_pointer(), str4_acc.get_pointer(), 4);
        results_acc[4] =
            memcmp(str4_acc.get_pointer(), str5_acc.get_pointer(), 2);
        results_acc[5] =
            memcmp(str6_acc.get_pointer(), str7_acc.get_pointer(), 6);
        results_acc[6] =
            memcmp(str5_acc.get_pointer(), str6_acc.get_pointer(), 0);
        results_acc[7] = memcmp(str7_acc.get_pointer(), str8_acc.get_pointer(),
                                sizeof(str7));
      });
    });
  }

  for (size_t idx = 0; idx < 8; ++idx) {
    if ((results[idx] * refer[idx]) > 0 ||
        ((results[idx] == 0) && (refer[idx] == 0))) {
      continue;
    } else {
      success = false;
      break;
    }
  }

  return success;
}

class KernelTestMemcmpAlign;
bool kernel_test_memcmp_align(sycl::queue &deviceQueue) {
  bool success = true;
  int cmps[16] = {
      -1,
  };
  int refs[16] = {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, -1, 0, -1, 1};
  {
    sycl::buffer<int, 1> cmp_buf(cmps, sycl::range<1>{16});
    deviceQueue.submit([&](sycl::handler &cgh) {
      auto cmp_acc = cmp_buf.get_access<sycl::access::mode::write>(cgh);
      cgh.single_task<class KernelTestMemcmpAlign>([=]() {
        uint8_t s1_buf[32], s2_buf[32];
        uint8_t *s1_ptr = &s1_buf[0];
        uint8_t *s2_ptr = &s2_buf[0];
        if (reinterpret_cast<unsigned long>(s1_ptr) % 4 ==
            reinterpret_cast<unsigned long>(s2_ptr) % 4) {
          s1_ptr += 1;
          s2_ptr += 2;
        }

        s1_ptr[0] = s2_ptr[0] = 'a';
        s1_ptr[1] = s2_ptr[1] = 'x';
        s1_ptr[2] = s2_ptr[2] = 'f';
        s1_ptr[3] = s2_ptr[3] = '1';
        s1_ptr[4] = s2_ptr[4] = '9';
        s1_ptr[5] = s2_ptr[5] = 'T';
        cmp_acc[0] = memcmp(s1_ptr, s2_ptr, 0);
        cmp_acc[1] = memcmp(s1_ptr, s2_ptr, 1);
        cmp_acc[2] = memcmp(s1_ptr, s2_ptr, 2);
        cmp_acc[3] = memcmp(s1_ptr, s2_ptr, 3);
        cmp_acc[4] = memcmp(s1_ptr, s2_ptr, 4);
        cmp_acc[5] = memcmp(s1_ptr, s2_ptr, 5);
        cmp_acc[6] = memcmp(s1_ptr, s2_ptr, 6);
        s1_ptr[6] = 'Y';
        s2_ptr[6] = 'U';
        s1_ptr[7] = s2_ptr[7] = '7';
        s1_ptr[8] = 'b';
        s2_ptr[8] = 'g';
        cmp_acc[7] = memcmp(s1_ptr, s2_ptr, 7);
        cmp_acc[8] = memcmp(s1_ptr, s2_ptr, 8);
        cmp_acc[9] = memcmp(s1_ptr, s2_ptr, 9);
        s1_ptr[6] = 'U';
        cmp_acc[10] = memcmp(s1_ptr, s2_ptr, 7);
        cmp_acc[11] = memcmp(s1_ptr, s2_ptr, 8);
        cmp_acc[12] = memcmp(s1_ptr, s2_ptr, 9);
        s1_ptr[8] = 'g';
        s1_ptr[9] = s2_ptr[9] = 'j';
        cmp_acc[13] = memcmp(s1_ptr, s2_ptr, 10);
        s1_ptr[10] = '1';
        s2_ptr[10] = '2';
        cmp_acc[14] = memcmp(s1_ptr, s2_ptr, 11);
        s1_ptr[5] = 'Z';
        cmp_acc[15] = memcmp(s1_ptr, s2_ptr, 11);
      });
    });
  }

  for (size_t idx = 0; idx < 13; ++idx) {
    if ((cmps[idx] == 0 && refs[idx] == 0) || (cmps[idx] * refs[idx] > 0)) {
      continue;
    } else {
      success = false;
      break;
    }
  }

  return success;
}

class KernelTestMemcpyAddrSpace;

bool kernel_test_memcpy_addr_space(sycl::queue &deviceQueue) {
  char src[16] = "abcdefg";
  char dst[16];
  char dst1[16];
  {
    sycl::buffer<char, 1> buffer1(src, sycl::range<1>(16));
    sycl::buffer<char, 1> buffer2(dst, sycl::range<1>(16));
    sycl::buffer<char, 1> buffer3(dst1, sycl::range<1>(16));
    deviceQueue.submit([&](sycl::handler &cgh) {
      sycl::accessor<char, 1, sycl::access::mode::read,
                     sycl::access::target::constant_buffer,
                     sycl::access::placeholder::false_t>
          src_acc(buffer1, cgh);

      sycl::accessor<char, 1, sycl::access::mode::read_write,
                     sycl::access::target::local,
                     sycl::access::placeholder::false_t>
          local_acc(sycl::range<1>(16), cgh);

      sycl::accessor<char, 1, sycl::access::mode::write,
                     sycl::access::target::global_buffer,
                     sycl::access::placeholder::false_t>
          dst_acc(buffer2, cgh);

      sycl::accessor<char, 1, sycl::access::mode::write,
                     sycl::access::target::global_buffer,
                     sycl::access::placeholder::false_t>
          dst1_acc(buffer3, cgh);
      cgh.single_task<class KernelTestMemcpyAddrSpace>([=]() {
        // memcpy from constant buffer to local buffer
        memcpy(local_acc.get_pointer(), src_acc.get_pointer(), 8);
        for (size_t idx = 0; idx < 7; ++idx)
          local_acc[idx] += 1;
        // memcpy from local buffer to global buffer
        memcpy(dst_acc.get_pointer(), local_acc.get_pointer(), 8);
        char device_buf[16];
        // memcpy from constant buffer to private memory
        memcpy(device_buf, src_acc.get_pointer(), 8);
        for (size_t idx = 0; idx < 7; ++idx) {
          device_buf[idx] += 2;
          // memcpy from private to global buffer
          memcpy(dst1_acc.get_pointer(), device_buf, 8);
        }
      });
    });
  }

  if (strcmp("bcdefgh", dst) != 0)
    return false;

  if (strcmp("cdefghi", dst1) != 0)
    return false;

  return true;
}
int main() {
  bool success = true;
  sycl::queue deviceQueue;
  success = kernel_test_memcpy(deviceQueue);
  assert(((void)"memcpy test failed!", success));

  success = kernel_test_memset(deviceQueue);
  assert(((void)"memset test failed!", success));

  success = kernel_test_memcmp(deviceQueue);
  assert(((void)"memcmp test failed!", success));

  success = kernel_test_memcmp_align(deviceQueue);
  assert(((void)"memcmp alignment test failed!", success));

  success = kernel_test_memcpy_addr_space(deviceQueue);
  assert(((void)"memcpy test with address space failed!", success));
  std::cout << "passed!" << std::endl;
  return 0;
}
