// UNSUPPORTED: hip
// RUN: %{build} -fno-builtin -o %t.out
// RUN: %{run} %t.out
//
// RUN: %{build} -fno-builtin -fsycl-device-lib-jit-link -o %t.out
// RUN: %if !gpu %{ %{run} %t.out %}

// UNSUPPORTED: accelerator

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>
enum USM_TEST_RES { USM_ALLOC_FAIL = -1, USM_TEST_PASS = 0, USM_TEST_FAIL = 1 };

template <class DeviceMemcpyTest>
void device_memcpy_invoke(sycl::queue &deviceQueue, uint8_t *dest,
                          const uint8_t *src, size_t n) {
  deviceQueue
      .submit([&](sycl::handler &cgh) {
        cgh.single_task<DeviceMemcpyTest>([=]() { memcpy(dest, src, n); });
      })
      .wait();
}

class KernelTestMemcpy;
bool kernel_test_memcpy(sycl::queue &deviceQueue) {
  bool success = true;
  char src[20] = "abcdefg012345xyzvvv";
  char dst[20] = {
      0,
  };
  {
    sycl::buffer<char, 1> buffer1(src, sycl::range<1>(20));
    sycl::buffer<char, 1> buffer2(dst, sycl::range<1>(20));
    deviceQueue.submit([&](sycl::handler &cgh) {
      auto dst_acc = buffer2.get_access<sycl::access::mode::write>(cgh);
      auto src_acc = buffer1.get_access<sycl::access::mode::read>(cgh);
      cgh.single_task<class KernelTestMemcpy>([=]() {
        memcpy(dst_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
               src_acc.get_multi_ptr<sycl::access::decorated::no>().get(), 20);
      });
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

class KernelTestMemcpyInit;
class KernelTestMemcpyUSM0;
class KernelTestMemcpyUSM1;
class KernelTestMemcpyUSM2;
class KernelTestMemcpyUSM3;
class KernelTestMemcpyUSM4;
USM_TEST_RES kernel_test_memcpy_usm(sycl::queue &deviceQueue) {
  sycl::device dev = deviceQueue.get_device();
  sycl::context ctxt = deviceQueue.get_context();
  uint8_t *usm_shared_dest =
      (uint8_t *)sycl::aligned_alloc_shared(alignof(uint32_t), 32, dev, ctxt);
  uint8_t *usm_shared_src =
      (uint8_t *)sycl::aligned_alloc_shared(alignof(uint32_t), 32, dev, ctxt);
  if (usm_shared_dest == nullptr || usm_shared_src == nullptr)
    return USM_ALLOC_FAIL;
  // Init src usm memory
  const char *host_init_str = "abcdefghijklmnopqrstuvwxyz";
  size_t str_len = strlen(host_init_str);
  deviceQueue
      .submit([&](sycl::handler &cgh) {
        cgh.single_task<class KernelTestMemcpyInit>([=]() {
          char c = 'a';
          for (size_t idx = 0; idx < 32; ++idx)
            usm_shared_src[idx] = c++;
        });
      })
      .wait();
  int usm_memcheck_pass = 0;
  // Memcpy 3 bytest from aligned src to aligned dest
  device_memcpy_invoke<KernelTestMemcpyUSM0>(deviceQueue, usm_shared_dest,
                                             usm_shared_src, 3);
  usm_memcheck_pass = memcmp(usm_shared_dest, usm_shared_src, 3);
  if (usm_memcheck_pass != 0) {
    sycl::free(usm_shared_src, ctxt);
    sycl::free(usm_shared_dest, ctxt);
    return USM_TEST_FAIL;
  }

  // Memcpy 15 bytest from aligned src to aligned dest
  device_memcpy_invoke<KernelTestMemcpyUSM1>(deviceQueue, usm_shared_dest,
                                             usm_shared_src, 15);
  usm_memcheck_pass = memcmp(usm_shared_dest, usm_shared_src, 15);
  if (usm_memcheck_pass != 0) {
    sycl::free(usm_shared_src, ctxt);
    sycl::free(usm_shared_dest, ctxt);
    return USM_TEST_FAIL;
  }

  deviceQueue
      .submit([&](sycl::handler &cgh) { cgh.memset(usm_shared_dest, 0, 32); })
      .wait();
  // Memcpy 1 byte from unaligned src to unaligned dest;
  device_memcpy_invoke<KernelTestMemcpyUSM2>(deviceQueue, usm_shared_dest + 1,
                                             usm_shared_src + 1, 1);
  usm_memcheck_pass = memcmp(usm_shared_dest + 1, usm_shared_src + 1, 1);
  if (usm_memcheck_pass != 0) {
    sycl::free(usm_shared_src, ctxt);
    sycl::free(usm_shared_dest, ctxt);
    return USM_TEST_FAIL;
  }

  // Memcpy 12 bytes from unaligned src to unalinged dest;
  device_memcpy_invoke<KernelTestMemcpyUSM3>(deviceQueue, usm_shared_dest + 3,
                                             usm_shared_src + 3, 12);
  usm_memcheck_pass = memcmp(usm_shared_dest + 3, usm_shared_src + 3, 12);
  if (usm_memcheck_pass != 0) {
    sycl::free(usm_shared_src, ctxt);
    sycl::free(usm_shared_dest, ctxt);
    return USM_TEST_FAIL;
  }

  // Memcpy 7 bytes from unaligned src to unaligned dest
  device_memcpy_invoke<KernelTestMemcpyUSM4>(deviceQueue, usm_shared_dest + 9,
                                             usm_shared_src + 7, 7);
  usm_memcheck_pass = memcmp(usm_shared_dest + 9, usm_shared_src + 7, 7);
  if (usm_memcheck_pass != 0) {
    sycl::free(usm_shared_src, ctxt);
    sycl::free(usm_shared_dest, ctxt);
    return USM_TEST_FAIL;
  }
  sycl::free(usm_shared_src, ctxt);
  sycl::free(usm_shared_dest, ctxt);
  return USM_TEST_PASS;
}

template <class DeviceMemsetTest>
void device_memset_invoke(sycl::queue &deviceQueue, uint8_t *dest, int c,
                          size_t n) {
  deviceQueue
      .submit([&](sycl::handler &cgh) {
        cgh.single_task<DeviceMemsetTest>([=]() { memset(dest, c, n); });
      })
      .wait();
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
      cgh.single_task<class KernelTestMemset>([=]() {
        memset(dst_acc.get_multi_ptr<sycl::access::decorated::no>().get(), 'P',
               18);
      });
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

class KernelTestMemsetUSM0;
class KernelTestMemsetUSM1;
class KernelTestMemsetUSM2;
class KernelTestMemsetUSM3;

USM_TEST_RES kernel_test_memset_usm(sycl::queue &deviceQueue) {
  sycl::device dev = deviceQueue.get_device();
  sycl::context ctxt = deviceQueue.get_context();
  uint8_t host_ref_buffer[32];
  uint8_t *usm_shared_buffer =
      (uint8_t *)sycl::aligned_alloc_shared(alignof(uint32_t), 32, dev, ctxt);
  if (usm_shared_buffer == nullptr)
    return USM_ALLOC_FAIL;

  deviceQueue
      .submit(
          [&](sycl::handler &cgh) { cgh.memset(usm_shared_buffer, 0xFF, 32); })
      .wait();

  int usm_memcheck_pass = 0;
  // memset 17 bytes on aligned address
  device_memset_invoke<KernelTestMemsetUSM0>(deviceQueue, usm_shared_buffer,
                                             0xEE, 17);
  memset(host_ref_buffer, 0xFF, 32);
  memset(host_ref_buffer, 0xEE, 17);
  usm_memcheck_pass = memcmp(host_ref_buffer, usm_shared_buffer, 32);
  if (usm_memcheck_pass != 0) {
    sycl::free(usm_shared_buffer, ctxt);
    return USM_TEST_FAIL;
  }

  // memset 3 bytes on aligned address
  device_memset_invoke<KernelTestMemsetUSM1>(deviceQueue, usm_shared_buffer,
                                             0xCC, 3);
  memset(host_ref_buffer, 0xCC, 3);
  usm_memcheck_pass = memcmp(host_ref_buffer, usm_shared_buffer, 32);
  if (usm_memcheck_pass != 0) {
    sycl::free(usm_shared_buffer, ctxt);
    return USM_TEST_FAIL;
  }

  // memset 15 bytes on unaligned address
  device_memset_invoke<KernelTestMemsetUSM2>(deviceQueue, usm_shared_buffer + 1,
                                             0xAA, 21);
  memset(host_ref_buffer + 1, 0xAA, 21);
  usm_memcheck_pass = memcmp(host_ref_buffer, usm_shared_buffer, 32);
  if (usm_memcheck_pass != 0) {
    sycl::free(usm_shared_buffer, ctxt);
    return USM_TEST_FAIL;
  }

  // memset 2 bytes on unaligned address
  device_memset_invoke<KernelTestMemsetUSM3>(deviceQueue,
                                             usm_shared_buffer + 13, 0xBB, 2);
  memset(host_ref_buffer + 13, 0xBB, 2);
  usm_memcheck_pass = memcmp(host_ref_buffer, usm_shared_buffer, 32);
  if (usm_memcheck_pass != 0) {
    sycl::free(usm_shared_buffer, ctxt);
    return USM_TEST_FAIL;
  }

  sycl::free(usm_shared_buffer, ctxt);
  return USM_TEST_PASS;
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
        results_acc[0] = memcmp(
            str1_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
            str1_acc.get_multi_ptr<sycl::access::decorated::no>().get(), 1);
        results_acc[1] = memcmp(
            str2_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
            str1_acc.get_multi_ptr<sycl::access::decorated::no>().get(), 1);
        results_acc[2] = memcmp(
            str3_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
            str2_acc.get_multi_ptr<sycl::access::decorated::no>().get(), 2);
        results_acc[3] = memcmp(
            str3_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
            str4_acc.get_multi_ptr<sycl::access::decorated::no>().get(), 4);
        results_acc[4] = memcmp(
            str4_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
            str5_acc.get_multi_ptr<sycl::access::decorated::no>().get(), 2);
        results_acc[5] = memcmp(
            str6_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
            str7_acc.get_multi_ptr<sycl::access::decorated::no>().get(), 6);
        results_acc[6] = memcmp(
            str5_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
            str6_acc.get_multi_ptr<sycl::access::decorated::no>().get(), 0);
        results_acc[7] =
            memcmp(str7_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                   str8_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
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

      sycl::local_accessor<char, 1> local_acc(sycl::range<1>(16), cgh);

      sycl::accessor<char, 1, sycl::access::mode::write,
                     sycl::access::target::device,
                     sycl::access::placeholder::false_t>
          dst_acc(buffer2, cgh);

      sycl::accessor<char, 1, sycl::access::mode::write,
                     sycl::access::target::device,
                     sycl::access::placeholder::false_t>
          dst1_acc(buffer3, cgh);
      cgh.parallel_for<class KernelTestMemcpyAddrSpace>(
          sycl::nd_range<1>{16, 16}, [=](sycl::nd_item<1>) {
            // memcpy from constant buffer to local buffer
            memcpy(local_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                   src_acc.get_pointer(), 8);
            for (size_t idx = 0; idx < 7; ++idx)
              local_acc[idx] += 1;
            // memcpy from local buffer to global buffer
            memcpy(dst_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                   local_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                   8);
            char device_buf[16];
            // memcpy from constant buffer to private memory
            memcpy(device_buf, src_acc.get_pointer(), 8);
            for (size_t idx = 0; idx < 7; ++idx) {
              device_buf[idx] += 2;
              // memcpy from private to global buffer
              memcpy(
                  dst1_acc.get_multi_ptr<sycl::access::decorated::no>().get(),
                  device_buf, 8);
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
  sycl::device dev = deviceQueue.get_device();
  bool shared_usm_enabled = false;
  USM_TEST_RES usm_tres;
  if (dev.get_info<sycl::info::device::usm_shared_allocations>())
    shared_usm_enabled = true;
  success = kernel_test_memcpy(deviceQueue);
  if (shared_usm_enabled) {
    usm_tres = kernel_test_memcpy_usm(deviceQueue);
    if (usm_tres == USM_ALLOC_FAIL)
      std::cout << "USM shared memory alloc failed, USM tests skipped"
                << std::endl;
    else if (usm_tres == USM_TEST_FAIL)
      success = false;
  }
  assert(((void)"memcpy test failed!", success));
  success = kernel_test_memset(deviceQueue);
  if (shared_usm_enabled) {
    usm_tres = kernel_test_memset_usm(deviceQueue);
    if (usm_tres == USM_ALLOC_FAIL)
      std::cout << "USM shared memory alloc failed, USM tests skipped"
                << std::endl;
    else if (usm_tres == USM_TEST_FAIL)
      success = false;
  }
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
