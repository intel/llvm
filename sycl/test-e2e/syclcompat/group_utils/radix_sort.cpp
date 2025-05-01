/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Part of the LLVM Project, under the Apache License v2.0 with LLVM
 *  Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCLcompat API
 *
 *  radix_sort.cpp
 *
 *  Description:
 *    Group radix sort API tests
 **************************************************************************/

// ===------- radix_sort.cpp-------------------- *- C++ -* ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: spirv-backend
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/17400

#include <iostream>

#include <sycl/detail/core.hpp>

#include <syclcompat/device.hpp>
#include <syclcompat/group_utils.hpp>
#include <syclcompat/memory.hpp>

void Sort(int *data, const sycl::nd_item<3> &item_ct1, uint8_t *temp_storage) {

  using BlockRadixSort = syclcompat::group::group_radix_sort<int, 4>;

  int thread_keys[4];
  syclcompat::group::load_direct_blocked(item_ct1, data, thread_keys);
  BlockRadixSort(temp_storage).sort(item_ct1, thread_keys);
  syclcompat::group::store_direct_blocked(item_ct1, data, thread_keys);
}

void SortDescending(int *data, const sycl::nd_item<3> &item_ct1,
                    uint8_t *temp_storage) {

  using BlockRadixSort = syclcompat::group::group_radix_sort<int, 4>;

  int thread_keys[4];
  syclcompat::group::load_direct_blocked(item_ct1, data, thread_keys);
  BlockRadixSort(temp_storage).sort_descending(item_ct1, thread_keys);
  syclcompat::group::store_direct_blocked(item_ct1, data, thread_keys);
}

void SortBlockedToStriped(int *data, const sycl::nd_item<3> &item_ct1,
                          uint8_t *load_temp_storage,
                          uint8_t *store_temp_storage, uint8_t *temp_storage) {
  using BlockLoadT = syclcompat::group::group_load<int, 4>;
  using BlockStoreT = syclcompat::group::group_store<int, 4>;
  using BlockRadixSort = syclcompat::group::group_radix_sort<int, 4>;

  int thread_keys[4];
  BlockLoadT(load_temp_storage).load(item_ct1, data, thread_keys);
  BlockRadixSort(temp_storage).sort_blocked_to_striped(item_ct1, thread_keys);
  BlockStoreT(store_temp_storage).store(item_ct1, data, thread_keys);
}

void SortDescendingBlockedToStriped(int *data, const sycl::nd_item<3> &item_ct1,
                                    uint8_t *load_temp_storage,
                                    uint8_t *store_temp_storage,
                                    uint8_t *temp_storage) {
  using BlockLoadT = syclcompat::group::group_load<
      int, 4, syclcompat::group::group_load_algorithm::blocked>;
  using BlockStoreT = syclcompat::group::group_store<
      int, 4, syclcompat::group::group_store_algorithm::blocked>;
  using BlockRadixSort = syclcompat::group::group_radix_sort<int, 4>;

  int thread_keys[4];
  BlockLoadT(load_temp_storage).load(item_ct1, data, thread_keys);
  BlockRadixSort(temp_storage)
      .sort_descending_blocked_to_striped(item_ct1, thread_keys);
  BlockStoreT(store_temp_storage).store(item_ct1, data, thread_keys);
}

void SortBit(int *data, const sycl::nd_item<3> &item_ct1,
             uint8_t *temp_storage) {

  using BlockRadixSort = syclcompat::group::group_radix_sort<int, 4>;

  int thread_keys[4];
  syclcompat::group::load_direct_blocked(item_ct1, data, thread_keys);
  BlockRadixSort(temp_storage).sort(item_ct1, thread_keys, 4, 16);
  syclcompat::group::store_direct_blocked(item_ct1, data, thread_keys);
}

void SortDescendingBit(int *data, const sycl::nd_item<3> &item_ct1,
                       uint8_t *temp_storage) {

  using BlockRadixSort = syclcompat::group::group_radix_sort<int, 4>;

  int thread_keys[4];
  syclcompat::group::load_direct_blocked(item_ct1, data, thread_keys);
  BlockRadixSort(temp_storage).sort_descending(item_ct1, thread_keys, 4, 16);
  syclcompat::group::store_direct_blocked(item_ct1, data, thread_keys);
}

void SortBlockedToStripedBit(int *data, const sycl::nd_item<3> &item_ct1,
                             uint8_t *temp_storage) {

  using BlockRadixSort = syclcompat::group::group_radix_sort<int, 4>;

  int thread_keys[4];
  syclcompat::group::load_direct_blocked(item_ct1, data, thread_keys);

  BlockRadixSort(temp_storage)
      .sort_blocked_to_striped(item_ct1, thread_keys, 4, 16);
  syclcompat::group::store_direct_blocked(item_ct1, data, thread_keys);
}

void SortDescendingBlockedToStripedBit(int *data,
                                       const sycl::nd_item<3> &item_ct1,
                                       uint8_t *temp_storage) {

  using BlockRadixSort = syclcompat::group::group_radix_sort<int, 4>;

  int thread_keys[4];
  syclcompat::group::load_direct_blocked(item_ct1, data, thread_keys);

  BlockRadixSort(temp_storage)
      .sort_descending_blocked_to_striped(item_ct1, thread_keys, 4, 16);
  syclcompat::group::store_direct_blocked(item_ct1, data, thread_keys);
}

template <typename T, int N> void print_array(T (&arr)[N]) {
  for (int i = 0; i < N; ++i)
    printf("%d%c", arr[i], (i == N - 1 ? '\n' : ','));
}

bool test_sort() {
  syclcompat::device_ext &dev_ct1 = syclcompat::get_current_device();
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  int data[512] = {0}, *d_data = nullptr;
  d_data = sycl::malloc_device<int>(512, q_ct1);
  for (int i = 0, x = 0, y = 511; i < 128; ++i) {
    data[i * 4 + 0] = x++;
    data[i * 4 + 1] = y--;
    data[i * 4 + 2] = x++;
    data[i * 4 + 3] = y--;
  }
  q_ct1.memcpy(d_data, data, sizeof(data)).wait();

  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint8_t, 1> temp_storage_acc(
        syclcompat::group::group_radix_sort<int, 4>::get_local_memory_size(
            sycl::range<3>(1, 1, 128).size()),
        cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item_ct1) {
          Sort(d_data, item_ct1, &temp_storage_acc[0]);
        });
  });
  dev_ct1.queues_wait_and_throw();
  q_ct1.memcpy(data, d_data, sizeof(data)).wait();
  syclcompat::wait_and_free(d_data, q_ct1);
  for (int i = 0; i < 512; ++i)
    if (data[i] != i) {
      printf("test_sort failed\n");
      print_array(data);
      return false;
    }
  printf("test_sort pass\n");
  return true;
}

bool test_sort_descending() {
  syclcompat::device_ext &dev_ct1 = syclcompat::get_current_device();
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  int data[512] = {0}, *d_data = nullptr;
  d_data = sycl::malloc_device<int>(512, q_ct1);
  for (int i = 0, x = 0, y = 511; i < 128; ++i) {
    data[i * 4 + 0] = x++;
    data[i * 4 + 1] = y--;
    data[i * 4 + 2] = x++;
    data[i * 4 + 3] = y--;
  }
  q_ct1.memcpy(d_data, data, sizeof(data)).wait();

  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint8_t, 1> temp_storage_acc(
        syclcompat::group::group_radix_sort<int, 4>::get_local_memory_size(
            sycl::range<3>(1, 1, 128).size()),
        cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item_ct1) {
          SortDescending(d_data, item_ct1, &temp_storage_acc[0]);
        });
  });
  dev_ct1.queues_wait_and_throw();
  q_ct1.memcpy(data, d_data, sizeof(data)).wait();
  syclcompat::wait_and_free(d_data, q_ct1);
  for (int i = 0; i < 512; ++i)
    if (data[i] != 511 - i) {
      printf("test_sort_descending failed\n");
      print_array(data);
      return false;
    }
  printf("test_sort_descending pass\n");
  return true;
}

bool test_sort_blocked_to_striped() {
  syclcompat::device_ext &dev_ct1 = syclcompat::get_current_device();
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  int data[512] = {0}, *d_data = nullptr;
  d_data = sycl::malloc_device<int>(512, q_ct1);
  for (int i = 0, x = 0, y = 511; i < 128; ++i) {
    data[i * 4 + 0] = x++;
    data[i * 4 + 1] = y--;
    data[i * 4 + 2] = x++;
    data[i * 4 + 3] = y--;
  }
  q_ct1.memcpy(d_data, data, sizeof(data)).wait();

  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint8_t, 1> load_temp_storage_acc(
        syclcompat::group::group_load<int, 4>::get_local_memory_size(
            sycl::range<3>(1, 1, 128).size()),
        cgh);
    sycl::local_accessor<uint8_t, 1> store_temp_storage_acc(
        syclcompat::group::group_store<int, 4>::get_local_memory_size(
            sycl::range<3>(1, 1, 128).size()),
        cgh);
    sycl::local_accessor<uint8_t, 1> temp_storage_acc(
        syclcompat::group::group_radix_sort<int, 4>::get_local_memory_size(
            sycl::range<3>(1, 1, 128).size()),
        cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item_ct1) {
          SortBlockedToStriped(d_data, item_ct1, &load_temp_storage_acc[0],
                               &store_temp_storage_acc[0],
                               &temp_storage_acc[0]);
        });
  });
  dev_ct1.queues_wait_and_throw();
  q_ct1.memcpy(data, d_data, sizeof(data)).wait();
  syclcompat::wait_and_free(d_data, q_ct1);
  int expected[512];
  for (int i = 0; i < 128; ++i) {
    expected[4 * i + 0] = i;
    expected[4 * i + 1] = i + 1 * 128;
    expected[4 * i + 2] = i + 2 * 128;
    expected[4 * i + 3] = i + 3 * 128;
  }
  for (int i = 0; i < 512; ++i)
    if (data[i] != expected[i]) {
      printf("test_sort_blocked_to_striped failed\n");
      print_array(data);
      return false;
    }
  printf("test_sort_blocked_to_striped pass\n");
  return true;
}

bool test_sort_descending_blocked_to_striped() {
  syclcompat::device_ext &dev_ct1 = syclcompat::get_current_device();
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  int data[512] = {0}, *d_data = nullptr;
  d_data = sycl::malloc_device<int>(512, q_ct1);
  for (int i = 0, x = 0, y = 511; i < 128; ++i) {
    data[i * 4 + 0] = x++;
    data[i * 4 + 1] = y--;
    data[i * 4 + 2] = x++;
    data[i * 4 + 3] = y--;
  }
  q_ct1.memcpy(d_data, data, sizeof(data)).wait();

  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint8_t, 1> load_temp_storage_acc(
        syclcompat::group::group_load<int, 4>::get_local_memory_size(
            sycl::range<3>(1, 1, 128).size()),
        cgh);
    sycl::local_accessor<uint8_t, 1> store_temp_storage_acc(
        syclcompat::group::group_store<int, 4>::get_local_memory_size(
            sycl::range<3>(1, 1, 128).size()),
        cgh);
    sycl::local_accessor<uint8_t, 1> temp_storage_acc(
        syclcompat::group::group_radix_sort<int, 4>::get_local_memory_size(
            sycl::range<3>(1, 1, 128).size()),
        cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item_ct1) {
          SortDescendingBlockedToStriped(
              d_data, item_ct1, &load_temp_storage_acc[0],
              &store_temp_storage_acc[0], &temp_storage_acc[0]);
        });
  });
  dev_ct1.queues_wait_and_throw();
  q_ct1.memcpy(data, d_data, sizeof(data)).wait();
  syclcompat::wait_and_free(d_data, q_ct1);
  int expected[512];
  for (int i = 0; i < 128; ++i) {
    expected[4 * i + 0] = 511 - i;
    expected[4 * i + 1] = 511 - i - 1 * 128;
    expected[4 * i + 2] = 511 - i - 2 * 128;
    expected[4 * i + 3] = 511 - i - 3 * 128;
  }
  for (int i = 0; i < 512; ++i)
    if (data[i] != expected[i]) {
      printf("test_sort_descending_blocked_to_striped failed\n");
      print_array(data);
      return false;
    }
  printf("test_sort_descending_blocked_to_striped pass\n");
  return true;
}

bool test_sort_bit() {
  syclcompat::device_ext &dev_ct1 = syclcompat::get_current_device();
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  int data[512] = {0}, *d_data = nullptr;
  int expected[512] = {
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,
      15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
      30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,
      45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
      60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,
      75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
      90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104,
      105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
      120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
      135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
      150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
      165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
      180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
      195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
      210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224,
      225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
      240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254,
      255, 271, 270, 269, 268, 267, 266, 265, 264, 263, 262, 261, 260, 259, 258,
      257, 256, 287, 286, 285, 284, 283, 282, 281, 280, 279, 278, 277, 276, 275,
      274, 273, 272, 303, 302, 301, 300, 299, 298, 297, 296, 295, 294, 293, 292,
      291, 290, 289, 288, 319, 318, 317, 316, 315, 314, 313, 312, 311, 310, 309,
      308, 307, 306, 305, 304, 335, 334, 333, 332, 331, 330, 329, 328, 327, 326,
      325, 324, 323, 322, 321, 320, 351, 350, 349, 348, 347, 346, 345, 344, 343,
      342, 341, 340, 339, 338, 337, 336, 367, 366, 365, 364, 363, 362, 361, 360,
      359, 358, 357, 356, 355, 354, 353, 352, 383, 382, 381, 380, 379, 378, 377,
      376, 375, 374, 373, 372, 371, 370, 369, 368, 399, 398, 397, 396, 395, 394,
      393, 392, 391, 390, 389, 388, 387, 386, 385, 384, 415, 414, 413, 412, 411,
      410, 409, 408, 407, 406, 405, 404, 403, 402, 401, 400, 431, 430, 429, 428,
      427, 426, 425, 424, 423, 422, 421, 420, 419, 418, 417, 416, 447, 446, 445,
      444, 443, 442, 441, 440, 439, 438, 437, 436, 435, 434, 433, 432, 463, 462,
      461, 460, 459, 458, 457, 456, 455, 454, 453, 452, 451, 450, 449, 448, 479,
      478, 477, 476, 475, 474, 473, 472, 471, 470, 469, 468, 467, 466, 465, 464,
      495, 494, 493, 492, 491, 490, 489, 488, 487, 486, 485, 484, 483, 482, 481,
      480, 511, 510, 509, 508, 507, 506, 505, 504, 503, 502, 501, 500, 499, 498,
      497, 496};
  d_data = sycl::malloc_device<int>(512, q_ct1);
  for (int i = 0, x = 0, y = 511; i < 128; ++i) {
    data[i * 4 + 0] = x++;
    data[i * 4 + 1] = y--;
    data[i * 4 + 2] = x++;
    data[i * 4 + 3] = y--;
  }
  q_ct1.memcpy(d_data, data, sizeof(data)).wait();

  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint8_t, 1> temp_storage_acc(
        syclcompat::group::group_radix_sort<int, 4>::get_local_memory_size(
            sycl::range<3>(1, 1, 128).size()),
        cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item_ct1) {
          SortBit(d_data, item_ct1, &temp_storage_acc[0]);
        });
  });
  dev_ct1.queues_wait_and_throw();
  q_ct1.memcpy(data, d_data, sizeof(data)).wait();
  syclcompat::wait_and_free(d_data, q_ct1);
  for (int i = 0; i < 512; ++i)
    if (data[i] != expected[i]) {
      printf("test_sort_bit failed\n");
      print_array(data);
      return false;
    }
  printf("test_sort_bit pass\n");
  return true;
}

bool test_sort_descending_bit() {
  syclcompat::device_ext &dev_ct1 = syclcompat::get_current_device();
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  int data[512] = {0}, *d_data = nullptr;
  int expected[512] = {
      511, 510, 509, 508, 507, 506, 505, 504, 503, 502, 501, 500, 499, 498, 497,
      496, 495, 494, 493, 492, 491, 490, 489, 488, 487, 486, 485, 484, 483, 482,
      481, 480, 479, 478, 477, 476, 475, 474, 473, 472, 471, 470, 469, 468, 467,
      466, 465, 464, 463, 462, 461, 460, 459, 458, 457, 456, 455, 454, 453, 452,
      451, 450, 449, 448, 447, 446, 445, 444, 443, 442, 441, 440, 439, 438, 437,
      436, 435, 434, 433, 432, 431, 430, 429, 428, 427, 426, 425, 424, 423, 422,
      421, 420, 419, 418, 417, 416, 415, 414, 413, 412, 411, 410, 409, 408, 407,
      406, 405, 404, 403, 402, 401, 400, 399, 398, 397, 396, 395, 394, 393, 392,
      391, 390, 389, 388, 387, 386, 385, 384, 383, 382, 381, 380, 379, 378, 377,
      376, 375, 374, 373, 372, 371, 370, 369, 368, 367, 366, 365, 364, 363, 362,
      361, 360, 359, 358, 357, 356, 355, 354, 353, 352, 351, 350, 349, 348, 347,
      346, 345, 344, 343, 342, 341, 340, 339, 338, 337, 336, 335, 334, 333, 332,
      331, 330, 329, 328, 327, 326, 325, 324, 323, 322, 321, 320, 319, 318, 317,
      316, 315, 314, 313, 312, 311, 310, 309, 308, 307, 306, 305, 304, 303, 302,
      301, 300, 299, 298, 297, 296, 295, 294, 293, 292, 291, 290, 289, 288, 287,
      286, 285, 284, 283, 282, 281, 280, 279, 278, 277, 276, 275, 274, 273, 272,
      271, 270, 269, 268, 267, 266, 265, 264, 263, 262, 261, 260, 259, 258, 257,
      256, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253,
      254, 255, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236,
      237, 238, 239, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
      220, 221, 222, 223, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202,
      203, 204, 205, 206, 207, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185,
      186, 187, 188, 189, 190, 191, 160, 161, 162, 163, 164, 165, 166, 167, 168,
      169, 170, 171, 172, 173, 174, 175, 144, 145, 146, 147, 148, 149, 150, 151,
      152, 153, 154, 155, 156, 157, 158, 159, 128, 129, 130, 131, 132, 133, 134,
      135, 136, 137, 138, 139, 140, 141, 142, 143, 112, 113, 114, 115, 116, 117,
      118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 96,  97,  98,  99,  100,
      101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 80,  81,  82,  83,
      84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  64,  65,  66,
      67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  48,  49,
      50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  32,
      33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
      16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,
      31,  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,
      14,  15};
  d_data = sycl::malloc_device<int>(512, q_ct1);
  for (int i = 0, x = 0, y = 511; i < 128; ++i) {
    data[i * 4 + 0] = x++;
    data[i * 4 + 1] = y--;
    data[i * 4 + 2] = x++;
    data[i * 4 + 3] = y--;
  }
  q_ct1.memcpy(d_data, data, sizeof(data)).wait();

  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint8_t, 1> temp_storage_acc(
        syclcompat::group::group_radix_sort<int, 4>::get_local_memory_size(
            sycl::range<3>(1, 1, 128).size()),
        cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item_ct1) {
          SortDescendingBit(d_data, item_ct1, &temp_storage_acc[0]);
        });
  });
  dev_ct1.queues_wait_and_throw();
  q_ct1.memcpy(data, d_data, sizeof(data)).wait();
  syclcompat::wait_and_free(d_data, q_ct1);
  for (int i = 0; i < 512; ++i)
    if (data[i] != expected[i]) {
      printf("test_sort_descending_bit failed\n");
      print_array(data);
      return false;
    }
  printf("test_sort_descending_bit pass\n");
  return true;
}

bool test_sort_blocked_to_striped_bit() {
  syclcompat::device_ext &dev_ct1 = syclcompat::get_current_device();
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  int data[512] = {0}, *d_data = nullptr;
  d_data = sycl::malloc_device<int>(512, q_ct1);
  for (int i = 0, x = 0, y = 511; i < 128; ++i) {
    data[i * 4 + 0] = x++;
    data[i * 4 + 1] = y--;
    data[i * 4 + 2] = x++;
    data[i * 4 + 3] = y--;
  }
  q_ct1.memcpy(d_data, data, sizeof(data)).wait();

  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint8_t, 1> temp_storage_acc(
        syclcompat::group::group_radix_sort<int, 4>::get_local_memory_size(
            sycl::range<3>(1, 1, 128).size()),
        cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item_ct1) {
          SortBlockedToStripedBit(d_data, item_ct1, &temp_storage_acc[0]);
        });
  });
  dev_ct1.queues_wait_and_throw();
  q_ct1.memcpy(data, d_data, sizeof(data)).wait();
  syclcompat::wait_and_free(d_data, q_ct1);
  int expected[512] = {
      0,   128, 271, 399, 1,   129, 270, 398, 2,   130, 269, 397, 3,   131, 268,
      396, 4,   132, 267, 395, 5,   133, 266, 394, 6,   134, 265, 393, 7,   135,
      264, 392, 8,   136, 263, 391, 9,   137, 262, 390, 10,  138, 261, 389, 11,
      139, 260, 388, 12,  140, 259, 387, 13,  141, 258, 386, 14,  142, 257, 385,
      15,  143, 256, 384, 16,  144, 287, 415, 17,  145, 286, 414, 18,  146, 285,
      413, 19,  147, 284, 412, 20,  148, 283, 411, 21,  149, 282, 410, 22,  150,
      281, 409, 23,  151, 280, 408, 24,  152, 279, 407, 25,  153, 278, 406, 26,
      154, 277, 405, 27,  155, 276, 404, 28,  156, 275, 403, 29,  157, 274, 402,
      30,  158, 273, 401, 31,  159, 272, 400, 32,  160, 303, 431, 33,  161, 302,
      430, 34,  162, 301, 429, 35,  163, 300, 428, 36,  164, 299, 427, 37,  165,
      298, 426, 38,  166, 297, 425, 39,  167, 296, 424, 40,  168, 295, 423, 41,
      169, 294, 422, 42,  170, 293, 421, 43,  171, 292, 420, 44,  172, 291, 419,
      45,  173, 290, 418, 46,  174, 289, 417, 47,  175, 288, 416, 48,  176, 319,
      447, 49,  177, 318, 446, 50,  178, 317, 445, 51,  179, 316, 444, 52,  180,
      315, 443, 53,  181, 314, 442, 54,  182, 313, 441, 55,  183, 312, 440, 56,
      184, 311, 439, 57,  185, 310, 438, 58,  186, 309, 437, 59,  187, 308, 436,
      60,  188, 307, 435, 61,  189, 306, 434, 62,  190, 305, 433, 63,  191, 304,
      432, 64,  192, 335, 463, 65,  193, 334, 462, 66,  194, 333, 461, 67,  195,
      332, 460, 68,  196, 331, 459, 69,  197, 330, 458, 70,  198, 329, 457, 71,
      199, 328, 456, 72,  200, 327, 455, 73,  201, 326, 454, 74,  202, 325, 453,
      75,  203, 324, 452, 76,  204, 323, 451, 77,  205, 322, 450, 78,  206, 321,
      449, 79,  207, 320, 448, 80,  208, 351, 479, 81,  209, 350, 478, 82,  210,
      349, 477, 83,  211, 348, 476, 84,  212, 347, 475, 85,  213, 346, 474, 86,
      214, 345, 473, 87,  215, 344, 472, 88,  216, 343, 471, 89,  217, 342, 470,
      90,  218, 341, 469, 91,  219, 340, 468, 92,  220, 339, 467, 93,  221, 338,
      466, 94,  222, 337, 465, 95,  223, 336, 464, 96,  224, 367, 495, 97,  225,
      366, 494, 98,  226, 365, 493, 99,  227, 364, 492, 100, 228, 363, 491, 101,
      229, 362, 490, 102, 230, 361, 489, 103, 231, 360, 488, 104, 232, 359, 487,
      105, 233, 358, 486, 106, 234, 357, 485, 107, 235, 356, 484, 108, 236, 355,
      483, 109, 237, 354, 482, 110, 238, 353, 481, 111, 239, 352, 480, 112, 240,
      383, 511, 113, 241, 382, 510, 114, 242, 381, 509, 115, 243, 380, 508, 116,
      244, 379, 507, 117, 245, 378, 506, 118, 246, 377, 505, 119, 247, 376, 504,
      120, 248, 375, 503, 121, 249, 374, 502, 122, 250, 373, 501, 123, 251, 372,
      500, 124, 252, 371, 499, 125, 253, 370, 498, 126, 254, 369, 497, 127, 255,
      368, 496};
  for (int i = 0; i < 512; ++i)
    if (data[i] != expected[i]) {
      printf("test_sort_blocked_to_striped_bit failed\n");
      print_array(data);
      return false;
    }
  printf("test_sort_blocked_to_striped_bit pass\n");
  return true;
}

bool test_sort_descending_blocked_to_striped_bit() {
  syclcompat::device_ext &dev_ct1 = syclcompat::get_current_device();
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  int data[512] = {0}, *d_data = nullptr;
  d_data = sycl::malloc_device<int>(512, q_ct1);
  for (int i = 0, x = 0, y = 511; i < 128; ++i) {
    data[i * 4 + 0] = x++;
    data[i * 4 + 1] = y--;
    data[i * 4 + 2] = x++;
    data[i * 4 + 3] = y--;
  }
  q_ct1.memcpy(d_data, data, sizeof(data)).wait();

  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint8_t, 1> temp_storage_acc(
        syclcompat::group::group_radix_sort<int, 4>::get_local_memory_size(
            sycl::range<3>(1, 1, 128).size()),
        cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item_ct1) {
          SortDescendingBlockedToStripedBit(d_data, item_ct1,
                                            &temp_storage_acc[0]);
        });
  });
  dev_ct1.queues_wait_and_throw();
  q_ct1.memcpy(data, d_data, sizeof(data)).wait();
  syclcompat::wait_and_free(d_data, q_ct1);
  int expected[512] = {
      511, 383, 240, 112, 510, 382, 241, 113, 509, 381, 242, 114, 508, 380, 243,
      115, 507, 379, 244, 116, 506, 378, 245, 117, 505, 377, 246, 118, 504, 376,
      247, 119, 503, 375, 248, 120, 502, 374, 249, 121, 501, 373, 250, 122, 500,
      372, 251, 123, 499, 371, 252, 124, 498, 370, 253, 125, 497, 369, 254, 126,
      496, 368, 255, 127, 495, 367, 224, 96,  494, 366, 225, 97,  493, 365, 226,
      98,  492, 364, 227, 99,  491, 363, 228, 100, 490, 362, 229, 101, 489, 361,
      230, 102, 488, 360, 231, 103, 487, 359, 232, 104, 486, 358, 233, 105, 485,
      357, 234, 106, 484, 356, 235, 107, 483, 355, 236, 108, 482, 354, 237, 109,
      481, 353, 238, 110, 480, 352, 239, 111, 479, 351, 208, 80,  478, 350, 209,
      81,  477, 349, 210, 82,  476, 348, 211, 83,  475, 347, 212, 84,  474, 346,
      213, 85,  473, 345, 214, 86,  472, 344, 215, 87,  471, 343, 216, 88,  470,
      342, 217, 89,  469, 341, 218, 90,  468, 340, 219, 91,  467, 339, 220, 92,
      466, 338, 221, 93,  465, 337, 222, 94,  464, 336, 223, 95,  463, 335, 192,
      64,  462, 334, 193, 65,  461, 333, 194, 66,  460, 332, 195, 67,  459, 331,
      196, 68,  458, 330, 197, 69,  457, 329, 198, 70,  456, 328, 199, 71,  455,
      327, 200, 72,  454, 326, 201, 73,  453, 325, 202, 74,  452, 324, 203, 75,
      451, 323, 204, 76,  450, 322, 205, 77,  449, 321, 206, 78,  448, 320, 207,
      79,  447, 319, 176, 48,  446, 318, 177, 49,  445, 317, 178, 50,  444, 316,
      179, 51,  443, 315, 180, 52,  442, 314, 181, 53,  441, 313, 182, 54,  440,
      312, 183, 55,  439, 311, 184, 56,  438, 310, 185, 57,  437, 309, 186, 58,
      436, 308, 187, 59,  435, 307, 188, 60,  434, 306, 189, 61,  433, 305, 190,
      62,  432, 304, 191, 63,  431, 303, 160, 32,  430, 302, 161, 33,  429, 301,
      162, 34,  428, 300, 163, 35,  427, 299, 164, 36,  426, 298, 165, 37,  425,
      297, 166, 38,  424, 296, 167, 39,  423, 295, 168, 40,  422, 294, 169, 41,
      421, 293, 170, 42,  420, 292, 171, 43,  419, 291, 172, 44,  418, 290, 173,
      45,  417, 289, 174, 46,  416, 288, 175, 47,  415, 287, 144, 16,  414, 286,
      145, 17,  413, 285, 146, 18,  412, 284, 147, 19,  411, 283, 148, 20,  410,
      282, 149, 21,  409, 281, 150, 22,  408, 280, 151, 23,  407, 279, 152, 24,
      406, 278, 153, 25,  405, 277, 154, 26,  404, 276, 155, 27,  403, 275, 156,
      28,  402, 274, 157, 29,  401, 273, 158, 30,  400, 272, 159, 31,  399, 271,
      128, 0,   398, 270, 129, 1,   397, 269, 130, 2,   396, 268, 131, 3,   395,
      267, 132, 4,   394, 266, 133, 5,   393, 265, 134, 6,   392, 264, 135, 7,
      391, 263, 136, 8,   390, 262, 137, 9,   389, 261, 138, 10,  388, 260, 139,
      11,  387, 259, 140, 12,  386, 258, 141, 13,  385, 257, 142, 14,  384, 256,
      143, 15};
  for (int i = 0; i < 512; ++i)
    if (data[i] != expected[i]) {
      printf("test_sort_descending_blocked_to_striped_bit failed\n");
      print_array(data);
      return false;
    }
  printf("test_sort_descending_blocked_to_striped_bit pass\n");
  return true;
}

int main() {
  return !(test_sort() && test_sort_descending() &&
           test_sort_blocked_to_striped() &&
           test_sort_descending_blocked_to_striped() && test_sort_bit() &&
           test_sort_descending_bit() && test_sort_blocked_to_striped_bit() &&
           test_sort_descending_blocked_to_striped_bit());
}
