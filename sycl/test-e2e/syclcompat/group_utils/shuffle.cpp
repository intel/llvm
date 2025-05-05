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
 *  shuffle.cpp
 *
 *  Description:
 *    Group shuffle API tests
 **************************************************************************/

// ===------- shuffle.cpp -------------------- *- C++ -* ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>

#include <sycl/detail/core.hpp>

#include <syclcompat/device.hpp>
#include <syclcompat/group_utils.hpp>

int expect1[128] = {
    2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,
    17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
    32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,
    47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,
    62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,
    77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,
    92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106,
    107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
    122, 123, 124, 125, 126, 127, 0,   0};

int expect2[128] = {
    2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,
    17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
    32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,
    47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,
    62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,
    77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,
    92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106,
    107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
    122, 123, 124, 125, 126, 127, 0,   1};

int expect3[513] = {
    0,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,
    14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,
    29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
    44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,
    59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,
    74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,
    89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103,
    104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
    119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
    134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148,
    149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163,
    164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178,
    179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193,
    194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208,
    209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
    224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238,
    239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253,
    254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268,
    269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283,
    284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298,
    299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313,
    314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328,
    329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343,
    344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358,
    359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373,
    374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388,
    389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403,
    404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418,
    419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433,
    434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448,
    449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463,
    464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478,
    479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493,
    494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508,
    509, 510, 511};

int expect4[513] = {
    1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,
    16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,
    31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,
    46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,
    61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,
    76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
    91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105,
    106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
    121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
    136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150,
    151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165,
    166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180,
    181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195,
    196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
    211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225,
    226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240,
    241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
    256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270,
    271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
    286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300,
    301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315,
    316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330,
    331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345,
    346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360,
    361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375,
    376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390,
    391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405,
    406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420,
    421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435,
    436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450,
    451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465,
    466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480,
    481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495,
    496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510,
    511, 510, 0};

void BlockShuffleKernel1(int *input, int *output,
                         const sycl::nd_item<3> &item_ct1,
                         uint8_t *temp_storage) {

  typedef syclcompat::group::group_shuffle<int, 128> BS;

  BS(temp_storage)
      .select(item_ct1, input[item_ct1.get_local_id(2)],
              output[item_ct1.get_local_id(2)], 2);
}

void BlockShuffleKernel2(int *input, int *output,
                         const sycl::nd_item<3> &item_ct1,
                         uint8_t *temp_storage) {
  typedef syclcompat::group::group_shuffle<int, 128> BS;

  BS(temp_storage)
      .select2(item_ct1, input[item_ct1.get_local_id(2)],
               output[item_ct1.get_local_id(2)], 2);
}

void BlockShuffleKernel3(int *input, int *output, int *extra,
                         const sycl::nd_item<3> &item_ct1,
                         uint8_t *temp_storage) {
  typedef syclcompat::group::group_shuffle<int, 128> BS;

  BS(temp_storage)
      .shuffle_right(
          item_ct1,
          *reinterpret_cast<int(*)[4]>(input + item_ct1.get_local_id(2) * 4),
          *reinterpret_cast<int(*)[4]>(output + item_ct1.get_local_id(2) * 4),
          *extra);
}

void BlockShuffleKernel4(int *input, int *output, int *extra,
                         const sycl::nd_item<3> &item_ct1,
                         uint8_t *temp_storage) {
  typedef syclcompat::group::group_shuffle<int, 128> BS;

  BS(temp_storage)
      .shuffle_left(
          item_ct1,
          *reinterpret_cast<int(*)[4]>(input + item_ct1.get_local_id(2) * 4),
          *reinterpret_cast<int(*)[4]>(output + item_ct1.get_local_id(2) * 4),
          *extra);
}

int main() {
  syclcompat::device_ext &dev_ct1 = syclcompat::get_current_device();
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  int *input1, *output1;
  int *input4, *output4;
  int *extra;
  int host_input1[128];
  int host_output1[128];
  int host_input4[128 * 4];
  int host_output4[128 * 4];
  int host_extra = 0;
  input1 = sycl::malloc_device<int>(128, q_ct1);
  output1 = sycl::malloc_device<int>(128, q_ct1);
  input4 = (int *)sycl::malloc_device(sizeof(int) * 128 * 4, q_ct1);
  output4 = (int *)sycl::malloc_device(sizeof(int) * 128 * 4, q_ct1);
  extra = sycl::malloc_device<int>(1, q_ct1);

  for (int i = 0; i < 128; i++) {
    host_input1[i] = i;
    host_output1[i] = 0;
  }

  for (int i = 0; i < 128 * 4; i++) {
    host_input4[i] = i;
    host_output4[i] = 0;
  }

  q_ct1.memcpy(input1, host_input1, sizeof(int) * 128);
  q_ct1.memcpy(input4, host_input4, sizeof(int) * 128 * 4);
  q_ct1.memcpy(output1, host_output1, sizeof(int) * 128);
  q_ct1.memcpy(output4, host_output4, sizeof(int) * 128 * 4);
  q_ct1.memcpy(extra, &host_extra, sizeof(int));

  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint8_t, 1> temp_storage_acc(
        syclcompat::group::group_shuffle<int, 128>::get_local_memory_size(
            sycl::range<3>(1, 1, 128).size()),
        cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item_ct1) {
          BlockShuffleKernel1(input1, output1, item_ct1, &temp_storage_acc[0]);
        });
  });

  q_ct1.memcpy(host_output1, output1, sizeof(int) * 128).wait();
  dev_ct1.queues_wait_and_throw();
  for (int i = 0; i < 128; i++) {
    if (host_output1[i] != expect1[i]) {
      std::cout << "test 1 failed" << std::endl;
      exit(-1);
    }
  }

  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint8_t, 1> temp_storage_acc(
        syclcompat::group::group_shuffle<int, 128>::get_local_memory_size(
            sycl::range<3>(1, 1, 128).size()),
        cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item_ct1) {
          BlockShuffleKernel2(input1, output1, item_ct1, &temp_storage_acc[0]);
        });
  });

  q_ct1.memcpy(host_output1, output1, sizeof(int) * 128).wait();
  dev_ct1.queues_wait_and_throw();
  for (int i = 0; i < 128; i++) {
    if (host_output1[i] != expect2[i]) {
      std::cout << "test 2 failed" << std::endl;
      exit(-1);
    }
  }

  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint8_t, 1> temp_storage_acc(
        syclcompat::group::group_shuffle<int, 128>::get_local_memory_size(
            sycl::range<3>(1, 1, 128).size()),
        cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item_ct1) {
          BlockShuffleKernel3(input4, output4, extra, item_ct1,
                              &temp_storage_acc[0]);
        });
  });

  q_ct1.memcpy(host_output4, output4, sizeof(int) * 128 * 4);
  q_ct1.memcpy(&host_extra, extra, sizeof(int)).wait();
  dev_ct1.queues_wait_and_throw();

  for (int i = 0; i < 128 * 4; i++) {
    if (host_output4[i] != expect3[i]) {
      std::cout << "test 3 failed" << std::endl;
      exit(-1);
    }
  }
  if (host_extra != expect3[512]) {
    std::cout << "test 3 failed" << std::endl;
    exit(-1);
  }

  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint8_t, 1> temp_storage_acc(
        syclcompat::group::group_shuffle<int, 128>::get_local_memory_size(
            sycl::range<3>(1, 1, 128).size()),
        cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item_ct1) {
          BlockShuffleKernel4(input4, output4, extra, item_ct1,
                              &temp_storage_acc[0]);
        });
  });

  q_ct1.memcpy(host_output4, output4, sizeof(int) * 128 * 4);
  q_ct1.memcpy(&host_extra, extra, sizeof(int)).wait();
  dev_ct1.queues_wait_and_throw();
  for (int i = 0; i < 128 * 4; i++) {
    if (host_output4[i] != expect4[i]) {
      std::cout << "test 4 failed" << std::endl;
      exit(-1);
    }
  }
  if (host_extra != expect4[512]) {
    std::cout << "test 4 failed" << std::endl;
    exit(-1);
  }
  std::cout << "test pass" << std::endl;
  return 0;
};
