// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// RUN: %clangxx -fsycl -fno-builtin -fsycl-device-lib-jit-link %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// UNSUPPORTED: cuda || hip

#include <cassert>
#include <iostream>
#include <sycl/sycl.hpp>

namespace s = sycl;
constexpr s::access::mode sycl_read = s::access::mode::read;
constexpr s::access::mode sycl_write = s::access::mode::write;

extern "C" {
unsigned __imf_vabs2(unsigned);
unsigned __imf_vabs4(unsigned);
unsigned __imf_vabsss2(unsigned);
unsigned __imf_vabsss4(unsigned);
unsigned __imf_vneg2(unsigned);
unsigned __imf_vneg4(unsigned);
unsigned __imf_vnegss2(unsigned);
unsigned __imf_vnegss4(unsigned);
unsigned __imf_vabsdiffs2(unsigned, unsigned);
unsigned __imf_vabsdiffs4(unsigned, unsigned);
unsigned __imf_vabsdiffu2(unsigned, unsigned);
unsigned __imf_vabsdiffu4(unsigned, unsigned);
unsigned __imf_vadd2(unsigned, unsigned);
unsigned __imf_vadd4(unsigned, unsigned);
unsigned __imf_vaddss2(unsigned, unsigned);
unsigned __imf_vaddss4(unsigned, unsigned);
unsigned __imf_vaddus2(unsigned, unsigned);
unsigned __imf_vaddus4(unsigned, unsigned);
unsigned __imf_vsub2(unsigned, unsigned);
unsigned __imf_vsub4(unsigned, unsigned);
unsigned __imf_vsubss2(unsigned, unsigned);
unsigned __imf_vsubss4(unsigned, unsigned);
unsigned __imf_vsubus2(unsigned, unsigned);
unsigned __imf_vsubus4(unsigned, unsigned);
unsigned __imf_vavgs2(unsigned, unsigned);
unsigned __imf_vavgs4(unsigned, unsigned);
unsigned __imf_vavgu2(unsigned, unsigned);
unsigned __imf_vavgu4(unsigned, unsigned);
unsigned __imf_vcmpgts2(unsigned, unsigned);
unsigned __imf_vcmpgts4(unsigned, unsigned);
unsigned __imf_vhaddu2(unsigned, unsigned);
unsigned __imf_vhaddu4(unsigned, unsigned);
unsigned __imf_vmaxs2(unsigned, unsigned);
unsigned __imf_vmaxs4(unsigned, unsigned);
unsigned __imf_vmaxu2(unsigned, unsigned);
unsigned __imf_vmaxu4(unsigned, unsigned);
unsigned __imf_vmins2(unsigned, unsigned);
unsigned __imf_vmins4(unsigned, unsigned);
unsigned __imf_vminu2(unsigned, unsigned);
unsigned __imf_vminu4(unsigned, unsigned);
unsigned __imf_vsads2(unsigned, unsigned);
unsigned __imf_vsads4(unsigned, unsigned);
unsigned __imf_vsadu2(unsigned, unsigned);
unsigned __imf_vsadu4(unsigned, unsigned);
unsigned __imf_vcmpeq2(unsigned, unsigned);
unsigned __imf_vcmpeq4(unsigned, unsigned);
unsigned __imf_vcmpne2(unsigned, unsigned);
unsigned __imf_vcmpne4(unsigned, unsigned);
unsigned __imf_vseteq2(unsigned, unsigned);
unsigned __imf_vseteq4(unsigned, unsigned);
unsigned __imf_vsetne2(unsigned, unsigned);
unsigned __imf_vsetne4(unsigned, unsigned);
unsigned __imf_vcmpges2(unsigned, unsigned);
unsigned __imf_vcmpges4(unsigned, unsigned);
unsigned __imf_vcmpgeu2(unsigned, unsigned);
unsigned __imf_vcmpgeu4(unsigned, unsigned);
unsigned __imf_vsetges2(unsigned, unsigned);
unsigned __imf_vsetges4(unsigned, unsigned);
unsigned __imf_vsetgeu2(unsigned, unsigned);
unsigned __imf_vsetgeu4(unsigned, unsigned);
unsigned __imf_vcmplts2(unsigned, unsigned);
unsigned __imf_vcmplts4(unsigned, unsigned);
unsigned __imf_vcmpltu2(unsigned, unsigned);
unsigned __imf_vcmpltu4(unsigned, unsigned);
unsigned __imf_vsetlts2(unsigned, unsigned);
unsigned __imf_vsetlts4(unsigned, unsigned);
unsigned __imf_vsetltu2(unsigned, unsigned);
unsigned __imf_vsetltu4(unsigned, unsigned);
unsigned __imf_vcmpgts2(unsigned, unsigned);
unsigned __imf_vcmpgts4(unsigned, unsigned);
unsigned __imf_vcmpgtu2(unsigned, unsigned);
unsigned __imf_vcmpgtu4(unsigned, unsigned);
unsigned __imf_vcmples2(unsigned, unsigned);
unsigned __imf_vcmples4(unsigned, unsigned);
unsigned __imf_vcmpleu2(unsigned, unsigned);
unsigned __imf_vcmpleu4(unsigned, unsigned);
unsigned __imf_vsetgts2(unsigned, unsigned);
unsigned __imf_vsetgts4(unsigned, unsigned);
unsigned __imf_vsetgtu2(unsigned, unsigned);
unsigned __imf_vsetgtu4(unsigned, unsigned);
unsigned __imf_vsetles2(unsigned, unsigned);
unsigned __imf_vsetles4(unsigned, unsigned);
unsigned __imf_vsetleu2(unsigned, unsigned);
unsigned __imf_vsetleu4(unsigned, unsigned);
}

void run_vabs2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_vals[NUM] = {0,          1,          0x98324,    0xFFFFFFFF,
                              0xFEDEAB98, 0x12345678, 0xBBCC9927, 0xA09BC176};
  unsigned output2_vals[NUM];
  unsigned output4_vals[NUM];
  unsigned ref2_vals[NUM] = {0,         1,          0x97CDC,    0x10001,
                             0x1225468, 0x12345678, 0x443466D9, 0x5F653E8A};

  unsigned ref4_vals[NUM] = {0,         1,          0x97D24,    0x1010101,
                             0x2225568, 0x12345678, 0x45346727, 0x60653f76};
  {
    s::buffer<unsigned, 1> output2_buf(&output2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output4_buf(&output4_vals[0], s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output2_acc = output2_buf.get_access<sycl_write>(cgh);
          auto output4_acc = output4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vabs2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output2_acc[I[0]] = __imf_vabs2(input_vals[I[0]]);
                output4_acc[I[0]] = __imf_vabs4(input_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output2_vals[idx] != ref2_vals[idx]) {
      std::cout << "__imf_vabs2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output4_vals[idx] != ref4_vals[idx]) {
      std::cout << "__imf_vabs4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "__imf_vabs2_4 test pass." << std::endl;
}

void run_vabsss2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_vals[NUM] = {0,          1,          0x98324,    0x07FF07FE,
                              0x80008000, 0x80808080, 0xBBCC9927, 0x80000001};
  unsigned output2_vals[NUM];
  unsigned output4_vals[NUM];
  unsigned ref2_vals[NUM] = {0,          1,          0x97CDC,    0x07FF07FE,
                             0x7FFF7FFF, 0x7F807F80, 0x443466D9, 0x7FFF0001};

  unsigned ref4_vals[NUM] = {0,          1,          0x97D24,    0x07010702,
                             0x7F007F00, 0x7F7F7F7F, 0x45346727, 0x7F000001};
  {
    s::buffer<unsigned, 1> output2_buf(&output2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output4_buf(&output4_vals[0], s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output2_acc = output2_buf.get_access<sycl_write>(cgh);
          auto output4_acc = output4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vabsss2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output2_acc[I[0]] = __imf_vabsss2(input_vals[I[0]]);
                output4_acc[I[0]] = __imf_vabsss4(input_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output2_vals[idx] != ref2_vals[idx]) {
      std::cout << "__imf_vabsss2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output4_vals[idx] != ref4_vals[idx]) {
      std::cout << "__imf_vabsss4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "__imf_vabsss2_4 test pass." << std::endl;
}

void run_vabsdiffsu2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_x_vals[NUM] = {0,          1,          0x98324,    0xFFFFFFFF,
                                0xFEDEAB98, 0x12345678, 0xBBCC9927, 0xA09BC176};
  unsigned input_y_vals[NUM] = {0,          1,          0x8372833,  0xABFF98EC,
                                0xA8372992, 0xBCCFED83, 0x9AA337BE, 0x73821207};
  unsigned output_s2_vals[NUM];
  unsigned output_s4_vals[NUM];
  unsigned output_u2_vals[NUM];
  unsigned output_u4_vals[NUM];
  unsigned ref_s2_vals[NUM] = {0,          0,          0x82EA50F,  0x54006713,
                               0x56A77DFA, 0x556568F5, 0x21299E97, 0xD2E75091};
  unsigned ref_s4_vals[NUM] = {0,          0,          0x82EA50F,  0x54006713,
                               0x56597E06, 0x566569F5, 0x21299E69, 0xD319516F};
  unsigned ref_u2_vals[NUM] = {0,          0,          0x82E5AF1,  0x54006713,
                               0x56A78206, 0xAA9B970B, 0x21296169, 0x2D19AF6F};
  unsigned ref_u4_vals[NUM] = {0,          0,          0x82E5B0F,  0x54006713,
                               0x56A78206, 0xAA9B970B, 0x21296297, 0x2D19AF6F};

  {
    s::buffer<unsigned, 1> output_s2_buf(&output_s2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_s4_buf(&output_s4_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_u2_buf(&output_u2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_u4_buf(&output_u4_vals[0], s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output_s2_acc = output_s2_buf.get_access<sycl_write>(cgh);
          auto output_s4_acc = output_s4_buf.get_access<sycl_write>(cgh);
          auto output_u2_acc = output_u2_buf.get_access<sycl_write>(cgh);
          auto output_u4_acc = output_u4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vabsdiffsu2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output_s2_acc[I[0]] =
                    __imf_vabsdiffs2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_s4_acc[I[0]] =
                    __imf_vabsdiffs4(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_u2_acc[I[0]] =
                    __imf_vabsdiffu2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_u4_acc[I[0]] =
                    __imf_vabsdiffu4(input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_s2_vals[idx] != ref_s2_vals[idx]) {
      std::cout << "__imf_vabsdiffs2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_s4_vals[idx] != ref_s4_vals[idx]) {
      std::cout << "__imf_vabsdiffs4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u2_vals[idx] != ref_u2_vals[idx]) {
      std::cout << "__imf_vabsdiffu2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u4_vals[idx] != ref_u4_vals[idx]) {
      std::cout << "__imf_vabsdiffu4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "__imf_vabsdiffsu_2_4 test pass." << std::endl;
}

void run_vadd_ss_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_x_vals[NUM] = {0,          0x7F7F7EFF, 0x98324,    0xFFFFFFFF,
                                0xFEDEAB98, 0x12345678, 0xBBCC9927, 0xA09BC176};
  unsigned input_y_vals[NUM] = {0,          0x6DDF66CC, 0x8372833,  0xABFF98EC,
                                0xA8372992, 0xBCCFED83, 0x9AA337BE, 0x73821207};
  unsigned output_u2_vals[NUM];
  unsigned output_u4_vals[NUM];
  unsigned output_ss2_vals[NUM];
  unsigned output_ss4_vals[NUM];
  unsigned ref_u2_vals[NUM] = {0,          0xED5EE5CB, 0x840AB57,  0xABFE98EB,
                               0xA715D52A, 0xCF0343FB, 0x566FD0E5, 0x141DD37D};
  unsigned ref_u4_vals[NUM] = {0,          0xEC5EE4CB, 0x840AB57,  0xAAFE97EB,
                               0xA615D42A, 0xCE0343FB, 0x556FD0E5, 0x131DD37D};
  unsigned ref_ss2_vals[NUM] = {0,          0x7FFF7FFF, 0x840AB57,  0xABFE98EB,
                                0xA715D52A, 0xCF0343FB, 0x8000D0E5, 0x141DD37D};
  unsigned ref_ss4_vals[NUM] = {0,          0x7F5E7FCB, 0x840AB57,  0xAAFE97EB,
                                0xA615D480, 0xCE0343FB, 0x8080D0E5, 0x1380D37D};

  {
    s::buffer<unsigned, 1> output_u2_buf(&output_u2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_u4_buf(&output_u4_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_ss2_buf(&output_ss2_vals[0],
                                          s::range<1>(NUM));
    s::buffer<unsigned, 1> output_ss4_buf(&output_ss4_vals[0],
                                          s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output_u2_acc = output_u2_buf.get_access<sycl_write>(cgh);
          auto output_u4_acc = output_u4_buf.get_access<sycl_write>(cgh);
          auto output_ss2_acc = output_ss2_buf.get_access<sycl_write>(cgh);
          auto output_ss4_acc = output_ss4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vadd2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output_u2_acc[I[0]] =
                    __imf_vadd2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_u4_acc[I[0]] =
                    __imf_vadd4(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_ss2_acc[I[0]] =
                    __imf_vaddss2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_ss4_acc[I[0]] =
                    __imf_vaddss4(input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u2_vals[idx] != ref_u2_vals[idx]) {
      std::cout << "__imf_vadd2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u4_vals[idx] != ref_u4_vals[idx]) {
      std::cout << "__imf_vadd4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_ss2_vals[idx] != ref_ss2_vals[idx]) {
      std::cout << "__imf_vaddss2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_ss4_vals[idx] != ref_ss4_vals[idx]) {
      std::cout << "__imf_vaddss4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "__imf_vadd_2_4 test pass." << std::endl;
}

void run_vadd_us_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_x_vals[NUM] = {0,          0x7F7F7EFF, 0x98324,    0xFFFFFFFF,
                                0xFEDEAB98, 0x12345678, 0xBBCC9927, 0xA09BC176};
  unsigned input_y_vals[NUM] = {0,          0x6DDF66CC, 0x8372833,  0xABFF98EC,
                                0xA8372992, 0xBCCFED83, 0x9AA337BE, 0x73821207};
  unsigned output_u2_vals[NUM];
  unsigned output_u4_vals[NUM];
  unsigned ref_u2_vals[NUM] = {0,          0xED5EE5CB, 0x840AB57,  0xFFFFFFFF,
                               0xFFFFD52A, 0xCF03FFFF, 0xFFFFD0E5, 0xFFFFD37D};
  unsigned ref_u4_vals[NUM] = {0,          0xECFFE4FF, 0x840AB57,  0xFFFFFFFF,
                               0xFFFFD4FF, 0xCEFFFFFB, 0xFFFFD0E5, 0xFFFFD37D};

  {
    s::buffer<unsigned, 1> output_u2_buf(&output_u2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_u4_buf(&output_u4_vals[0], s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output_u2_acc = output_u2_buf.get_access<sycl_write>(cgh);
          auto output_u4_acc = output_u4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vaddus2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output_u2_acc[I[0]] =
                    __imf_vaddus2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_u4_acc[I[0]] =
                    __imf_vaddus4(input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u2_vals[idx] != ref_u2_vals[idx]) {
      std::cout << "__imf_vaddus2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u4_vals[idx] != ref_u4_vals[idx]) {
      std::cout << "__imf_vaddus4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "__imf_vaddus_2_4 test pass." << std::endl;
}

void run_vhaddu_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_x_vals[NUM] = {0,          0x7F7F7EFF, 0x98324,    0xFFFFFFFF,
                                0xFEDEAB98, 0x12345678, 0xBBCC9927, 0xA09BC176};
  unsigned input_y_vals[NUM] = {0,          0x6DDF66CC, 0x8372833,  0xABFF98EC,
                                0xA8372992, 0xBCCFED83, 0x9AA337BE, 0x73821207};
  unsigned output_u2_vals[NUM];
  unsigned output_u4_vals[NUM];
  unsigned ref_u2_vals[NUM] = {0,          0x76AF72E5, 0x42055AB,  0xD5FFCC75,
                               0xD38A6A95, 0x6781A1FD, 0xAB376872, 0x8A0E69BE};
  unsigned ref_u4_vals[NUM] = {0,          0x76AF72E5, 0x420552B,  0xD5FFCBF5,
                               0xD38A6A95, 0x6781A17D, 0xAAB76872, 0x898E693E};

  {
    s::buffer<unsigned, 1> output_u2_buf(&output_u2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_u4_buf(&output_u4_vals[0], s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output_u2_acc = output_u2_buf.get_access<sycl_write>(cgh);
          auto output_u4_acc = output_u4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vhaddu_2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output_u2_acc[I[0]] =
                    __imf_vhaddu2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_u4_acc[I[0]] =
                    __imf_vhaddu4(input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u2_vals[idx] != ref_u2_vals[idx]) {
      std::cout << "__imf_vhaddu2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u4_vals[idx] != ref_u4_vals[idx]) {
      std::cout << "__imf_vhaddu4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "__imf_vhaddu_2_4 test pass." << std::endl;
}

void run_vsub_ss_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_x_vals[NUM] = {0,          0x7F7F7EFF, 0x98324,    0xFFFFFFFF,
                                0xFEDEAB98, 0x12345678, 0xBBCC9927, 0xA09BC176};
  unsigned input_y_vals[NUM] = {0,          0x6DDF66CC, 0x8372833,  0xABFF98EC,
                                0xA8372992, 0xBCCFED83, 0x9AA337BE, 0x73821207};
  unsigned output_u2_vals[NUM];
  unsigned output_u4_vals[NUM];
  unsigned output_ss2_vals[NUM];
  unsigned output_ss4_vals[NUM];
  unsigned ref_u2_vals[NUM] = {0,          0x11A01833, 0xF7D25AF1, 0x54006713,
                               0x56A78206, 0x556568F5, 0x21296169, 0x2D19AF6F};
  unsigned ref_u4_vals[NUM] = {0,          0x12A01833, 0xF8D25BF1, 0x54006713,
                               0x56A78206, 0x566569F5, 0x21296269, 0x2D19AF6F};
  unsigned ref_ss2_vals[NUM] = {0,          0x11A01833, 0xF7D28000, 0x54006713,
                                0x56A78206, 0x556568F5, 0x21298000, 0x8000AF6F};
  unsigned ref_ss4_vals[NUM] = {0,          0x127F1833, 0xF8D280F1, 0x54006713,
                                0x56A78206, 0x5665697F, 0x21298069, 0x8019AF6F};

  {
    s::buffer<unsigned, 1> output_u2_buf(&output_u2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_u4_buf(&output_u4_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_ss2_buf(&output_ss2_vals[0],
                                          s::range<1>(NUM));
    s::buffer<unsigned, 1> output_ss4_buf(&output_ss4_vals[0],
                                          s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output_u2_acc = output_u2_buf.get_access<sycl_write>(cgh);
          auto output_u4_acc = output_u4_buf.get_access<sycl_write>(cgh);
          auto output_ss2_acc = output_ss2_buf.get_access<sycl_write>(cgh);
          auto output_ss4_acc = output_ss4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vsub2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output_u2_acc[I[0]] =
                    __imf_vsub2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_u4_acc[I[0]] =
                    __imf_vsub4(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_ss2_acc[I[0]] =
                    __imf_vsubss2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_ss4_acc[I[0]] =
                    __imf_vsubss4(input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u2_vals[idx] != ref_u2_vals[idx]) {
      std::cout << "__imf_vsub2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u4_vals[idx] != ref_u4_vals[idx]) {
      std::cout << "__imf_vsub4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_ss2_vals[idx] != ref_ss2_vals[idx]) {
      std::cout << "__imf_vsubss2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_ss4_vals[idx] != ref_ss4_vals[idx]) {
      std::cout << "__imf_vsubss4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "__imf_vsub_2_4 test pass." << std::endl;
}

void run_vsub_us_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_x_vals[NUM] = {0,          0x7F7F7EFF, 0x98324,    0xFFFFFFFF,
                                0xFEDEAB98, 0x12345678, 0xBBCC9927, 0xA09BC176};
  unsigned input_y_vals[NUM] = {0,          0x6DDF66CC, 0x8372833,  0xABFF98EC,
                                0xA8372992, 0xBCCFED83, 0x9AA337BE, 0x73821207};
  unsigned output_u2_vals[NUM];
  unsigned output_u4_vals[NUM];
  unsigned ref_u2_vals[NUM] = {0,          0x11A01833, 0x5AF1,     0x54006713,
                               0x56A78206, 0x0,        0x21296169, 0x2D19AF6F};
  unsigned ref_u4_vals[NUM] = {0,          0x12001833, 0x5B00,     0x54006713,
                               0x56A78206, 0x0,        0x21296200, 0x2D19AF6F};

  {
    s::buffer<unsigned, 1> output_u2_buf(&output_u2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_u4_buf(&output_u4_vals[0], s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output_u2_acc = output_u2_buf.get_access<sycl_write>(cgh);
          auto output_u4_acc = output_u4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vsubus2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output_u2_acc[I[0]] =
                    __imf_vsubus2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_u4_acc[I[0]] =
                    __imf_vsubus4(input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u2_vals[idx] != ref_u2_vals[idx]) {
      std::cout << "__imf_vsubus2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u4_vals[idx] != ref_u4_vals[idx]) {
      std::cout << "__imf_vsubus4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "__imf_vsubus_2_4 test pass." << std::endl;
}

void run_vavgs_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_x_vals[NUM] = {0,          0x7F7F7EFF, 0x98324017, 0xFFFFFFFF,
                                0xFEDEAB98, 0x829C5678, 0x80007FFF, 0xA09BC176};
  unsigned input_y_vals[NUM] = {0,          0x7DDF56CC, 0x83728331, 0xABFF98EC,
                                0xA8372992, 0x8A6CED83, 0x800037BE, 0x73821207};
  unsigned output_s2_vals[NUM];
  unsigned output_s4_vals[NUM];
  unsigned ref_s2_vals[NUM] = {0,          0x7EAF6AE6, 0x8DD2E1A4, 0xD5FFCC75,
                               0xD38AEA95, 0x868421FE, 0x80005BDF, 0xA0FE9BE};
  unsigned ref_s4_vals[NUM] = {0,          0x7E2F6AE5, 0x8D52E124, 0xD5FFCBF5,
                               0xD30BEA95, 0x860422FD, 0x80005BDE, 0xA8EE93F};

  {
    s::buffer<unsigned, 1> output_s2_buf(&output_s2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_s4_buf(&output_s4_vals[0], s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output_s2_acc = output_s2_buf.get_access<sycl_write>(cgh);
          auto output_s4_acc = output_s4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vavgs_2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output_s2_acc[I[0]] =
                    __imf_vavgs2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_s4_acc[I[0]] =
                    __imf_vavgs4(input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_s2_vals[idx] != ref_s2_vals[idx]) {
      std::cout << "__imf_vavgs2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_s4_vals[idx] != ref_s4_vals[idx]) {
      std::cout << "__imf_vavgs4 failed! idx = " << idx << std::endl;
      std::cout << std::hex << output_s4_vals[idx] << "  " << ref_s4_vals[idx]
                << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "__imf_vavgs_2_4 test pass." << std::endl;
}

void run_vavgu_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_x_vals[NUM] = {0,          0x7F7F7EFF, 0x98324017, 0xFFFFFFFF,
                                0xFEDEAB98, 0x829C5678, 0x80007FFF, 0xA09BC176};
  unsigned input_y_vals[NUM] = {1,          0x7DDF56CC, 0x83728331, 0xABFF98EC,
                                0xA8372992, 0x8A6CED83, 0x800037BE, 0x73821207};
  unsigned output_u2_vals[NUM];
  unsigned output_u4_vals[NUM];
  unsigned ref_u2_vals[NUM] = {1,          0x7EAF6AE6, 0x8DD261A4, 0xD5FFCC76,
                               0xD38B6A95, 0x8684A1FE, 0x80005BDF, 0x8A0F69BF};
  unsigned ref_u4_vals[NUM] = {1,          0x7EAF6AE6, 0x8E526224, 0xD5FFCCF6,
                               0xD38B6A95, 0x8684A27E, 0x80005BDF, 0x8A8F6A3F};

  {
    s::buffer<unsigned, 1> output_u2_buf(&output_u2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_u4_buf(&output_u4_vals[0], s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output_u2_acc = output_u2_buf.get_access<sycl_write>(cgh);
          auto output_u4_acc = output_u4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vavgu_2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output_u2_acc[I[0]] =
                    __imf_vavgu2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_u4_acc[I[0]] =
                    __imf_vavgu4(input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u2_vals[idx] != ref_u2_vals[idx]) {
      std::cout << "__imf_vavgu2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u4_vals[idx] != ref_u4_vals[idx]) {
      std::cout << "__imf_vavgu4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "__imf_vavgu_2_4 test pass." << std::endl;
}

void run_vcmpgts_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_x_vals[NUM] = {0,          0x7F7F7EFF, 0x98324017, 0xFFFFFFFF,
                                0xFEDEAB98, 0x829C5678, 0x80007FFF, 0xA09BC176};
  unsigned input_y_vals[NUM] = {0,          0x7DDF56CC, 0x98338331, 0xABFF98EC,
                                0xA8372992, 0x8A6CED83, 0x800037BE, 0x73821207};
  unsigned output_s2_vals[NUM];
  unsigned output_s4_vals[NUM];
  unsigned ref_s2_vals[NUM] = {0,          0xFFFFFFFF, 0x0000FFFF, 0xFFFFFFFF,
                               0xFFFF0000, 0x0000FFFF, 0x0000FFFF, 0};
  unsigned ref_s4_vals[NUM] = {0,          0xFFFFFFFF, 0x0000FF00, 0xFF00FFFF,
                               0xFF0000FF, 0x0000FFFF, 0x0000FFFF, 0x00FF00FF};

  {
    s::buffer<unsigned, 1> output_s2_buf(&output_s2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_s4_buf(&output_s4_vals[0], s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output_s2_acc = output_s2_buf.get_access<sycl_write>(cgh);
          auto output_s4_acc = output_s4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vcmpgts_2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output_s2_acc[I[0]] =
                    __imf_vcmpgts2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_s4_acc[I[0]] =
                    __imf_vcmpgts4(input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_s2_vals[idx] != ref_s2_vals[idx]) {
      std::cout << "__imf_vcmpgts2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_s4_vals[idx] != ref_s4_vals[idx]) {
      std::cout << "__imf_vcmpgts4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "__imf_vcmpgts_2_4 test pass." << std::endl;
}

void run_vmaxmin_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_x_vals[NUM] = {0,          1,          0x98324,    0xFFFFFFFF,
                                0xFEDEAB98, 0x12345678, 0xBBCC9927, 0xA09BC176};
  unsigned input_y_vals[NUM] = {0,          1,          0x83712833, 0xABFF98EC,
                                0xA8372992, 0xBCCFED83, 0x9AA337BE, 0x73821207};
  unsigned max_output_s2_vals[NUM];
  unsigned max_output_s4_vals[NUM];
  unsigned max_output_u2_vals[NUM];
  unsigned max_output_u4_vals[NUM];
  unsigned min_output_s2_vals[NUM];
  unsigned min_output_s4_vals[NUM];
  unsigned min_output_u2_vals[NUM];
  unsigned min_output_u4_vals[NUM];
  unsigned max_ref_s2_vals[NUM] = {0,          1,          0x92833,
                                   0xFFFFFFFF, 0xFEDE2992, 0x12345678,
                                   0xBBCC37BE, 0x73821207};
  unsigned max_ref_s4_vals[NUM] = {0,          1,          0x712833,
                                   0xFFFFFFFF, 0xFE372998, 0x12345678,
                                   0xBBCC3727, 0x739B1276};
  unsigned max_ref_u2_vals[NUM] = {0,          1,          0x83718324,
                                   0xFFFFFFFF, 0xFEDEAB98, 0xBCCFED83,
                                   0xBBCC9927, 0xA09BC176};
  unsigned max_ref_u4_vals[NUM] = {0,          1,          0x83718333,
                                   0xFFFFFFFF, 0xFEDEAB98, 0xBCCFED83,
                                   0xBBCC99BE, 0xA09BC176};

  unsigned min_ref_s2_vals[NUM] = {0,          1,          0x83718324,
                                   0xABFF98EC, 0xA837Ab98, 0xBCCFED83,
                                   0x9AA39927, 0xA09BC176};
  unsigned min_ref_s4_vals[NUM] = {0,          1,          0x83098324,
                                   0xABFF98EC, 0xA8DEAB92, 0xBCCFED83,
                                   0x9AA399BE, 0xA082C107};
  unsigned min_ref_u2_vals[NUM] = {0,          1,          0x92833,
                                   0xABFF98EC, 0xA8372992, 0x12345678,
                                   0x9AA337BE, 0x73821207};
  unsigned min_ref_u4_vals[NUM] = {0,          1,          0x92824,
                                   0xABFF98EC, 0xA8372992, 0x12345678,
                                   0x9AA33727, 0x73821207};

  {
    s::buffer<unsigned, 1> max_output_s2_buf(&max_output_s2_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> max_output_s4_buf(&max_output_s4_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> max_output_u2_buf(&max_output_u2_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> max_output_u4_buf(&max_output_u4_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> min_output_s2_buf(&min_output_s2_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> min_output_s4_buf(&min_output_s4_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> min_output_u2_buf(&min_output_u2_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> min_output_u4_buf(&min_output_u4_vals[0],
                                             s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto max_output_s2_acc =
              max_output_s2_buf.get_access<sycl_write>(cgh);
          auto max_output_s4_acc =
              max_output_s4_buf.get_access<sycl_write>(cgh);
          auto max_output_u2_acc =
              max_output_u2_buf.get_access<sycl_write>(cgh);
          auto max_output_u4_acc =
              max_output_u4_buf.get_access<sycl_write>(cgh);
          auto min_output_s2_acc =
              min_output_s2_buf.get_access<sycl_write>(cgh);
          auto min_output_s4_acc =
              min_output_s4_buf.get_access<sycl_write>(cgh);
          auto min_output_u2_acc =
              min_output_u2_buf.get_access<sycl_write>(cgh);
          auto min_output_u4_acc =
              min_output_u4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vmaxmin_2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                max_output_s2_acc[I[0]] =
                    __imf_vmaxs2(input_x_vals[I[0]], input_y_vals[I[0]]);
                max_output_s4_acc[I[0]] =
                    __imf_vmaxs4(input_x_vals[I[0]], input_y_vals[I[0]]);
                max_output_u2_acc[I[0]] =
                    __imf_vmaxu2(input_x_vals[I[0]], input_y_vals[I[0]]);
                max_output_u4_acc[I[0]] =
                    __imf_vmaxu4(input_x_vals[I[0]], input_y_vals[I[0]]);
                min_output_s2_acc[I[0]] =
                    __imf_vmins2(input_x_vals[I[0]], input_y_vals[I[0]]);
                min_output_s4_acc[I[0]] =
                    __imf_vmins4(input_x_vals[I[0]], input_y_vals[I[0]]);
                min_output_u2_acc[I[0]] =
                    __imf_vminu2(input_x_vals[I[0]], input_y_vals[I[0]]);
                min_output_u4_acc[I[0]] =
                    __imf_vminu4(input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (max_output_s2_vals[idx] != max_ref_s2_vals[idx]) {
      std::cout << "__imf_vmaxs2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (max_output_s4_vals[idx] != max_ref_s4_vals[idx]) {
      std::cout << "__imf_vmaxs4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (max_output_u2_vals[idx] != max_ref_u2_vals[idx]) {
      std::cout << "__imf_vmaxu2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (max_output_u4_vals[idx] != max_ref_u4_vals[idx]) {
      std::cout << "__imf_vmaxu4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (min_output_s2_vals[idx] != min_ref_s2_vals[idx]) {
      std::cout << "__imf_vmins2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (min_output_s4_vals[idx] != min_ref_s4_vals[idx]) {
      std::cout << "__imf_vmins4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (min_output_u2_vals[idx] != min_ref_u2_vals[idx]) {
      std::cout << "__imf_vminu2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (min_output_u4_vals[idx] != min_ref_u4_vals[idx]) {
      std::cout << "__imf_vminu4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "__imf_vmaxmin_2_4 test pass." << std::endl;
}

void run_vneg_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_vals[NUM] = {0,          1,          0x98324,    0xFFFFFFFF,
                              0x1345AB98, 0x12345678, 0xBBCC9927, 0xA09BC176};
  unsigned output2_vals[NUM];
  unsigned output4_vals[NUM];
  unsigned ref2_vals[NUM] = {0,          0xFFFF,     0xFFF77CDC, 0x10001,
                             0xECBB5468, 0xEDCCA988, 0x443466D9, 0x5F653E8A};

  unsigned ref4_vals[NUM] = {0,          0xFF,       0xF77DDC,   0x1010101,
                             0xEDBB5568, 0xEECCAA88, 0x453467D9, 0x60653F8A};
  {
    s::buffer<unsigned, 1> output2_buf(&output2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output4_buf(&output4_vals[0], s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output2_acc = output2_buf.get_access<sycl_write>(cgh);
          auto output4_acc = output4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vneg_2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output2_acc[I[0]] = __imf_vneg2(input_vals[I[0]]);
                output4_acc[I[0]] = __imf_vneg4(input_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output2_vals[idx] != ref2_vals[idx]) {
      std::cout << "__imf_vneg2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output4_vals[idx] != ref4_vals[idx]) {
      std::cout << "__imf_vneg4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "__imf_vneg_2_4 test pass." << std::endl;
}

void run_vnegss_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 4;
  unsigned input_vals[NUM] = {1, 0x98324, 0x7FFF8000, 0x7F801127};
  unsigned output2_vals[NUM];
  unsigned output4_vals[NUM];
  unsigned ref2_vals[NUM] = {0xFFFF, 0xFFF77CDC, 0x80017FFF, 0x8080EED9};
  unsigned ref4_vals[NUM] = {0xFF, 0xF77DDC, 0x81017F00, 0x817FEFD9};
  {
    s::buffer<unsigned, 1> output2_buf(&output2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output4_buf(&output4_vals[0], s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output2_acc = output2_buf.get_access<sycl_write>(cgh);
          auto output4_acc = output4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vnegss_2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output2_acc[I[0]] = __imf_vnegss2(input_vals[I[0]]);
                output4_acc[I[0]] = __imf_vnegss4(input_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output2_vals[idx] != ref2_vals[idx]) {
      std::cout << "__imf_vnegss2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output4_vals[idx] != ref4_vals[idx]) {
      std::cout << "__imf_vnegss4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "__imf_vnegss_2_4 test pass." << std::endl;
}

void run_vsad_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 8;
  unsigned input_x_vals[NUM] = {0,          0xFFFF,     0x76498324, 0x80008000,
                                0x7F6F5F4F, 0xFFFFFFFF, 0xBBCC9927, 0xA09BC176};
  unsigned input_y_vals[NUM] = {0x7F7F8000, 0x12458964, 0x34212833, 0x7FFF7FFF,
                                0x80818283, 0x0,        0x7FFF6883, 0x12345678};
  unsigned output_s2_vals[NUM];
  unsigned output_s4_vals[NUM];
  unsigned output_u2_vals[NUM];
  unsigned output_u4_vals[NUM];
  unsigned ref_s2_vals[NUM] = {0xFF7F,  0x88E0, 0xE737,  0x1FFFE,
                               0x1DBBA, 0x2,    0x1938F, 0x1069B};
  unsigned ref_s4_vals[NUM] = {0x17E, 0x132, 0x11E, 0x200,
                               0x396, 0x4,   0x26A, 0x1A2};
  unsigned ref_u2_vals[NUM] = {0xFF7F, 0x88E0,  0x9D19, 0x2,
                               0x2446, 0x1FFFE, 0x6C71, 0xF965};
  unsigned ref_u4_vals[NUM] = {0x17E, 0x168, 0xD4, 0x200,
                               0x6A,  0x3FC, 0xFC, 0x162};

  {
    s::buffer<unsigned, 1> output_s2_buf(&output_s2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_s4_buf(&output_s4_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_u2_buf(&output_u2_vals[0], s::range<1>(NUM));
    s::buffer<unsigned, 1> output_u4_buf(&output_u4_vals[0], s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output_s2_acc = output_s2_buf.get_access<sycl_write>(cgh);
          auto output_s4_acc = output_s4_buf.get_access<sycl_write>(cgh);
          auto output_u2_acc = output_u2_buf.get_access<sycl_write>(cgh);
          auto output_u4_acc = output_u4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vsad_2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output_s2_acc[I[0]] =
                    __imf_vsads2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_s4_acc[I[0]] =
                    __imf_vsads4(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_u2_acc[I[0]] =
                    __imf_vsadu2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_u4_acc[I[0]] =
                    __imf_vsadu4(input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_s2_vals[idx] != ref_s2_vals[idx]) {
      std::cout << "__imf_vsads2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_s4_vals[idx] != ref_s4_vals[idx]) {
      std::cout << "__imf_vsads4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u2_vals[idx] != ref_u2_vals[idx]) {
      std::cout << "__imf_vsadu2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_u4_vals[idx] != ref_u4_vals[idx]) {
      std::cout << "__imf_vsadu4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "__imf_vsad_2_4 test pass." << std::endl;
}

void run_veqne_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 5;
  unsigned input_x_vals[NUM] = {0x1, 0xFEDE1122, 0x12345678, 0xFFFFFFFF,
                                0x98765432};
  unsigned input_y_vals[NUM] = {0x2, 0xFEDF1133, 0x12345679, 0xFFFFFFFF,
                                0x87654321};
  unsigned output_cmpeq2_vals[NUM];
  unsigned output_cmpeq4_vals[NUM];
  unsigned output_cmpne2_vals[NUM];
  unsigned output_cmpne4_vals[NUM];
  unsigned output_seteq2_vals[NUM];
  unsigned output_seteq4_vals[NUM];
  unsigned output_setne2_vals[NUM];
  unsigned output_setne4_vals[NUM];
  unsigned cmpeq2_ref_vals[NUM] = {0xFFFF0000, 0, 0xFFFF0000, 0xFFFFFFFF, 0};
  unsigned cmpeq4_ref_vals[NUM] = {0xFFFFFF00, 0xFF00FF00, 0xFFFFFF00,
                                   0xFFFFFFFF, 0};
  unsigned cmpne2_ref_vals[NUM] = {0xFFFF, 0xFFFFFFFF, 0xFFFF, 0, 0xFFFFFFFF};
  unsigned cmpne4_ref_vals[NUM] = {0xFF, 0xFF00FF, 0xFF, 0, 0xFFFFFFFF};

  unsigned seteq2_ref_vals[NUM] = {0x10000, 0, 0x10000, 0x10001, 0};
  unsigned seteq4_ref_vals[NUM] = {0x1010100, 0x1000100, 0x1010100, 0x1010101,
                                   0};
  unsigned setne2_ref_vals[NUM] = {1, 0x10001, 1, 0, 0x10001};
  unsigned setne4_ref_vals[NUM] = {1, 0x10001, 1, 0x0, 0x1010101};

  {
    s::buffer<unsigned, 1> output_cmpeq2_buf(&output_cmpeq2_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmpeq4_buf(&output_cmpeq4_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmpne2_buf(&output_cmpne2_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmpne4_buf(&output_cmpne4_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> output_seteq2_buf(&output_seteq2_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> output_seteq4_buf(&output_seteq4_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setne2_buf(&output_setne2_vals[0],
                                             s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setne4_buf(&output_setne4_vals[0],
                                             s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output_cmpeq2_acc =
              output_cmpeq2_buf.get_access<sycl_write>(cgh);
          auto output_cmpeq4_acc =
              output_cmpeq4_buf.get_access<sycl_write>(cgh);
          auto output_cmpne2_acc =
              output_cmpne2_buf.get_access<sycl_write>(cgh);
          auto output_cmpne4_acc =
              output_cmpne4_buf.get_access<sycl_write>(cgh);
          auto output_seteq2_acc =
              output_seteq2_buf.get_access<sycl_write>(cgh);
          auto output_seteq4_acc =
              output_seteq4_buf.get_access<sycl_write>(cgh);
          auto output_setne2_acc =
              output_setne2_buf.get_access<sycl_write>(cgh);
          auto output_setne4_acc =
              output_setne4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class veqne_2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output_cmpeq2_acc[I[0]] =
                    __imf_vcmpeq2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmpeq4_acc[I[0]] =
                    __imf_vcmpeq4(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmpne2_acc[I[0]] =
                    __imf_vcmpne2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmpne4_acc[I[0]] =
                    __imf_vcmpne4(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_seteq2_acc[I[0]] =
                    __imf_vseteq2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_seteq4_acc[I[0]] =
                    __imf_vseteq4(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setne2_acc[I[0]] =
                    __imf_vsetne2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setne4_acc[I[0]] =
                    __imf_vsetne4(input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpeq2_vals[idx] != cmpeq2_ref_vals[idx]) {
      std::cout << "__imf_vcmpeq2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpeq4_vals[idx] != cmpeq4_ref_vals[idx]) {
      std::cout << "__imf_vcmpeq4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpne2_vals[idx] != cmpne2_ref_vals[idx]) {
      std::cout << "__imf_vcmpne2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpne4_vals[idx] != cmpne4_ref_vals[idx]) {
      std::cout << "__imf_vcmpne4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_seteq2_vals[idx] != seteq2_ref_vals[idx]) {
      std::cout << "__imf_vseteq2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_seteq4_vals[idx] != seteq4_ref_vals[idx]) {
      std::cout << "__imf_vseteq4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setne2_vals[idx] != setne2_ref_vals[idx]) {
      std::cout << "__imf_vsetne2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setne4_vals[idx] != setne4_ref_vals[idx]) {
      std::cout << "__imf_vsetne4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "__imf_veqne_2_4 test pass." << std::endl;
}

void run_vgelt_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 5;
  unsigned input_x_vals[NUM] = {0x1, 0xFEDE1122, 0x12345678, 0xFFFFFFFF,
                                0x98765432};
  unsigned input_y_vals[NUM] = {0x2, 0xFEDF1133, 0x12345679, 0xFFFFFFFF,
                                0x87654321};
  unsigned output_cmpges2_vals[NUM];
  unsigned output_cmpges4_vals[NUM];
  unsigned output_cmplts2_vals[NUM];
  unsigned output_cmplts4_vals[NUM];
  unsigned output_cmpgeu2_vals[NUM];
  unsigned output_cmpgeu4_vals[NUM];
  unsigned output_cmpltu2_vals[NUM];
  unsigned output_cmpltu4_vals[NUM];
  unsigned output_setges2_vals[NUM];
  unsigned output_setges4_vals[NUM];
  unsigned output_setlts2_vals[NUM];
  unsigned output_setlts4_vals[NUM];
  unsigned output_setgeu2_vals[NUM];
  unsigned output_setgeu4_vals[NUM];
  unsigned output_setltu2_vals[NUM];
  unsigned output_setltu4_vals[NUM];
  unsigned cmpges2_ref_vals[NUM] = {0xFFFF0000, 0, 0xFFFF0000, 0xFFFFFFFF,
                                    0xFFFFFFFF};
  unsigned cmpges4_ref_vals[NUM] = {0xFFFFFF00, 0xFF00FF00, 0xFFFFFF00,
                                    0xFFFFFFFF, 0xFFFFFFFF};
  unsigned cmplts2_ref_vals[NUM] = {0xFFFF, 0xFFFFFFFF, 0xFFFF, 0, 0};
  unsigned cmplts4_ref_vals[NUM] = {0xFF, 0xFF00FF, 0xFF, 0, 0};

  unsigned cmpgeu2_ref_vals[NUM] = {0xFFFF0000, 0, 0xFFFF0000, 0xFFFFFFFF,
                                    0xFFFFFFFF};
  unsigned cmpgeu4_ref_vals[NUM] = {0xFFFFFF00, 0xFF00FF00, 0xFFFFFF00,
                                    0xFFFFFFFF, 0xFFFFFFFF};
  unsigned cmpltu2_ref_vals[NUM] = {0xFFFF, 0xFFFFFFFF, 0xFFFF, 0, 0};
  unsigned cmpltu4_ref_vals[NUM] = {0xFF, 0xFF00FF, 0xFF, 0, 0};
  unsigned setges2_ref_vals[NUM] = {0x10000, 0, 0x10000, 0x10001, 0x10001};
  unsigned setges4_ref_vals[NUM] = {0x1010100, 0x1000100, 0x1010100, 0x1010101,
                                    0x1010101};
  unsigned setlts2_ref_vals[NUM] = {1, 0x10001, 1, 0, 0};
  unsigned setlts4_ref_vals[NUM] = {1, 0x10001, 1, 0, 0};
  unsigned setgeu2_ref_vals[NUM] = {0x10000, 0, 0x10000, 0x10001, 0x10001};
  unsigned setgeu4_ref_vals[NUM] = {0x1010100, 0x1000100, 0x1010100, 0x1010101,
                                    0x1010101};
  unsigned setltu2_ref_vals[NUM] = {1, 0x10001, 1, 0, 0};
  unsigned setltu4_ref_vals[NUM] = {1, 0x10001, 1, 0, 0};

  {
    s::buffer<unsigned, 1> output_cmpges2_buf(&output_cmpges2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmpges4_buf(&output_cmpges4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmplts2_buf(&output_cmplts2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmplts4_buf(&output_cmplts4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmpgeu2_buf(&output_cmpgeu2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmpgeu4_buf(&output_cmpgeu4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmpltu2_buf(&output_cmpltu2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmpltu4_buf(&output_cmpltu4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setges2_buf(&output_setges2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setges4_buf(&output_setges4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setlts2_buf(&output_setlts2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setlts4_buf(&output_setlts4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setgeu2_buf(&output_setgeu2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setgeu4_buf(&output_setgeu4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setltu2_buf(&output_setltu2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setltu4_buf(&output_setltu4_vals[0],
                                              s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output_cmpges2_acc =
              output_cmpges2_buf.get_access<sycl_write>(cgh);
          auto output_cmpges4_acc =
              output_cmpges4_buf.get_access<sycl_write>(cgh);
          auto output_cmplts2_acc =
              output_cmplts2_buf.get_access<sycl_write>(cgh);
          auto output_cmplts4_acc =
              output_cmplts4_buf.get_access<sycl_write>(cgh);
          auto output_cmpgeu2_acc =
              output_cmpgeu2_buf.get_access<sycl_write>(cgh);
          auto output_cmpgeu4_acc =
              output_cmpgeu4_buf.get_access<sycl_write>(cgh);
          auto output_cmpltu2_acc =
              output_cmpltu2_buf.get_access<sycl_write>(cgh);
          auto output_cmpltu4_acc =
              output_cmpltu4_buf.get_access<sycl_write>(cgh);
          auto output_setges2_acc =
              output_setges2_buf.get_access<sycl_write>(cgh);
          auto output_setges4_acc =
              output_setges4_buf.get_access<sycl_write>(cgh);
          auto output_setlts2_acc =
              output_setlts2_buf.get_access<sycl_write>(cgh);
          auto output_setlts4_acc =
              output_setlts4_buf.get_access<sycl_write>(cgh);
          auto output_setgeu2_acc =
              output_setgeu2_buf.get_access<sycl_write>(cgh);
          auto output_setgeu4_acc =
              output_setgeu4_buf.get_access<sycl_write>(cgh);
          auto output_setltu2_acc =
              output_setltu2_buf.get_access<sycl_write>(cgh);
          auto output_setltu4_acc =
              output_setltu4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vgelt_2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output_cmpges2_acc[I[0]] =
                    __imf_vcmpges2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmpges4_acc[I[0]] =
                    __imf_vcmpges4(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmplts2_acc[I[0]] =
                    __imf_vcmplts2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmplts4_acc[I[0]] =
                    __imf_vcmplts4(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmpgeu2_acc[I[0]] =
                    __imf_vcmpgeu2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmpgeu4_acc[I[0]] =
                    __imf_vcmpgeu4(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmpltu2_acc[I[0]] =
                    __imf_vcmpltu2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmpltu4_acc[I[0]] =
                    __imf_vcmpltu4(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setges2_acc[I[0]] =
                    __imf_vsetges2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setges4_acc[I[0]] =
                    __imf_vsetges4(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setlts2_acc[I[0]] =
                    __imf_vsetlts2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setlts4_acc[I[0]] =
                    __imf_vsetlts4(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setgeu2_acc[I[0]] =
                    __imf_vsetgeu2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setgeu4_acc[I[0]] =
                    __imf_vsetgeu4(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setltu2_acc[I[0]] =
                    __imf_vsetltu2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setltu4_acc[I[0]] =
                    __imf_vsetltu4(input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpges2_vals[idx] != cmpges2_ref_vals[idx]) {
      std::cout << "__imf_vcmpges2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpges4_vals[idx] != cmpges4_ref_vals[idx]) {
      std::cout << "__imf_vcmpges4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmplts2_vals[idx] != cmplts2_ref_vals[idx]) {
      std::cout << "__imf_vcmplts2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmplts4_vals[idx] != cmplts4_ref_vals[idx]) {
      std::cout << "__imf_vcmplts4 failed! idx = " << idx << std::endl;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpgeu2_vals[idx] != cmpgeu2_ref_vals[idx]) {
      std::cout << "__imf_vcmpgeu2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpgeu4_vals[idx] != cmpgeu4_ref_vals[idx]) {
      std::cout << "__imf_vcmpgeu4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpltu2_vals[idx] != cmpltu2_ref_vals[idx]) {
      std::cout << "__imf_vcmpltu2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpltu4_vals[idx] != cmpltu4_ref_vals[idx]) {
      std::cout << "__imf_vcmpltu4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setges2_vals[idx] != setges2_ref_vals[idx]) {
      std::cout << "__imf_vsetges2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setges4_vals[idx] != setges4_ref_vals[idx]) {
      std::cout << "__imf_vsetges4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setlts2_vals[idx] != setlts2_ref_vals[idx]) {
      std::cout << "__imf_vsetlts2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setlts4_vals[idx] != setlts4_ref_vals[idx]) {
      std::cout << "__imf_vsetlts4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setgeu2_vals[idx] != setgeu2_ref_vals[idx]) {
      std::cout << "__imf_vsetgeu2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setgeu4_vals[idx] != setgeu4_ref_vals[idx]) {
      std::cout << "__imf_vsetgeu4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setltu2_vals[idx] != setltu2_ref_vals[idx]) {
      std::cout << "__imf_vsetltu2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setltu4_vals[idx] != setltu4_ref_vals[idx]) {
      std::cout << "__imf_vsetltu4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "__imf_vgelt_2_4 test pass." << std::endl;
}

void run_vgtle_2_4_test(s::queue &queue) {
  bool pass = true;
  static const size_t NUM = 5;
  unsigned input_x_vals[NUM] = {0x1, 0xFEDE1122, 0x12345678, 0xFFFFFFFF,
                                0x98765432};
  unsigned input_y_vals[NUM] = {0x2, 0xFEDF1133, 0x12345679, 0xFFFFFFFF,
                                0x87654321};
  unsigned output_cmpgts2_vals[NUM];
  unsigned output_cmpgts4_vals[NUM];
  unsigned output_cmples2_vals[NUM];
  unsigned output_cmples4_vals[NUM];
  unsigned output_cmpgtu2_vals[NUM];
  unsigned output_cmpgtu4_vals[NUM];
  unsigned output_cmpleu2_vals[NUM];
  unsigned output_cmpleu4_vals[NUM];
  unsigned output_setgts2_vals[NUM];
  unsigned output_setgts4_vals[NUM];
  unsigned output_setles2_vals[NUM];
  unsigned output_setles4_vals[NUM];
  unsigned output_setgtu2_vals[NUM];
  unsigned output_setgtu4_vals[NUM];
  unsigned output_setleu2_vals[NUM];
  unsigned output_setleu4_vals[NUM];
  unsigned cmpgts2_ref_vals[NUM] = {0, 0, 0, 0, 0xFFFFFFFF};
  unsigned cmpgts4_ref_vals[NUM] = {0, 0, 0, 0, 0xFFFFFFFF};
  unsigned cmples2_ref_vals[NUM] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                                    0xFFFFFFFF, 0};
  unsigned cmples4_ref_vals[NUM] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                                    0xFFFFFFFF, 0};

  unsigned cmpgtu2_ref_vals[NUM] = {0, 0, 0, 0, 0xFFFFFFFF};
  unsigned cmpgtu4_ref_vals[NUM] = {0, 0, 0, 0, 0xFFFFFFFF};
  unsigned cmpleu2_ref_vals[NUM] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                                    0xFFFFFFFF, 0};
  unsigned cmpleu4_ref_vals[NUM] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                                    0xFFFFFFFF, 0};
  unsigned setgts2_ref_vals[NUM] = {0, 0, 0, 0, 0x10001};
  unsigned setgts4_ref_vals[NUM] = {0, 0, 0, 0, 0x1010101};
  unsigned setles2_ref_vals[NUM] = {0x10001, 0x10001, 0x10001, 0x10001, 0};
  unsigned setles4_ref_vals[NUM] = {0x1010101, 0x1010101, 0x1010101, 0x1010101,
                                    0};
  unsigned setgtu2_ref_vals[NUM] = {0, 0, 0, 0, 0x10001};
  unsigned setgtu4_ref_vals[NUM] = {0, 0, 0, 0, 0x1010101};
  unsigned setleu2_ref_vals[NUM] = {0x10001, 0x10001, 0x10001, 0x10001, 0};
  unsigned setleu4_ref_vals[NUM] = {0x1010101, 0x1010101, 0x1010101, 0x1010101,
                                    0};

  {
    s::buffer<unsigned, 1> output_cmpgts2_buf(&output_cmpgts2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmpgts4_buf(&output_cmpgts4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmples2_buf(&output_cmples2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmples4_buf(&output_cmples4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmpgtu2_buf(&output_cmpgtu2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmpgtu4_buf(&output_cmpgtu4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmpleu2_buf(&output_cmpleu2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_cmpleu4_buf(&output_cmpleu4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setgts2_buf(&output_setgts2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setgts4_buf(&output_setgts4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setles2_buf(&output_setles2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setles4_buf(&output_setles4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setgtu2_buf(&output_setgtu2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setgtu4_buf(&output_setgtu4_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setleu2_buf(&output_setleu2_vals[0],
                                              s::range<1>(NUM));
    s::buffer<unsigned, 1> output_setleu4_buf(&output_setleu4_vals[0],
                                              s::range<1>(NUM));
    queue
        .submit([&](s::handler &cgh) {
          auto output_cmpgts2_acc =
              output_cmpgts2_buf.get_access<sycl_write>(cgh);
          auto output_cmpgts4_acc =
              output_cmpgts4_buf.get_access<sycl_write>(cgh);
          auto output_cmples2_acc =
              output_cmples2_buf.get_access<sycl_write>(cgh);
          auto output_cmples4_acc =
              output_cmples4_buf.get_access<sycl_write>(cgh);
          auto output_cmpgtu2_acc =
              output_cmpgtu2_buf.get_access<sycl_write>(cgh);
          auto output_cmpgtu4_acc =
              output_cmpgtu4_buf.get_access<sycl_write>(cgh);
          auto output_cmpleu2_acc =
              output_cmpleu2_buf.get_access<sycl_write>(cgh);
          auto output_cmpleu4_acc =
              output_cmpleu4_buf.get_access<sycl_write>(cgh);
          auto output_setgts2_acc =
              output_setgts2_buf.get_access<sycl_write>(cgh);
          auto output_setgts4_acc =
              output_setgts4_buf.get_access<sycl_write>(cgh);
          auto output_setles2_acc =
              output_setles2_buf.get_access<sycl_write>(cgh);
          auto output_setles4_acc =
              output_setles4_buf.get_access<sycl_write>(cgh);
          auto output_setgtu2_acc =
              output_setgtu2_buf.get_access<sycl_write>(cgh);
          auto output_setgtu4_acc =
              output_setgtu4_buf.get_access<sycl_write>(cgh);
          auto output_setleu2_acc =
              output_setleu2_buf.get_access<sycl_write>(cgh);
          auto output_setleu4_acc =
              output_setleu4_buf.get_access<sycl_write>(cgh);
          cgh.parallel_for<class vgtle_2_4_test>(
              s::range<1>(NUM), [=](s::id<1> I) {
                output_cmpgts2_acc[I[0]] =
                    __imf_vcmpgts2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmpgts4_acc[I[0]] =
                    __imf_vcmpgts4(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmples2_acc[I[0]] =
                    __imf_vcmples2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmples4_acc[I[0]] =
                    __imf_vcmples4(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmpgtu2_acc[I[0]] =
                    __imf_vcmpgtu2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmpgtu4_acc[I[0]] =
                    __imf_vcmpgtu4(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmpleu2_acc[I[0]] =
                    __imf_vcmpleu2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_cmpleu4_acc[I[0]] =
                    __imf_vcmpleu4(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setgts2_acc[I[0]] =
                    __imf_vsetgts2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setgts4_acc[I[0]] =
                    __imf_vsetgts4(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setles2_acc[I[0]] =
                    __imf_vsetles2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setles4_acc[I[0]] =
                    __imf_vsetles4(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setgtu2_acc[I[0]] =
                    __imf_vsetgtu2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setgtu4_acc[I[0]] =
                    __imf_vsetgtu4(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setleu2_acc[I[0]] =
                    __imf_vsetleu2(input_x_vals[I[0]], input_y_vals[I[0]]);
                output_setleu4_acc[I[0]] =
                    __imf_vsetleu4(input_x_vals[I[0]], input_y_vals[I[0]]);
              });
        })
        .wait();
  }

  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpgts2_vals[idx] != cmpgts2_ref_vals[idx]) {
      std::cout << "__imf_vcmpgts2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpgts4_vals[idx] != cmpgts4_ref_vals[idx]) {
      std::cout << "__imf_vcmpgts4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmples2_vals[idx] != cmples2_ref_vals[idx]) {
      std::cout << "__imf_vcmples2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmples4_vals[idx] != cmples4_ref_vals[idx]) {
      std::cout << "__imf_vcmples4 failed! idx = " << idx << std::endl;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpgtu2_vals[idx] != cmpgtu2_ref_vals[idx]) {
      std::cout << "__imf_vcmpgtu2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpgtu4_vals[idx] != cmpgtu4_ref_vals[idx]) {
      std::cout << "__imf_vcmpgtu4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpleu2_vals[idx] != cmpleu2_ref_vals[idx]) {
      std::cout << "__imf_vcmpleu2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_cmpleu4_vals[idx] != cmpleu4_ref_vals[idx]) {
      std::cout << "__imf_vcmpleu4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setgts2_vals[idx] != setgts2_ref_vals[idx]) {
      std::cout << "__imf_vsetgts2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setgts4_vals[idx] != setgts4_ref_vals[idx]) {
      std::cout << "__imf_vsetgts4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setles2_vals[idx] != setles2_ref_vals[idx]) {
      std::cout << "__imf_vsetles2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setles4_vals[idx] != setles4_ref_vals[idx]) {
      std::cout << "__imf_vsetles4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setgtu2_vals[idx] != setgtu2_ref_vals[idx]) {
      std::cout << "__imf_vsetgtu2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setgtu4_vals[idx] != setgtu4_ref_vals[idx]) {
      std::cout << "__imf_vsetgtu4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setleu2_vals[idx] != setleu2_ref_vals[idx]) {
      std::cout << "__imf_vsetleu2 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  for (size_t idx = 0; idx < NUM; ++idx) {
    if (output_setleu4_vals[idx] != setleu4_ref_vals[idx]) {
      std::cout << "__imf_vsetleu4 failed! idx = " << idx << std::endl;
      pass = false;
      break;
    }
  }
  assert(pass);
  std::cout << "__imf_vgtle_2_4 test pass." << std::endl;
}

int main(int, char **) {
  s::default_selector device_selector;
  s::queue device_queue(device_selector);
  std::cout << "Running on "
            << device_queue.get_device().get_info<s::info::device::name>()
            << "\n";
  run_vabs2_4_test(device_queue);
  run_vabsss2_4_test(device_queue);
  run_vabsdiffsu2_4_test(device_queue);
  run_vadd_ss_2_4_test(device_queue);
  run_vadd_us_2_4_test(device_queue);
  run_vsub_ss_2_4_test(device_queue);
  run_vsub_us_2_4_test(device_queue);
  run_vhaddu_2_4_test(device_queue);
  run_vavgu_2_4_test(device_queue);
  run_vcmpgts_2_4_test(device_queue);
  run_vmaxmin_2_4_test(device_queue);
  run_vneg_2_4_test(device_queue);
  run_vnegss_2_4_test(device_queue);
  run_vsad_2_4_test(device_queue);
  run_veqne_2_4_test(device_queue);
  run_vgelt_2_4_test(device_queue);
  run_vgtle_2_4_test(device_queue);
}
