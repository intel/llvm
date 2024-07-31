// REQUIRES: gpu
// RUN: %{build} -o %t.exe
// RUN: %{run} %t.exe 2> %t.out || true

// RUN: FileCheck %s --check-prefix=CHECK-EXPECTED-ERROR --input-file %t.out
// CHECK-EXPECTED-ERROR: error: backend compiler failed build

// XFAIL: *

#include <stdio.h>
#include <sycl/detail/core.hpp>

long long int app_var_3 = 7450649278945924256LL;
long long int app_var_4 = -7822479480217888654LL;
unsigned long long int app_var_6 = 16042174700188974009ULL;
unsigned char app_var_9 = (unsigned char)209;
bool app_var_10 = (bool)1;
unsigned int app_var_11 = 3357916866U;
int app_var_12 = 1490439599;
bool app_var_13 = (bool)0;
unsigned short app_var_14 = (unsigned short)13255;
signed char app_var_15 = (signed char)-82;
short app_var_16 = (short)-9459;
unsigned short app_var_17 = (unsigned short)33701;
long long int app_var_18 = -8262062769670300563LL;
unsigned long long int app_var_19 = 16817880494818592221ULL;
short app_var_20 = (short)10201;
short app_var_21 = (short)20564;
bool app_var_22 = (bool)1;
unsigned char app_var_23 = (unsigned char)21;
unsigned char app_var_24 = (unsigned char)69;
unsigned short app_var_25 = (unsigned short)35772;
unsigned int app_var_26 = 3471154395U;
unsigned char app_var_27 = (unsigned char)14;
int app_var_28 = -402832105;
unsigned char app_var_29 = (unsigned char)64;
unsigned int app_var_30 = 2041401567U;
unsigned long long int app_var_31 = 16124465128117296508ULL;
unsigned char app_var_32 = (unsigned char)129;
signed char app_var_33 = (signed char)-87;
int app_var_34 = 1679760643;
unsigned char app_var_35 = (unsigned char)71;
unsigned char app_var_36 = (unsigned char)245;
unsigned short app_var_37 = (unsigned short)9667;
long long int app_var_38 = -5600606352884923795LL;

void test() {
  using namespace sycl;
#if CPU
  cpu_selector a;
#else
  gpu_selector a;
#endif
  queue b(a);
  buffer c{&app_var_3, range(1)};
  buffer d{&app_var_4, range(1)};
  buffer e{&app_var_6, range(1)};
  buffer f{&app_var_9, range(1)};
  buffer g{&app_var_10, range(1)};
  buffer h{&app_var_11, range(1)};
  buffer i{&app_var_12, range(1)};
  buffer j{&app_var_13, range(1)};
  buffer k{&app_var_14, range(1)};
  buffer l{&app_var_15, range(1)};
  buffer m{&app_var_16, range(1)};
  buffer n{&app_var_17, range(1)};
  buffer o{&app_var_18, range(1)};
  buffer p{&app_var_19, range(1)};
  buffer q{&app_var_20, range(1)};
  buffer r{&app_var_21, range(1)};
  buffer s{&app_var_22, range(1)};
  buffer t{&app_var_23, range(1)};
  buffer u{&app_var_24, range(1)};
  buffer v{&app_var_25, range(1)};
  buffer w{&app_var_26, range(1)};
  buffer x{&app_var_27, range(1)};
  buffer y{&app_var_28, range(1)};
  buffer z{&app_var_29, range(1)};
  buffer aa{&app_var_30, range(1)};
  buffer ab{&app_var_31, range(1)};
  buffer ac{&app_var_32, range(1)};
  buffer ad{&app_var_33, range(1)};
  buffer ae{&app_var_34, range(1)};
  buffer af{&app_var_35, range(1)};
  buffer ag{&app_var_36, range(1)};
  buffer ah{&app_var_37, range(1)};
  buffer ai{&app_var_38, range(1)};
  b.submit([&](handler &aj) {
    auto ak = c.get_access<access::mode::read>(aj);
    auto al = d.get_access<access::mode::read>(aj);
    auto am = e.get_access<access::mode::read>(aj);
    auto an = f.get_access<access::mode::read>(aj);
    auto ao = g.get_access<access::mode::read>(aj);
    auto ap = h.get_access<access::mode::read>(aj);
    auto aq = i.get_access<access::mode::read>(aj);
    auto ar = j.get_access<access::mode::read>(aj);
    auto as = k.get_access<access::mode::read>(aj);
    auto at = l.get_access<access::mode::read>(aj);
    auto au = m.get_access<access::mode::write>(aj);
    auto av = n.get_access<access::mode::write>(aj);
    auto aw = o.get_access<access::mode::write>(aj);
    auto ax = p.get_access<access::mode::write>(aj);
    auto ay = q.get_access<access::mode::write>(aj);
    auto az = r.get_access<access::mode::write>(aj);
    auto ba = s.get_access<access::mode::write>(aj);
    auto bb = t.get_access<access::mode::write>(aj);
    auto bc = u.get_access<access::mode::write>(aj);
    auto bd = v.get_access<access::mode::write>(aj);
    auto be = w.get_access<access::mode::write>(aj);
    auto bf = x.get_access<access::mode::write>(aj);
    auto bg = y.get_access<access::mode::write>(aj);
    auto bh = z.get_access<access::mode::write>(aj);
    auto bi = aa.get_access<access::mode::write>(aj);
    auto bj = ab.get_access<access::mode::write>(aj);
    auto bk = ac.get_access<access::mode::write>(aj);
    auto bl = ad.get_access<access::mode::write>(aj);
    auto bm = ae.get_access<access::mode::write>(aj);
    auto bn = af.get_access<access::mode::write>(aj);
    auto bo = ag.get_access<access::mode::write>(aj);
    auto bp = ah.get_access<access::mode::write>(aj);
    auto bq = ai.get_access<access::mode::write>(aj);
    aj.single_task<class br>([=] {
      au[0] = am[0];
      if (ao[0] ? as[0] ? ar[0] : 9 : ap[0])
        av;
      aw;
      if (aq[0])
        if (an[0])
          ax;
      ay;
      az;
      ba[0] = al[0];
      bb[0] = at[0];
      bc;
      bd;
      be;
      bf;
      bg;
      bh;
      bi;
      bj;
      bk;
      bl;
      bm;
      if (ak[0])
        bn;
      bo;
      bp;
      bq;
    });
  });
}

int main() { test(); }
