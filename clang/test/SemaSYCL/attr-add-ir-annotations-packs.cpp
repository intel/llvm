// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s

// Tests that __sycl_detail__::add_ir_annotations_member allows pack expansions
// in its arguments.

constexpr const char AttrName1[] = "Attr1";
constexpr const char AttrName2[] = "Attr2";
constexpr const char AttrName3[] = "Attr3";
constexpr const char AttrVal1[] = "Val1";
constexpr const char AttrVal2[] = "Val2";
constexpr const char AttrVal3[] = "Val3";

template <int... Is> struct ClassWithAnnotFieldTemplate1 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member("Attr1", "Attr2", "Attr3", Is...)]]; // expected-error {{attribute 'add_ir_annotations_member' must specify a value for each specified name in the argument list}}
};
template <int... Is> struct ClassWithAnnotFieldTemplate2 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, "Attr1", "Attr2", "Attr3", Is...)]]; // expected-error {{attribute 'add_ir_annotations_member' must specify a value for each specified name in the argument list}}
};
template <const char *...Names> struct ClassWithAnnotFieldTemplate3 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(Names..., 1, 2, 3)]]; // expected-error {{attribute 'add_ir_annotations_member' must specify a value for each specified name in the argument list}}
};
template <const char *...Names> struct ClassWithAnnotFieldTemplate4 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, Names..., 1, 2, 3)]]; // expected-error {{attribute 'add_ir_annotations_member' must specify a value for each specified name in the argument list}}
};
template <const char *...Strs> struct ClassWithAnnotFieldTemplate5 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member(Strs...)]]; // expected-error {{attribute 'add_ir_annotations_member' must specify a value for each specified name in the argument list}}
};
template <const char *...Strs> struct ClassWithAnnotFieldTemplate6 {
  int *ptr [[__sycl_detail__::add_ir_annotations_member({"Attr1", "Attr3"}, Strs...)]]; // expected-error {{attribute 'add_ir_annotations_member' must specify a value for each specified name in the argument list}}
};

void InstantiateClassWithAnnotFieldTemplates() {
  ClassWithAnnotFieldTemplate1<1, 2, 3> InstantiatedCWAFS1;
  ClassWithAnnotFieldTemplate1<1, 2> InstantiatedCWAFS2; // expected-note {{in instantiation of template class 'ClassWithAnnotFieldTemplate1<1, 2>' requested here}}

  ClassWithAnnotFieldTemplate2<1, 2, 3> InstantiatedCWAFS3;
  ClassWithAnnotFieldTemplate2<1, 2> InstantiatedCWAFS4; // expected-note {{in instantiation of template class 'ClassWithAnnotFieldTemplate2<1, 2>' requested here}}

  ClassWithAnnotFieldTemplate3<AttrName1, AttrName2, AttrName3> InstantiatedCWAFS5;
  ClassWithAnnotFieldTemplate3<AttrName1, AttrName2> InstantiatedCWAFS6; // expected-note {{in instantiation of template class 'ClassWithAnnotFieldTemplate3<AttrName1, AttrName2>' requested here}}

  ClassWithAnnotFieldTemplate4<AttrName1, AttrName2, AttrName3> InstantiatedCWAFS7;
  ClassWithAnnotFieldTemplate4<AttrName1, AttrName2> InstantiatedCWAFS8; // expected-note {{in instantiation of template class 'ClassWithAnnotFieldTemplate4<AttrName1, AttrName2>' requested here}}

  ClassWithAnnotFieldTemplate5<AttrName1, AttrVal1> InstantiatedCWAFS9;
  ClassWithAnnotFieldTemplate5<AttrName1, AttrName2, AttrVal1, AttrVal2> InstantiatedCWAFS10;
  ClassWithAnnotFieldTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> InstantiatedCWAFS11;
  ClassWithAnnotFieldTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> InstantiatedCWAFS12; // expected-note {{in instantiation of template class 'ClassWithAnnotFieldTemplate5<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2>' requested here}}

  ClassWithAnnotFieldTemplate6<AttrName1, AttrVal1> InstantiatedCWAFS13;
  ClassWithAnnotFieldTemplate6<AttrName1, AttrName2, AttrVal1, AttrVal2> InstantiatedCWAFS14;
  ClassWithAnnotFieldTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2, AttrVal3> InstantiatedCWAFS15;
  ClassWithAnnotFieldTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2> InstantiatedCWAFS16; // expected-note {{in instantiation of template class 'ClassWithAnnotFieldTemplate6<AttrName1, AttrName2, AttrName3, AttrVal1, AttrVal2>' requested here}}

  (void)*InstantiatedCWAFS1.ptr;
  (void)*InstantiatedCWAFS2.ptr;
  (void)*InstantiatedCWAFS3.ptr;
  (void)*InstantiatedCWAFS4.ptr;
  (void)*InstantiatedCWAFS5.ptr;
  (void)*InstantiatedCWAFS6.ptr;
  (void)*InstantiatedCWAFS7.ptr;
  (void)*InstantiatedCWAFS8.ptr;
  (void)*InstantiatedCWAFS9.ptr;
  (void)*InstantiatedCWAFS10.ptr;
  (void)*InstantiatedCWAFS11.ptr;
  (void)*InstantiatedCWAFS12.ptr;
  (void)*InstantiatedCWAFS13.ptr;
  (void)*InstantiatedCWAFS14.ptr;
  (void)*InstantiatedCWAFS15.ptr;
  (void)*InstantiatedCWAFS16.ptr;
}
