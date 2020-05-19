// RUN: %clang_cc1 -fsycl -fsycl-is-device -verify -fsyntax-only %s

void bar(int & Data) {}
void bar2(int & Data) {}
void bar(__attribute__((opencl_private)) int  & Data) {}
void bar3(__attribute__((opencl_global)) int &Data) {}
void foo(int * Data) {}
void foo2(int * Data) {}
void foo(__attribute__((opencl_private)) int * Data) {}
void foo3(__attribute__((opencl_global)) int *Data) {}

template<typename T>
void tmpl(T *t){}

void usages() {
  __attribute__((opencl_global)) int *GLOB;
  __attribute__((opencl_usm_device)) int *USMDEV;
  __attribute__((opencl_usm_host)) int *USMHOST;
  __attribute__((opencl_private)) int *PRIV;
  __attribute__((opencl_local)) int *LOC;
  int *NoAS;

  bar(*GLOB);
  bar2(*GLOB);

  bar(*USMDEV);
  bar2(*USMDEV);
  bar3(*USMDEV);

  bar(*USMHOST);
  bar2(*USMHOST);
  bar3(*USMHOST);

  bar(*PRIV);
  bar2(*PRIV);

  bar(*NoAS);
  bar2(*NoAS);

  bar(*LOC);
  bar2(*LOC);

  foo(GLOB);
  foo2(GLOB);
  foo(USMDEV);
  foo2(USMDEV);
  foo3(USMDEV);
  foo(USMDEV);
  foo2(USMHOST);
  foo3(USMHOST);
  foo(PRIV);
  foo2(PRIV);
  foo(NoAS);
  foo2(NoAS);
  foo(LOC);
  foo2(LOC);

  tmpl(GLOB);
  tmpl(USMDEV);
  tmpl(USMHOST);
  tmpl(PRIV);
  tmpl(NoAS);
  tmpl(LOC);

  (void)static_cast<int*>(GLOB);
  (void)static_cast<void*>(GLOB);
  (void)static_cast<int *>(USMDEV);
  (void)static_cast<void *>(USMDEV);
  (void)static_cast<int *>(USMHOST);
  (void)static_cast<void *>(USMHOST);
  // FIXME: determine if we can warn on the below conversions.
  int *i = GLOB;
  void *v = GLOB;
  (void)i;
  (void)v;


  // expected-error@+1{{address space is negative}}
  __attribute__((address_space(-1))) int *TooLow;
  // expected-error@+1{{unknown type name '__generic'}}
  __generic int *IsGeneric;

}
