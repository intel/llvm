// RUN: %clang_cc1  -fsycl-is-device -verify -fsyntax-only %s

void * fn_1 (const void *__ptr) { return (void *)(void __attribute__((address_space(1))) *)__ptr; }
void * fn_2 (const void *__ptr) { return (void *)(void __attribute__((address_space(3))) *)__ptr; }
void * fn_3 (const void *__ptr) { return (void *)(void __attribute__((address_space(4))) *)__ptr; }
void * fn_4 (const void *__ptr) { return (void *)(void __attribute__((address_space(5))) *)__ptr; }

void * fn_5 (const void *__ptr) { return (void __attribute__((address_space(3))) *)(void __attribute__((address_space(1))) *)__ptr; }
// expected-error@-1 {{C-style cast from '__attribute__((address_space(1))) void *' to '__attribute__((address_space(3))) void *' converts between mismatching address spaces}}
void * fn_6 (const void *__ptr) { return (void __attribute__((address_space(1))) *)(void __attribute__((address_space(3))) *)__ptr; }
// expected-error@-1 {{C-style cast from '__attribute__((address_space(3))) void *' to '__attribute__((address_space(1))) void *' converts between mismatching address spaces}}
void * fn_7 (const void *__ptr) { return (void __attribute__((address_space(5))) *)(void __attribute__((address_space(3))) *)__ptr; }
// expected-error@-1 {{C-style cast from '__attribute__((address_space(3))) void *' to '__attribute__((address_space(5))) void *' converts between mismatching address spaces}}
void * fn_8 (const void *__ptr) { return (void __attribute__((address_space(3))) *)(void __attribute__((address_space(5))) *)__ptr; }
// expected-error@-1 {{C-style cast from '__attribute__((address_space(5))) void *' to '__attribute__((address_space(3))) void *' converts between mismatching address spaces}}

