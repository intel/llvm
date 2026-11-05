// Stands in for the integration header, which forward declares every free
// function kernel of the translation unit ahead of the application's code.
#pragma once

void before(int *p, double v);
void late(int *p, double v);
void late(int *p, float v);
template <typename T> void late_tmpl(int *p, T v);
void dep(int *p, double v);
void dep(int *p, float v);
template <typename T> void ffk(int *p, T v);
template <> void ffk<int>(int *p, int v);
template <typename T> void spec_after_use(int *p, T v);
template <> void spec_after_use<int>(int *p, int v);
template <typename T> void overload_after_use(int *p, T v);
void overload_after_use(int *p, double *v);
template <typename T> void template_after_use(int *p, T v);
template <typename T> void template_after_use(int *p, T *v);
