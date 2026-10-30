// Stands in for the integration header, which forward declares every free
// function kernel of the translation unit ahead of the application's code.
#pragma once

void before(int *p, double v);
void late(int *p, double v);
void late(int *p, float v);
template <typename T> void late_tmpl(int *p, T v);
void dep(int *p, double v);
void dep(int *p, float v);
