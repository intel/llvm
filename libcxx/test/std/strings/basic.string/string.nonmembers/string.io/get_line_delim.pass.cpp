//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<class charT, class traits, class Allocator>
//   basic_istream<charT,traits>&
//   getline(basic_istream<charT,traits>& is,
//           basic_string<charT,traits,Allocator>& str, charT delim);

#include <string>
#include <sstream>
#include <cassert>

#include "min_allocator.h"

int main(int, char**)
{
    {
        std::istringstream in(" abc*  def**   ghij");
        std::string s("initial text");
        getline(in, s, '*');
        assert(in.good());
        assert(s == " abc");
        getline(in, s, '*');
        assert(in.good());
        assert(s == "  def");
        getline(in, s, '*');
        assert(in.good());
        assert(s == "");
        getline(in, s, '*');
        assert(in.eof());
        assert(s == "   ghij");
    }
    {
        std::wistringstream in(L" abc*  def**   ghij");
        std::wstring s(L"initial text");
        getline(in, s, L'*');
        assert(in.good());
        assert(s == L" abc");
        getline(in, s, L'*');
        assert(in.good());
        assert(s == L"  def");
        getline(in, s, L'*');
        assert(in.good());
        assert(s == L"");
        getline(in, s, L'*');
        assert(in.eof());
        assert(s == L"   ghij");
    }
#if TEST_STD_VER >= 11
    {
        typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
        std::istringstream in(" abc*  def**   ghij");
        S s("initial text");
        getline(in, s, '*');
        assert(in.good());
        assert(s == " abc");
        getline(in, s, '*');
        assert(in.good());
        assert(s == "  def");
        getline(in, s, '*');
        assert(in.good());
        assert(s == "");
        getline(in, s, '*');
        assert(in.eof());
        assert(s == "   ghij");
    }
    {
        typedef std::basic_string<wchar_t, std::char_traits<wchar_t>, min_allocator<wchar_t>> S;
        std::wistringstream in(L" abc*  def**   ghij");
        S s(L"initial text");
        getline(in, s, L'*');
        assert(in.good());
        assert(s == L" abc");
        getline(in, s, L'*');
        assert(in.good());
        assert(s == L"  def");
        getline(in, s, L'*');
        assert(in.good());
        assert(s == L"");
        getline(in, s, L'*');
        assert(in.eof());
        assert(s == L"   ghij");
    }
#endif

  return 0;
}
