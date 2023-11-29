// RUN: llvm-tblgen -gen-dynamic-tables -I %p/../../include %s | FileCheck %s
// RUN: not llvm-tblgen -gen-dynamic-tables -I %p/../../include -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s
// XFAIL: vg_leak

include "llvm/TableGen/DynamicTable.td"

// CHECK-LABEL: GET_ATable_IMPL
// CHECK: std::map<std::string, AEntry> ATable = {
// CHECK:   { "bar", {0x5, 0x3 }}, // 0
// CHECK:   { "baz", {0x2, 0x6 }}, // 1
// CHECK:   { "foo", {0x4, 0x4 }}, // 2
// CHECK:   { "foobar", {0x4, 0x5 }}, // 3
// CHECK: };

class AEntry<string str, int val1, int val2> {
  string Str = str;
  bits<8> Val1 = val1;
  bits<10> Val2 = val2;
}

def : AEntry<"bar",    5, 3>;
def : AEntry<"baz",    2, 6>;
def : AEntry<"foo",    4, 4>;
def : AEntry<"foobar", 4, 5>;

def ATable : DynamicTable {
  let FilterClass = "AEntry";
  let Fields = ["Str", "Val1", "Val2"];
}


// CHECK-LABEL: GET_BTable_IMPL
// CHECK: std::map<std::string, BTypeName> BTable = {
// CHECK:   { "BAlice", {0xAC, false,  }}, // 0
// CHECK:   { "BBob", {0x14, false, Bob == 13 }}, // 1
// CHECK:   { "BCharlie", {0x80, true, Charlie == 42 }}, // 2
// CHECK:   { "BEve", {0x4C, true, Eve == 108 }}, // 3
// CHECK:  };

class BEntry<bits<16> enc, bit flag = 0, code test = [{}]> {
  string Name = NAME;
  bits<16> Encoding = enc;
  bit Flag = flag;
  code Test = test;
}

def BAlice   : BEntry<0xac>;
def BBob     : BEntry<0x14, 0, [{Bob == 13}]>;
def BCharlie : BEntry<0x80, 1, "Charlie == 42">;
def BEve     : BEntry<0x4c, 1, [{Eve == }] # 108>;

def BTable : DynamicTable {
  let FilterClass = "BEntry";
  string CppTypeName = "BTypeName";
  let Fields = ["Name", "Encoding", "Flag", "Test"];
  string TypeOf_Test = "code";
}

// CHECK-LABEL: GET_CTable_IMPL
// CHECK: std::map<std::string, CEntry> CTable = {
// CHECK:   { "alice", {0xA }}, // 0
// CHECK:   { "alice", {0xD }}, // 1
// CHECK:   { "bob", {0xF }}, // 2
// CHECK:  };

class CEntry<string name, int enc> {
  string Name = name;
  bits<16> Encoding = enc;
}

def : CEntry<"alice", 10>;
def : CEntry<"alice", 13>;
def : CEntry<"bob",   15>;

def CTable : DynamicTable {
  let FilterClass = "CEntry";
  let Fields = ["Name", "Encoding"];
}

// CHECK-LABEL: GET_DTable_IMPL
// CHECK: std::map<std::string, DEntry> DTable = {
// CHECK:  };

class DEntry<string name> {
    string Name = name;
}

def DTable : DynamicTable {
    let FilterClass = "DEntry";
    let Fields = ["Name"];
}

#ifdef ERROR1

class EEntry<string str, int val1> {
  string Str = str;
  bits<8> Val1 = val1;
}

def EFoo : EEntry<"foo", 1>;
// ERROR1: [[@LINE+1]]:5: error: Record 'EBar' for table 'ETable' is missing field 'Val1'
def EBar : EEntry<"bar", ?>;

def ETable : DynamicTable {
  let FilterClass = "EEntry";
  let Fields = ["Str", "Val1"];
}

#endif // ERROR1
