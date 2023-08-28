//===- DynamicTable.td ----------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the key top-level classes needed to produce a reasonably
// generic dynamic table that can be updated in runtime. DynamicTable objects
// can be defined using the class in this file:
// 1. (Dynamic) Tables. By instantiating the DynamicTable 
// class once, a table with the name of the instantiating def is generated and
// guarded by the GET_name_IMPL preprocessor guard.
//
//===----------------------------------------------------------------------===//

// Define a record derived from this class to generate a dynamic table. This
// table resembles a hashtable with a key-value pair, and can updated in runtime. 
//
// The name of the record is used as the name of the global primary array of
// entries of the table in C++.
class DynamicTable {
  // Name of a class. The table will have one entry for each record that
  // derives from that class.
  string FilterClass;

  // Name of the C++ struct/class type that holds table entries. The
  // declaration of this type is not generated automatically.
  string CppTypeName = FilterClass;

  // List of the names of fields of collected records that contain the data for
  // table entries, in the order that is used for initialization in C++.
  //
  // TableGen needs to know the type of the fields so that it can format
  // the initializers correctly. 
  //
  // For each field of the table named xxx, TableGen will look for a field
  // named TypeOf_xxx and use that as a more detailed description of the
  // type of the field.

  //   class MyTableEntry {
  //     MyEnum V;
  //     ...
  //   }
  //
  //   def MyTable : DynamicTable {
  //     let FilterClass = "MyTableEntry";
  //     let Fields = ["V", ...];
  //     string TypeOf_V = "list<int>";
  //   }
  list<string> Fields;
}
