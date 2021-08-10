#ifndef SPIRV_DEBUG_H
#define SPIRV_DEBUG_H
#include "SPIRVUtil.h"
#include "spirv/unified1/spirv.hpp"
#include "llvm/BinaryFormat/Dwarf.h"

namespace SPIRVDebug {

const unsigned int DebugInfoVersion = 0x00010000;
static const std::string ProducerPrefix = {"Debug info producer: "};
static const std::string ChecksumKindPrefx = {"//__CSK_"};

// clang-format off

enum Instruction {
  DebugInfoNone                 = 0,
  CompilationUnit               = 1,
  TypeBasic                     = 2,
  TypePointer                   = 3,
  TypeQualifier                 = 4,
  TypeArray                     = 5,
  TypeVector                    = 6,
  Typedef                       = 7,
  TypeFunction                  = 8,
  TypeEnum                      = 9,
  TypeComposite                 = 10,
  TypeMember                    = 11,
  Inheritance                   = 12,
  TypePtrToMember               = 13,
  TypeTemplate                  = 14,
  TypeTemplateParameter         = 15,
  TypeTemplateParameterPack     = 16,
  TypeTemplateTemplateParameter = 17,
  GlobalVariable                = 18,
  FunctionDecl                  = 19,
  Function                      = 20,
  LexicalBlock                  = 21,
  LexicalBlockDiscriminator     = 22,
  Scope                         = 23,
  NoScope                       = 24,
  InlinedAt                     = 25,
  LocalVariable                 = 26,
  InlinedVariable               = 27,
  Declare                       = 28,
  Value                         = 29,
  Operation                     = 30,
  Expression                    = 31,
  MacroDef                      = 32,
  MacroUndef                    = 33,
  ImportedEntity                = 34,
  Source                        = 35,
  ModuleINTEL                   = 36,
  InstCount                     = 37
};

enum Flag {
  FlagIsProtected         = 1 << 0,
  FlagIsPrivate           = 1 << 1,
  FlagIsPublic            = FlagIsPrivate | FlagIsProtected,
  FlagAccess              = FlagIsPublic,
  FlagIsLocal             = 1 << 2,
  FlagIsDefinition        = 1 << 3,
  FlagIsFwdDecl           = 1 << 4,
  FlagIsArtificial        = 1 << 5,
  FlagIsExplicit          = 1 << 6,
  FlagIsPrototyped        = 1 << 7,
  FlagIsObjectPointer     = 1 << 8,
  FlagIsStaticMember      = 1 << 9,
  FlagIsIndirectVariable  = 1 << 10,
  FlagIsLValueReference   = 1 << 11,
  FlagIsRValueReference   = 1 << 12,
  FlagIsOptimized         = 1 << 13,
  FlagIsEnumClass         = 1 << 14,
  FlagTypePassByValue     = 1 << 15,
  FlagTypePassByReference = 1 << 16,
};

enum EncodingTag {
  Unspecified  = 0,
  Address      = 1,
  Boolean      = 2,
  Float        = 3,
  Signed       = 4,
  SignedChar   = 5,
  Unsigned     = 6,
  UnsignedChar = 7
};

enum CompositeTypeTag {
  Class     = 0,
  Structure = 1,
  Union     = 2
};

enum TypeQualifierTag {
  ConstType    = 0,
  VolatileType = 1,
  RestrictType = 2,
  AtomicType   = 3
};

enum ExpressionOpCode {
  Deref      = 0,
  Plus       = 1,
  Minus      = 2,
  PlusUconst = 3,
  BitPiece   = 4,
  Swap       = 5,
  Xderef     = 6,
  StackValue = 7,
  Constu     = 8,
  Fragment   = 9,
  Convert    = 10,
  Addr       = 11,
  Const1u    = 12,
  Const1s    = 13,
  Const2u    = 14,
  Const2s    = 15,
  Const4u    = 16,
  Const4s    = 17,
  Const8u    = 18,
  Const8s    = 19,
  Consts     = 20,
  Dup        = 21,
  Drop       = 22,
  Over       = 23,
  Pick       = 24,
  Rot        = 25,
  Abs        = 26,
  And        = 27,
  Div        = 28,
  Mod        = 29,
  Mul        = 30,
  Neg        = 31,
  Not        = 32,
  Or         = 33,
  Shl        = 34,
  Shr        = 35,
  Shra       = 36,
  Xor        = 37,
  Bra        = 38,
  Eq         = 39,
  Ge         = 40,
  Gt         = 41,
  Le         = 42,
  Lt         = 43,
  Ne         = 44,
  Skip       = 45,
  Lit0       = 46,
  Lit1       = 47,
  Lit2       = 48,
  Lit3       = 49,
  Lit4       = 50,
  Lit5       = 51,
  Lit6       = 52,
  Lit7       = 53,
  Lit8       = 54,
  Lit9       = 55,
  Lit10      = 56,
  Lit11      = 57,
  Lit12      = 58,
  Lit13      = 59,
  Lit14      = 60,
  Lit15      = 61,
  Lit16      = 62,
  Lit17      = 63,
  Lit18      = 64,
  Lit19      = 65,
  Lit20      = 66,
  Lit21      = 67,
  Lit22      = 68,
  Lit23      = 69,
  Lit24      = 70,
  Lit25      = 71,
  Lit26      = 72,
  Lit27      = 73,
  Lit28      = 74,
  Lit29      = 75,
  Lit30      = 76,
  Lit31      = 77,
  Reg0       = 78,
  Reg1       = 79,
  Reg2       = 80,
  Reg3       = 81,
  Reg4       = 82,
  Reg5       = 83,
  Reg6       = 84,
  Reg7       = 85,
  Reg8       = 86,
  Reg9       = 87,
  Reg10      = 88,
  Reg11      = 89,
  Reg12      = 90,
  Reg13      = 91,
  Reg14      = 92,
  Reg15      = 93,
  Reg16      = 94,
  Reg17      = 95,
  Reg18      = 96,
  Reg19      = 97,
  Reg20      = 98,
  Reg21      = 99,
  Reg22      = 100,
  Reg23      = 101,
  Reg24      = 102,
  Reg25      = 103,
  Reg26      = 104,
  Reg27      = 105,
  Reg28      = 106,
  Reg29      = 107,
  Reg30      = 108,
  Reg31      = 109,
  Breg0      = 110,
  Breg1      = 111,
  Breg2      = 112,
  Breg3      = 113,
  Breg4      = 114,
  Breg5      = 115,
  Breg6      = 116,
  Breg7      = 117,
  Breg8      = 118,
  Breg9      = 119,
  Breg10     = 120,
  Breg11     = 121,
  Breg12     = 122,
  Breg13     = 123,
  Breg14     = 124,
  Breg15     = 125,
  Breg16     = 126,
  Breg17     = 127,
  Breg18     = 128,
  Breg19     = 129,
  Breg20     = 130,
  Breg21     = 131,
  Breg22     = 132,
  Breg23     = 133,
  Breg24     = 134,
  Breg25     = 135,
  Breg26     = 136,
  Breg27     = 137,
  Breg28     = 138,
  Breg29     = 139,
  Breg30     = 140,
  Breg31     = 141,
  Regx       = 142,
  Fbreg      = 143,
  Bregx      = 144,
  Piece      = 145,
  DerefSize  = 146,
  XderefSize = 147,
  Nop        = 148,
  PushObjectAddress = 149,
  Call2             = 150,
  Call4             = 151,
  CallRef           = 152,
  FormTlsAddress    = 153,
  CallFrameCfa      = 154,
  ImplicitValue     = 155,
  ImplicitPointer   = 156,
  Addrx             = 157,
  Constx            = 158,
  EntryValue        = 159,
  ConstTypeOp       = 160,
  RegvalType        = 161,
  DerefType         = 162,
  XderefType        = 163,
  Reinterpret       = 164
};

enum ImportedEntityTag {
  ImportedModule      = 0,
  ImportedDeclaration = 1,
};

namespace Operand {

namespace CompilationUnit {
enum {
  SPIRVDebugInfoVersionIdx = 0,
  DWARFVersionIdx          = 1,
  SourceIdx                = 2,
  LanguageIdx              = 3,
  OperandCount             = 4
};
}

namespace Source {
enum {
  FileIdx      = 0,
  TextIdx      = 1,
  OperandCount = 2
};
}

namespace TypeBasic {
enum {
  NameIdx      = 0,
  SizeIdx      = 1,
  EncodingIdx  = 2,
  OperandCount = 3
};
}

namespace TypePointer {
enum {
  BaseTypeIdx     = 0,
  StorageClassIdx = 1,
  FlagsIdx        = 2,
  OperandCount    = 3
};
}

namespace TypeQualifier {
enum {
  BaseTypeIdx  = 0,
  QualifierIdx = 1,
  OperandCount = 2
};
}

namespace TypeArray {
enum {
  BaseTypeIdx       = 0,
  ComponentCountIdx = 1,
  MinOperandCount   = 2
};
}

namespace TypeVector = TypeArray;

namespace Typedef {
enum {
  NameIdx      = 0,
  BaseTypeIdx  = 1,
  SourceIdx    = 2,
  LineIdx      = 3,
  ColumnIdx    = 4,
  ParentIdx    = 5,
  OperandCount = 6
};
}

namespace TypeFunction {
enum {
  FlagsIdx          = 0,
  ReturnTypeIdx     = 1,
  FirstParameterIdx = 2,
  MinOperandCount   = 2
};
}

namespace TypeEnum {
enum {
  NameIdx            = 0,
  UnderlyingTypeIdx  = 1,
  SourceIdx          = 2,
  LineIdx            = 3,
  ColumnIdx          = 4,
  ParentIdx          = 5,
  SizeIdx            = 6,
  FlagsIdx           = 7,
  FirstEnumeratorIdx = 8,
  MinOperandCount    = 8
};
}

namespace TypeComposite {
enum {
  NameIdx         = 0,
  TagIdx          = 1,
  SourceIdx       = 2,
  LineIdx         = 3,
  ColumnIdx       = 4,
  ParentIdx       = 5,
  LinkageNameIdx  = 6,
  SizeIdx         = 7,
  FlagsIdx        = 8,
  FirstMemberIdx  = 9,
  MinOperandCount = 9
};
}

namespace TypeMember {
enum {
  NameIdx         = 0,
  TypeIdx         = 1,
  SourceIdx       = 2,
  LineIdx         = 3,
  ColumnIdx       = 4,
  ParentIdx       = 5,
  OffsetIdx       = 6,
  SizeIdx         = 7,
  FlagsIdx        = 8,
  ValueIdx        = 9,
  MinOperandCount = 9
};
}

namespace TypeInheritance {
enum {
  ChildIdx     = 0,
  ParentIdx    = 1,
  OffsetIdx    = 2,
  SizeIdx      = 3,
  FlagsIdx     = 4,
  OperandCount = 5
};
}

namespace PtrToMember {
enum {
  MemberTypeIdx = 0,
  ParentIdx     = 1,
  OperandCount  = 2
};
}

namespace Template {
enum {
  TargetIdx         = 0,
  FirstParameterIdx = 1,
  MinOperandCount   = 1
};
}

namespace TemplateParameter {
enum {
  NameIdx      = 0,
  TypeIdx      = 1,
  ValueIdx     = 2,
  SourceIdx    = 3,
  LineIdx      = 4,
  ColumnIdx    = 5,
  OperandCount = 6
};
}

namespace TemplateTemplateParameter {
enum {
  NameIdx         = 0,
  TemplateNameIdx = 1,
  SourceIdx       = 2,
  LineIdx         = 3,
  ColumnIdx       = 4,
  OperandCount    = 5
};
}

namespace TemplateParameterPack {
enum {
  NameIdx           = 0,
  SourceIdx         = 1,
  LineIdx           = 2,
  ColumnIdx         = 3,
  FirstParameterIdx = 4,
  MinOperandCount   = 4
};
}

namespace GlobalVariable {
enum {
  NameIdx                    = 0,
  TypeIdx                    = 1,
  SourceIdx                  = 2,
  LineIdx                    = 3,
  ColumnIdx                  = 4,
  ParentIdx                  = 5,
  LinkageNameIdx             = 6,
  VariableIdx                = 7,
  FlagsIdx                   = 8,
  StaticMemberDeclarationIdx = 9,
  MinOperandCount            = 9
};
}

namespace FunctionDeclaration {
enum {
  NameIdx        = 0,
  TypeIdx        = 1,
  SourceIdx      = 2,
  LineIdx        = 3,
  ColumnIdx      = 4,
  ParentIdx      = 5,
  LinkageNameIdx = 6,
  FlagsIdx       = 7,
  OperandCount   = 8
};
}

namespace Function {
enum {
  NameIdx         = 0,
  TypeIdx         = 1,
  SourceIdx       = 2,
  LineIdx         = 3,
  ColumnIdx       = 4,
  ParentIdx       = 5,
  LinkageNameIdx  = 6,
  FlagsIdx        = 7,
  ScopeLineIdx    = 8,
  FunctionIdIdx   = 9,
  DeclarationIdx  = 10,
  MinOperandCount = 10
};
}

namespace LexicalBlock {
enum {
  SourceIdx       = 0,
  LineIdx         = 1,
  ColumnIdx       = 2,
  ParentIdx       = 3,
  NameIdx         = 4,
  MinOperandCount = 4
};
}

namespace LexicalBlockDiscriminator {
enum {
  SourceIdx        = 0,
  DiscriminatorIdx = 1,
  ParentIdx        = 2,
  OperandCount     = 3
};
}

namespace Scope {
enum {
  ScopeIdx        = 0,
  InlinedAtIdx    = 1,
  MinOperandCount = 1
};
}

namespace NoScope {
// No operands
}

namespace InlinedAt {
enum {
  LineIdx         = 0,
  ScopeIdx        = 1,
  InlinedIdx      = 2,
  MinOperandCount = 2
};
}

namespace LocalVariable {
enum {
  NameIdx         = 0,
  TypeIdx         = 1,
  SourceIdx       = 2,
  LineIdx         = 3,
  ColumnIdx       = 4,
  ParentIdx       = 5,
  FlagsIdx        = 6,
  ArgNumberIdx    = 7,
  MinOperandCount = 7
};
}

namespace InlinedVariable {
enum {
  VariableIdx  = 0,
  InlinedIdx   = 1,
  OperandCount = 2
};
}

namespace DebugDeclare {
enum {
  DebugLocalVarIdx = 0,
  VariableIdx      = 1,
  ExpressionIdx    = 2,
  OperandCount     = 3
};
}

namespace DebugValue {
enum {
  DebugLocalVarIdx     = 0,
  ValueIdx             = 1,
  ExpressionIdx        = 2,
  FirstIndexOperandIdx = 3,
  MinOperandCount      = 3
};
}

namespace Operation {
enum {
  OpCodeIdx = 0
};
static std::map<ExpressionOpCode, unsigned> OpCountMap {
  { Deref,              1 },
  { Plus,               1 },
  { Minus,              1 },
  { PlusUconst,         2 },
  { BitPiece,           3 },
  { Swap,               1 },
  { Xderef,             1 },
  { StackValue,         1 },
  { Constu,             2 },
  { Fragment,           3 },
  { Convert,            3 },
  // { Addr,               2 }, /* not implemented */
  // { Const1u,            2 },
  // { Const1s,            2 },
  // { Const2u,            2 },
  // { Const2s,            2 },
  // { Const4u,            2 },
  // { Const4s,            2 },
  // { Const8u,            2 },
  // { Const8s,            2 },
  { Consts,             2 },
  { Dup,                1 },
  { Drop,               1 },
  { Over,               1 },
  { Pick,               1 },
  { Rot,                1 },
  { Abs,                1 },
  { And,                1 },
  { Div,                1 },
  { Mod,                1 },
  { Mul,                1 },
  { Neg,                1 },
  { Not,                1 },
  { Or,                 1 },
  { Shl,                1 },
  { Shr,                1 },
  { Shra,               1 },
  { Xor,                1 },
  // { Bra,                2 }, /* not implemented */
  { Eq,                 1 },
  { Ge,                 1 },
  { Gt,                 1 },
  { Le,                 1 },
  { Lt,                 1 },
  { Ne,                 1 },
  // { Skip,               2 }, /* not implemented */
  { Lit0,               1 },
  { Lit1,               1 },
  { Lit2,               1 },
  { Lit3,               1 },
  { Lit4,               1 },
  { Lit5,               1 },
  { Lit6,               1 },
  { Lit7,               1 },
  { Lit8,               1 },
  { Lit9,               1 },
  { Lit10,              1 },
  { Lit11,              1 },
  { Lit12,              1 },
  { Lit13,              1 },
  { Lit14,              1 },
  { Lit15,              1 },
  { Lit16,              1 },
  { Lit17,              1 },
  { Lit18,              1 },
  { Lit19,              1 },
  { Lit20,              1 },
  { Lit21,              1 },
  { Lit22,              1 },
  { Lit23,              1 },
  { Lit24,              1 },
  { Lit25,              1 },
  { Lit26,              1 },
  { Lit27,              1 },
  { Lit28,              1 },
  { Lit29,              1 },
  { Lit30,              1 },
  { Lit31,              1 },
  { Reg0,               1 },
  { Reg1,               1 },
  { Reg2,               1 },
  { Reg3,               1 },
  { Reg4,               1 },
  { Reg5,               1 },
  { Reg6,               1 },
  { Reg7,               1 },
  { Reg8,               1 },
  { Reg9,               1 },
  { Reg10,              1 },
  { Reg11,              1 },
  { Reg12,              1 },
  { Reg13,              1 },
  { Reg14,              1 },
  { Reg15,              1 },
  { Reg16,              1 },
  { Reg17,              1 },
  { Reg18,              1 },
  { Reg19,              1 },
  { Reg20,              1 },
  { Reg21,              1 },
  { Reg22,              1 },
  { Reg23,              1 },
  { Reg24,              1 },
  { Reg25,              1 },
  { Reg26,              1 },
  { Reg27,              1 },
  { Reg28,              1 },
  { Reg29,              1 },
  { Reg30,              1 },
  { Reg31,              1 },
  { Breg0,              2 },
  { Breg1,              2 },
  { Breg2,              2 },
  { Breg3,              2 },
  { Breg4,              2 },
  { Breg5,              2 },
  { Breg6,              2 },
  { Breg7,              2 },
  { Breg8,              2 },
  { Breg9,              2 },
  { Breg10,             2 },
  { Breg11,             2 },
  { Breg12,             2 },
  { Breg13,             2 },
  { Breg14,             2 },
  { Breg15,             2 },
  { Breg16,             2 },
  { Breg17,             2 },
  { Breg18,             2 },
  { Breg19,             2 },
  { Breg20,             2 },
  { Breg21,             2 },
  { Breg22,             2 },
  { Breg23,             2 },
  { Breg24,             2 },
  { Breg25,             2 },
  { Breg26,             2 },
  { Breg27,             2 },
  { Breg28,             2 },
  { Breg29,             2 },
  { Breg30,             2 },
  { Breg31,             2 },
  { Regx,               2 },
  // { Fbreg,              1 }, /* not implemented */
  { Bregx,              3 },
  // { Piece,              2 }, /* not implemented */
  { DerefSize,          2 },
  { XderefSize,         2 },
  { Nop,                1 },
  { PushObjectAddress,  1 },
  // { Call2,              2 }, /* not implemented */
  // { Call4,              2 },
  // { CallRef,            2 },
  // { FormTlsAddress,     1 },
  // { CallFrameCfa,       1 },
  // { ImplicitValue,      3 },
  // { ImplicitPointer,    3 },
  // { Addrx,              2 },
  // { Constx,             2 },
  // { EntryValue,         3 },
  // { ConstTypeOp,        4 },
  // { RegvalType,         3 },
  // { DerefType,          3 },
  // { XderefType,         3 },
  // { Reinterpret,        2 },
};
}

namespace ImportedEntity {
enum {
  NameIdx      = 0,
  TagIdx       = 1,
  SourceIdx    = 3,
  EntityIdx    = 4,
  LineIdx      = 5,
  ColumnIdx    = 6,
  ParentIdx    = 7,
  OperandCount = 8
};
}

namespace ModuleINTEL {
enum {
  NameIdx         = 0,
  SourceIdx       = 1,
  LineIdx         = 2,
  ParentIdx       = 3,
  ConfigMacrosIdx = 4,
  IncludePathIdx  = 5,
  ApiNotesIdx     = 6,
  IsDeclIdx       = 7,
  OperandCount    = 8
};
}

} // namespace Operand
} // namespace SPIRVDebug

using namespace llvm;

inline spv::SourceLanguage convertDWARFSourceLangToSPIRV(dwarf::SourceLanguage DwarfLang) {
  switch (DwarfLang) {
  // When updating this function, make sure to also
  // update convertSPIRVSourceLangToDWARF()

  // LLVM does not yet define DW_LANG_C_plus_plus_17
  // case dwarf::SourceLanguage::DW_LANG_C_plus_plus_17:
  case dwarf::SourceLanguage::DW_LANG_C_plus_plus_14:
  case dwarf::SourceLanguage::DW_LANG_C_plus_plus:
    return spv::SourceLanguage::SourceLanguageCPP_for_OpenCL;
  case dwarf::SourceLanguage::DW_LANG_C99:
  case dwarf::SourceLanguage::DW_LANG_OpenCL:
    return spv::SourceLanguage::SourceLanguageOpenCL_C;
  default:
    return spv::SourceLanguage::SourceLanguageUnknown;
  }
}

inline dwarf::SourceLanguage convertSPIRVSourceLangToDWARF(unsigned SourceLang) {
  switch (SourceLang) {
  // When updating this function, make sure to also
  // update convertDWARFSourceLangToSPIRV()
  case spv::SourceLanguage::SourceLanguageOpenCL_CPP:
    return dwarf::SourceLanguage::DW_LANG_C_plus_plus_14;
  case spv::SourceLanguage::SourceLanguageCPP_for_OpenCL:
    // LLVM does not yet define DW_LANG_C_plus_plus_17
    // SourceLang = dwarf::SourceLanguage::DW_LANG_C_plus_plus_17;
    return dwarf::SourceLanguage::DW_LANG_C_plus_plus_14;
  case spv::SourceLanguage::SourceLanguageOpenCL_C:
  case spv::SourceLanguage::SourceLanguageESSL:
  case spv::SourceLanguage::SourceLanguageGLSL:
  case spv::SourceLanguage::SourceLanguageHLSL:
  case spv::SourceLanguage::SourceLanguageUnknown:
  default:
    return dwarf::DW_LANG_OpenCL;
  }
}

namespace SPIRV {
typedef SPIRVMap<dwarf::TypeKind, SPIRVDebug::EncodingTag> DbgEncodingMap;
template <>
inline void DbgEncodingMap::init() {
  add(static_cast<dwarf::TypeKind>(0), SPIRVDebug::Unspecified);
  add(dwarf::DW_ATE_address,           SPIRVDebug::Address);
  add(dwarf::DW_ATE_boolean,           SPIRVDebug::Boolean);
  add(dwarf::DW_ATE_float,             SPIRVDebug::Float);
  add(dwarf::DW_ATE_signed,            SPIRVDebug::Signed);
  add(dwarf::DW_ATE_signed_char,       SPIRVDebug::SignedChar);
  add(dwarf::DW_ATE_unsigned,          SPIRVDebug::Unsigned);
  add(dwarf::DW_ATE_unsigned_char,     SPIRVDebug::UnsignedChar);
}

typedef SPIRVMap<dwarf::Tag, SPIRVDebug::TypeQualifierTag> DbgTypeQulifierMap;
template <>
inline void DbgTypeQulifierMap::init() {
  add(dwarf::DW_TAG_const_type,    SPIRVDebug::ConstType);
  add(dwarf::DW_TAG_volatile_type, SPIRVDebug::VolatileType);
  add(dwarf::DW_TAG_restrict_type, SPIRVDebug::RestrictType);
  add(dwarf::DW_TAG_atomic_type,   SPIRVDebug::AtomicType);
}

typedef SPIRVMap<dwarf::Tag, SPIRVDebug::CompositeTypeTag> DbgCompositeTypeMap;
template <>
inline void DbgCompositeTypeMap::init() {
  add(dwarf::DW_TAG_class_type,     SPIRVDebug::Class);
  add(dwarf::DW_TAG_structure_type, SPIRVDebug::Structure);
  add(dwarf::DW_TAG_union_type,     SPIRVDebug::Union);
}

typedef SPIRVMap<dwarf::LocationAtom, SPIRVDebug::ExpressionOpCode>
  DbgExpressionOpCodeMap;
template <>
inline void DbgExpressionOpCodeMap::init() {
  add(dwarf::DW_OP_deref,               SPIRVDebug::Deref);
  add(dwarf::DW_OP_plus,                SPIRVDebug::Plus);
  add(dwarf::DW_OP_minus,               SPIRVDebug::Minus);
  add(dwarf::DW_OP_plus_uconst,         SPIRVDebug::PlusUconst);
  add(dwarf::DW_OP_bit_piece,           SPIRVDebug::BitPiece);
  add(dwarf::DW_OP_swap,                SPIRVDebug::Swap);
  add(dwarf::DW_OP_xderef,              SPIRVDebug::Xderef);
  add(dwarf::DW_OP_stack_value,         SPIRVDebug::StackValue);
  add(dwarf::DW_OP_constu,              SPIRVDebug::Constu);
  add(dwarf::DW_OP_LLVM_fragment,       SPIRVDebug::Fragment);
  add(dwarf::DW_OP_LLVM_convert,        SPIRVDebug::Convert);
  add(dwarf::DW_OP_consts,              SPIRVDebug::Consts);
  add(dwarf::DW_OP_dup,                 SPIRVDebug::Dup);
  add(dwarf::DW_OP_drop,                SPIRVDebug::Drop);
  add(dwarf::DW_OP_over,                SPIRVDebug::Over);
  add(dwarf::DW_OP_pick,                SPIRVDebug::Pick);
  add(dwarf::DW_OP_rot,                 SPIRVDebug::Rot);
  add(dwarf::DW_OP_abs,                 SPIRVDebug::Abs);
  add(dwarf::DW_OP_and,                 SPIRVDebug::And);
  add(dwarf::DW_OP_div,                 SPIRVDebug::Div);
  add(dwarf::DW_OP_mod,                 SPIRVDebug::Mod);
  add(dwarf::DW_OP_mul,                 SPIRVDebug::Mul);
  add(dwarf::DW_OP_neg,                 SPIRVDebug::Neg);
  add(dwarf::DW_OP_not,                 SPIRVDebug::Not);
  add(dwarf::DW_OP_or,                  SPIRVDebug::Or);
  add(dwarf::DW_OP_shl,                 SPIRVDebug::Shl);
  add(dwarf::DW_OP_shr,                 SPIRVDebug::Shr);
  add(dwarf::DW_OP_shra,                SPIRVDebug::Shra);
  add(dwarf::DW_OP_xor,                 SPIRVDebug::Xor);
  add(dwarf::DW_OP_bra,                 SPIRVDebug::Bra);
  add(dwarf::DW_OP_eq,                  SPIRVDebug::Eq);
  add(dwarf::DW_OP_ge,                  SPIRVDebug::Ge);
  add(dwarf::DW_OP_gt,                  SPIRVDebug::Gt);
  add(dwarf::DW_OP_le,                  SPIRVDebug::Le);
  add(dwarf::DW_OP_lt,                  SPIRVDebug::Lt);
  add(dwarf::DW_OP_ne,                  SPIRVDebug::Ne);
  add(dwarf::DW_OP_lit0,                SPIRVDebug::Lit0);
  add(dwarf::DW_OP_lit1,                SPIRVDebug::Lit1);
  add(dwarf::DW_OP_lit2,                SPIRVDebug::Lit2);
  add(dwarf::DW_OP_lit3,                SPIRVDebug::Lit3);
  add(dwarf::DW_OP_lit4,                SPIRVDebug::Lit4);
  add(dwarf::DW_OP_lit5,                SPIRVDebug::Lit5);
  add(dwarf::DW_OP_lit6,                SPIRVDebug::Lit6);
  add(dwarf::DW_OP_lit7,                SPIRVDebug::Lit7);
  add(dwarf::DW_OP_lit8,                SPIRVDebug::Lit8);
  add(dwarf::DW_OP_lit9,                SPIRVDebug::Lit9);
  add(dwarf::DW_OP_lit10,               SPIRVDebug::Lit10);
  add(dwarf::DW_OP_lit11,               SPIRVDebug::Lit11);
  add(dwarf::DW_OP_lit12,               SPIRVDebug::Lit12);
  add(dwarf::DW_OP_lit13,               SPIRVDebug::Lit13);
  add(dwarf::DW_OP_lit14,               SPIRVDebug::Lit14);
  add(dwarf::DW_OP_lit15,               SPIRVDebug::Lit15);
  add(dwarf::DW_OP_lit16,               SPIRVDebug::Lit16);
  add(dwarf::DW_OP_lit17,               SPIRVDebug::Lit17);
  add(dwarf::DW_OP_lit18,               SPIRVDebug::Lit18);
  add(dwarf::DW_OP_lit19,               SPIRVDebug::Lit19);
  add(dwarf::DW_OP_lit20,               SPIRVDebug::Lit20);
  add(dwarf::DW_OP_lit21,               SPIRVDebug::Lit21);
  add(dwarf::DW_OP_lit22,               SPIRVDebug::Lit22);
  add(dwarf::DW_OP_lit23,               SPIRVDebug::Lit23);
  add(dwarf::DW_OP_lit24,               SPIRVDebug::Lit24);
  add(dwarf::DW_OP_lit25,               SPIRVDebug::Lit25);
  add(dwarf::DW_OP_lit26,               SPIRVDebug::Lit26);
  add(dwarf::DW_OP_lit27,               SPIRVDebug::Lit27);
  add(dwarf::DW_OP_lit28,               SPIRVDebug::Lit28);
  add(dwarf::DW_OP_lit29,               SPIRVDebug::Lit29);
  add(dwarf::DW_OP_lit30,               SPIRVDebug::Lit30);
  add(dwarf::DW_OP_lit31,               SPIRVDebug::Lit31);
  add(dwarf::DW_OP_reg0,                SPIRVDebug::Reg0);
  add(dwarf::DW_OP_reg1,                SPIRVDebug::Reg1);
  add(dwarf::DW_OP_reg2,                SPIRVDebug::Reg2);
  add(dwarf::DW_OP_reg3,                SPIRVDebug::Reg3);
  add(dwarf::DW_OP_reg4,                SPIRVDebug::Reg4);
  add(dwarf::DW_OP_reg5,                SPIRVDebug::Reg5);
  add(dwarf::DW_OP_reg6,                SPIRVDebug::Reg6);
  add(dwarf::DW_OP_reg7,                SPIRVDebug::Reg7);
  add(dwarf::DW_OP_reg8,                SPIRVDebug::Reg8);
  add(dwarf::DW_OP_reg9,                SPIRVDebug::Reg9);
  add(dwarf::DW_OP_reg10,               SPIRVDebug::Reg10);
  add(dwarf::DW_OP_reg11,               SPIRVDebug::Reg11);
  add(dwarf::DW_OP_reg12,               SPIRVDebug::Reg12);
  add(dwarf::DW_OP_reg13,               SPIRVDebug::Reg13);
  add(dwarf::DW_OP_reg14,               SPIRVDebug::Reg14);
  add(dwarf::DW_OP_reg15,               SPIRVDebug::Reg15);
  add(dwarf::DW_OP_reg16,               SPIRVDebug::Reg16);
  add(dwarf::DW_OP_reg17,               SPIRVDebug::Reg17);
  add(dwarf::DW_OP_reg18,               SPIRVDebug::Reg18);
  add(dwarf::DW_OP_reg19,               SPIRVDebug::Reg19);
  add(dwarf::DW_OP_reg20,               SPIRVDebug::Reg20);
  add(dwarf::DW_OP_reg21,               SPIRVDebug::Reg21);
  add(dwarf::DW_OP_reg22,               SPIRVDebug::Reg22);
  add(dwarf::DW_OP_reg23,               SPIRVDebug::Reg23);
  add(dwarf::DW_OP_reg24,               SPIRVDebug::Reg24);
  add(dwarf::DW_OP_reg25,               SPIRVDebug::Reg25);
  add(dwarf::DW_OP_reg26,               SPIRVDebug::Reg26);
  add(dwarf::DW_OP_reg27,               SPIRVDebug::Reg27);
  add(dwarf::DW_OP_reg28,               SPIRVDebug::Reg28);
  add(dwarf::DW_OP_reg29,               SPIRVDebug::Reg29);
  add(dwarf::DW_OP_reg30,               SPIRVDebug::Reg30);
  add(dwarf::DW_OP_reg31,               SPIRVDebug::Reg31);
  add(dwarf::DW_OP_breg0,               SPIRVDebug::Breg0);
  add(dwarf::DW_OP_breg1,               SPIRVDebug::Breg1);
  add(dwarf::DW_OP_breg2,               SPIRVDebug::Breg2);
  add(dwarf::DW_OP_breg3,               SPIRVDebug::Breg3);
  add(dwarf::DW_OP_breg4,               SPIRVDebug::Breg4);
  add(dwarf::DW_OP_breg5,               SPIRVDebug::Breg5);
  add(dwarf::DW_OP_breg6,               SPIRVDebug::Breg6);
  add(dwarf::DW_OP_breg7,               SPIRVDebug::Breg7);
  add(dwarf::DW_OP_breg8,               SPIRVDebug::Breg8);
  add(dwarf::DW_OP_breg9,               SPIRVDebug::Breg9);
  add(dwarf::DW_OP_breg10,              SPIRVDebug::Breg10);
  add(dwarf::DW_OP_breg11,              SPIRVDebug::Breg11);
  add(dwarf::DW_OP_breg12,              SPIRVDebug::Breg12);
  add(dwarf::DW_OP_breg13,              SPIRVDebug::Breg13);
  add(dwarf::DW_OP_breg14,              SPIRVDebug::Breg14);
  add(dwarf::DW_OP_breg15,              SPIRVDebug::Breg15);
  add(dwarf::DW_OP_breg16,              SPIRVDebug::Breg16);
  add(dwarf::DW_OP_breg17,              SPIRVDebug::Breg17);
  add(dwarf::DW_OP_breg18,              SPIRVDebug::Breg18);
  add(dwarf::DW_OP_breg19,              SPIRVDebug::Breg19);
  add(dwarf::DW_OP_breg20,              SPIRVDebug::Breg20);
  add(dwarf::DW_OP_breg21,              SPIRVDebug::Breg21);
  add(dwarf::DW_OP_breg22,              SPIRVDebug::Breg22);
  add(dwarf::DW_OP_breg23,              SPIRVDebug::Breg23);
  add(dwarf::DW_OP_breg24,              SPIRVDebug::Breg24);
  add(dwarf::DW_OP_breg25,              SPIRVDebug::Breg25);
  add(dwarf::DW_OP_breg26,              SPIRVDebug::Breg26);
  add(dwarf::DW_OP_breg27,              SPIRVDebug::Breg27);
  add(dwarf::DW_OP_breg28,              SPIRVDebug::Breg28);
  add(dwarf::DW_OP_breg29,              SPIRVDebug::Breg29);
  add(dwarf::DW_OP_breg30,              SPIRVDebug::Breg30);
  add(dwarf::DW_OP_breg31,              SPIRVDebug::Breg31);
  add(dwarf::DW_OP_regx,                SPIRVDebug::Regx);
  add(dwarf::DW_OP_bregx,               SPIRVDebug::Bregx);
  add(dwarf::DW_OP_deref_size,          SPIRVDebug::DerefSize );
  add(dwarf::DW_OP_xderef_size,         SPIRVDebug::XderefSize );
  add(dwarf::DW_OP_nop,                 SPIRVDebug::Nop);
  add(dwarf::DW_OP_push_object_address, SPIRVDebug::PushObjectAddress );
}

typedef SPIRVMap<dwarf::Tag, SPIRVDebug::ImportedEntityTag>
  DbgImportedEntityMap;
template <>
inline void DbgImportedEntityMap::init() {
  add(dwarf::DW_TAG_imported_module,      SPIRVDebug::ImportedModule);
  add(dwarf::DW_TAG_imported_declaration, SPIRVDebug::ImportedDeclaration);
}

} // namespace SPIRV

#endif // SPIRV_DEBUG_H
