#
# modify-python-lldb.py
#
# This script modifies the lldb module (which was automatically generated via
# running swig) to support iteration and/or equality operations for certain lldb
# objects, implements truth value testing for certain lldb objects, and adds a
# global variable 'debugger_unique_id' which is initialized to 0.
#
# As a cleanup step, it also removes the 'residues' from the autodoc features of
# swig.  For an example, take a look at SBTarget.h header file, where we take
# advantage of the already existing doxygen C++-docblock and make it the Python
# docstring for the same method.  The 'residues' in this context include the
# '#endif', the '#ifdef SWIG', the c comment marker, the trailing blank (SPC's)
# line, and the doxygen comment start marker.
#
# In addition to the 'residues' removal during the cleanup step, it also
# transforms the 'char' data type (which was actually 'char *' but the 'autodoc'
# feature of swig removes ' *' from it) into 'str' (as a Python str type).
#
# It also calls SBDebugger.Initialize() to initialize the lldb debugger
# subsystem.
#

# System modules
import sys
import re
if sys.version_info.major >= 3:
    import io as StringIO
else:
    import StringIO

# import use_lldb_suite so we can find third-party and helper modules
import use_lldb_suite

# Third party modules
import six

# LLDB modules

if len(sys.argv) != 2:
    output_name = "./lldb.py"
else:
    output_name = sys.argv[1] + "/lldb.py"

# print "output_name is '" + output_name + "'"

#
# Residues to be removed.
#
c_endif_swig = "#endif"
c_ifdef_swig = "#ifdef SWIG"
c_comment_marker = "//------------"
# The pattern for recognizing the doxygen comment block line.
doxygen_comment_start = re.compile("^\s*(/// ?)")
# The demarcation point for turning on/off residue removal state.
# When bracketed by the lines, the CLEANUP_DOCSTRING state (see below) is ON.
toggle_docstring_cleanup_line = '        """'


def char_to_str_xform(line):
    """This transforms the 'char', i.e, 'char *' to 'str', Python string."""
    line = line.replace(' char', ' str')
    line = line.replace('char ', 'str ')
    # Special case handling of 'char **argv' and 'char **envp'.
    line = line.replace('str argv', 'list argv')
    line = line.replace('str envp', 'list envp')
    return line

#
# The one-liner docstring also needs char_to_str transformation, btw.
#
TWO_SPACES = ' ' * 2
EIGHT_SPACES = ' ' * 8
one_liner_docstring_pattern = re.compile(
    '^(%s|%s)""".*"""$' %
    (TWO_SPACES, EIGHT_SPACES))

# This supports the iteration protocol.
iter_def = "    def __iter__(self): return lldb_iter(self, '%s', '%s')"
module_iter = "    def module_iter(self): return lldb_iter(self, '%s', '%s')"
breakpoint_iter = "    def breakpoint_iter(self): return lldb_iter(self, '%s', '%s')"
watchpoint_iter = "    def watchpoint_iter(self): return lldb_iter(self, '%s', '%s')"
section_iter = "    def section_iter(self): return lldb_iter(self, '%s', '%s')"
compile_unit_iter = "    def compile_unit_iter(self): return lldb_iter(self, '%s', '%s')"

# Called to implement the built-in function len().
# Eligible objects are those containers with unambiguous iteration support.
len_def = "    def __len__(self): return self.%s()"

# This supports the rich comparison methods of __eq__ and __ne__.
eq_def = "    def __eq__(self, other): return isinstance(other, %s) and %s"
ne_def = "    def __ne__(self, other): return not self.__eq__(other)"

# Called to implement truth value testing and the built-in operation bool();
# Note that Python 2 uses __nonzero__(), whereas Python 3 uses __bool__()
# should return False or True, or their integer equivalents 0 or 1.
# Delegate to self.IsValid() if it is defined for the current lldb object.

if six.PY2:
    nonzero_def = "    def __nonzero__(self): return self.IsValid()"
else:
    nonzero_def = "    def __bool__(self): return self.IsValid()"

# A convenience iterator for SBSymbol!
symbol_in_section_iter_def = '''
    def symbol_in_section_iter(self, section):
        """Given a module and its contained section, returns an iterator on the
        symbols within the section."""
        for sym in self:
            if in_range(sym, section):
                yield sym
'''

#
# This dictionary defines a mapping from classname to (getsize, getelem) tuple.
#
d = {'SBBreakpoint': ('GetNumLocations', 'GetLocationAtIndex'),
     'SBCompileUnit': ('GetNumLineEntries', 'GetLineEntryAtIndex'),
     'SBDebugger': ('GetNumTargets', 'GetTargetAtIndex'),
     'SBModule': ('GetNumSymbols', 'GetSymbolAtIndex'),
     'SBProcess': ('GetNumThreads', 'GetThreadAtIndex'),
     'SBSection': ('GetNumSubSections', 'GetSubSectionAtIndex'),
     'SBThread': ('GetNumFrames', 'GetFrameAtIndex'),

     'SBInstructionList': ('GetSize', 'GetInstructionAtIndex'),
     'SBStringList': ('GetSize', 'GetStringAtIndex',),
     'SBSymbolContextList': ('GetSize', 'GetContextAtIndex'),
     'SBTypeList': ('GetSize', 'GetTypeAtIndex'),
     'SBValueList': ('GetSize', 'GetValueAtIndex'),

     'SBType': ('GetNumberChildren', 'GetChildAtIndex'),
     'SBValue': ('GetNumChildren', 'GetChildAtIndex'),

     # SBTarget needs special processing, see below.
     'SBTarget': {'module': ('GetNumModules', 'GetModuleAtIndex'),
                  'breakpoint': ('GetNumBreakpoints', 'GetBreakpointAtIndex'),
                  'watchpoint': ('GetNumWatchpoints', 'GetWatchpointAtIndex')
                  },

     # SBModule has an additional section_iter(), see below.
     'SBModule-section': ('GetNumSections', 'GetSectionAtIndex'),
     # And compile_unit_iter().
     'SBModule-compile-unit': ('GetNumCompileUnits', 'GetCompileUnitAtIndex'),
     # As well as symbol_in_section_iter().
     'SBModule-symbol-in-section': symbol_in_section_iter_def
     }

#
# This dictionary defines a mapping from classname to equality method name(s).
#
e = {'SBAddress': ['GetFileAddress', 'GetModule'],
     'SBBreakpoint': ['GetID'],
     'SBWatchpoint': ['GetID'],
     'SBFileSpec': ['GetFilename', 'GetDirectory'],
     'SBModule': ['GetFileSpec', 'GetUUIDString'],
     'SBType': ['GetByteSize', 'GetName']
     }


def list_to_frag(list):
    """Transform a list to equality program fragment.

    For example, ['GetID'] is transformed to 'self.GetID() == other.GetID()',
    and ['GetFilename', 'GetDirectory'] to 'self.GetFilename() == other.GetFilename()
    and self.GetDirectory() == other.GetDirectory()'.
    """
    if not list:
        raise Exception("list should be non-empty")
    frag = StringIO.StringIO()
    for i in range(len(list)):
        if i > 0:
            frag.write(" and ")
        frag.write("self.{0}() == other.{0}()".format(list[i]))
    return frag.getvalue()


class NewContent(StringIO.StringIO):
    """Simple facade to keep track of the previous line to be committed."""

    def __init__(self):
        StringIO.StringIO.__init__(self)
        self.prev_line = None

    def add_line(self, a_line):
        """Add a line to the content, if there is a previous line, commit it."""
        if self.prev_line is not None:
            self.write(self.prev_line + "\n")
        self.prev_line = a_line

    def del_line(self):
        """Forget about the previous line, do not commit it."""
        self.prev_line = None

    def del_blank_line(self):
        """Forget about the previous line if it is a blank line."""
        if self.prev_line is not None and not self.prev_line.strip():
            self.prev_line = None

    def finish(self):
        """Call this when you're finished with populating content."""
        if self.prev_line is not None:
            self.write(self.prev_line + "\n")
        self.prev_line = None

# The new content will have the iteration protocol defined for our lldb
# objects.
new_content = NewContent()

with open(output_name, 'r') as f_in:
    content = f_in.read()

# The pattern for recognizing the SWIG Version string
version_pattern = re.compile("^# Version:? (.*)$")

# The pattern for recognizing the beginning of an SB class definition.
class_pattern = re.compile("^class (SB.*)\(_object\):$")

# The pattern for recognizing the beginning of the __init__ method definition.
init_pattern = re.compile("^    def __init__\(self.*\):")

# The pattern for recognizing the beginning of the IsValid method definition.
isvalid_pattern = re.compile("^    def IsValid\(")

# These define the states of our finite state machine.
NORMAL = 1
DEFINING_ITERATOR = 2
DEFINING_EQUALITY = 4
CLEANUP_DOCSTRING = 8

# Our FSM begins its life in the NORMAL state, and transitions to the
# DEFINING_ITERATOR and/or DEFINING_EQUALITY state whenever it encounters the
# beginning of certain class definitions, see dictionaries 'd' and 'e' above.
#
# Note that the two states DEFINING_ITERATOR and DEFINING_EQUALITY are
# orthogonal in that our FSM can be in one, the other, or both states at the
# same time.  During such time, the FSM is eagerly searching for the __init__
# method definition in order to insert the appropriate method(s) into the lldb
# module.
#
# The state CLEANUP_DOCSTRING can be entered from either the NORMAL or the
# DEFINING_ITERATOR/EQUALITY states.  While in this state, the FSM is fixing/
# cleaning the Python docstrings generated by the swig docstring features.
#
# The FSM, in all possible states, also checks the current input for IsValid()
# definition, and inserts a __nonzero__() method definition to implement truth
# value testing and the built-in operation bool().
state = NORMAL

for line in content.splitlines():
    # Handle the state transition into CLEANUP_DOCSTRING state as it is possible
    # to enter this state from either NORMAL or DEFINING_ITERATOR/EQUALITY.
    #
    # If '        """' is the sole line, prepare to transition to the
    # CLEANUP_DOCSTRING state or out of it.

    if line == toggle_docstring_cleanup_line:
        if state & CLEANUP_DOCSTRING:
            # Special handling of the trailing blank line right before the '"""'
            # end docstring marker.
            new_content.del_blank_line()
            state ^= CLEANUP_DOCSTRING
        else:
            state |= CLEANUP_DOCSTRING

    if state == NORMAL:
        match = class_pattern.search(line)
        # If we are at the beginning of the class definitions, prepare to
        # transition to the DEFINING_ITERATOR/DEFINING_EQUALITY state for the
        # right class names.
        if match:
            cls = match.group(1)
            if cls in d:
                # Adding support for iteration for the matched SB class.
                state |= DEFINING_ITERATOR
            if cls in e:
                # Adding support for eq and ne for the matched SB class.
                state |= DEFINING_EQUALITY

    if (state & DEFINING_ITERATOR) or (state & DEFINING_EQUALITY):
        match = init_pattern.search(line)
        if match:
            # We found the beginning of the __init__ method definition.
            # This is a good spot to insert the iter and/or eq-ne support.
            #
            # But note that SBTarget has three types of iterations.
            if cls == "SBTarget":
                new_content.add_line(module_iter % (d[cls]['module']))
                new_content.add_line(breakpoint_iter % (d[cls]['breakpoint']))
                new_content.add_line(watchpoint_iter % (d[cls]['watchpoint']))
            else:
                if (state & DEFINING_ITERATOR):
                    new_content.add_line(iter_def % d[cls])
                    new_content.add_line(len_def % d[cls][0])
                if (state & DEFINING_EQUALITY):
                    new_content.add_line(eq_def % (cls, list_to_frag(e[cls])))
                    new_content.add_line(ne_def)

            # SBModule has extra SBSection, SBCompileUnit iterators and
            # symbol_in_section_iter()!
            if cls == "SBModule":
                new_content.add_line(section_iter % d[cls + '-section'])
                new_content.add_line(compile_unit_iter %
                                     d[cls + '-compile-unit'])
                new_content.add_line(d[cls + '-symbol-in-section'])

            # Next state will be NORMAL.
            state = NORMAL

    if (state & CLEANUP_DOCSTRING):
        # Cleanse the lldb.py of the autodoc'ed residues.
        if c_ifdef_swig in line or c_endif_swig in line:
            continue
        # As well as the comment marker line.
        if c_comment_marker in line:
            continue

        # Also remove the '\a ' and '\b 'substrings.
        line = line.replace('\a ', '')
        line = line.replace('\b ', '')
        # And the leading '///' substring.
        doxygen_comment_match = doxygen_comment_start.match(line)
        if doxygen_comment_match:
            line = line.replace(doxygen_comment_match.group(1), '', 1)

        line = char_to_str_xform(line)

        # Note that the transition out of CLEANUP_DOCSTRING is handled at the
        # beginning of this function already.

    # This deals with one-liner docstring, for example, SBThread.GetName:
    # """GetName(self) -> char""".
    if one_liner_docstring_pattern.match(line):
        line = char_to_str_xform(line)

    # Look for 'def IsValid(*args):', and once located, add implementation
    # of truth value testing for this object by delegation.
    if isvalid_pattern.search(line):
        new_content.add_line(nonzero_def)

    # Pass the original line of content to new_content.
    new_content.add_line(line)

# We are finished with recording new content.
new_content.finish()

with open(output_name, 'w') as f_out:
    f_out.write(new_content.getvalue())
    f_out.write('''debugger_unique_id = 0
SBDebugger.Initialize()
debugger = None
target = SBTarget()
process = SBProcess()
thread = SBThread()
frame = SBFrame()''')
