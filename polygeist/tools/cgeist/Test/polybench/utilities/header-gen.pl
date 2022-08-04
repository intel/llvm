#!/usr/bin/perl

# Generates headers using specification in polybench.spec
#
# Written by Tomofumi Yuki, 11/21 2014
#

use File::Path;

if ($#ARGV != 0) {
   printf("usage perl header-gen.pl output-dir\n");
   exit(1);
}

my $SPECFILE = 'polybench.spec';
my $OUTDIR = $ARGV[0];
my @DATASET_NAMES = ('MINI', 'SMALL', 'MEDIUM', 'LARGE', 'EXTRALARGE');

if (!(-e $OUTDIR)) {
   mkdir $OUTDIR;
}

my %INPUT;
my @keys;

open FILE, $SPECFILE or die;
  while (<FILE>) {
    my $line = $_;
    $line =~ s/\r|\n//g;
    #lines tarting with # is treated as comments
    next if ($line=~/^\s*#/);
    next if ($line=~/^\s*[\r|\n]+$/);
    my @line = split(/\t+/, $line);

    if (!keys %INPUT ) {
       foreach (@line) {
           $INPUT{$_} = [];
       }
       @keys = @line;
    } else {
       for (my $i = 0; $i <= $#line; $i++) {
           push @{$INPUT{$keys[$i]}}, $line[$i] ;
       }
    }
  }

close FILE;

for (my $r = 0; $r <= $#{$INPUT{'kernel'}}; $r++) {
   &generateHeader($r);
}

sub generateHeader() {

   my $row = $_[0];
   my $name = $INPUT{'kernel'}[$row];
   my $category = $INPUT{'category'}[$row];
   my $datatype = $INPUT{'datatype'}[$row];
   my $datatypeUC = uc $datatype;
   my @params = split /\s+/, $INPUT{'params'}[$row];


   my $headerDef = '_'. uc $name . '_H';
   $headerDef =~ s/-/_/g;

   my $paramDefs;
   foreach $set (@DATASET_NAMES) {
      my @sizes = split /\s+/, $INPUT{$set}[$row]; 
      $paramDefs .= '#  ifdef '.$set."_DATASET\n";
      for (my $i = 0; $i <= $#params; $i++) {
         $paramDefs .= '#   define '.$params[$i].' '.$sizes[$i]."\n";
      }
      $paramDefs .= '#  endif '."\n\n";
   }

   my $paramCheck = '# if';
   my $loopBoundDef = '';
   {
      my $first = 1;
      foreach (@params) {
         $paramCheck.= ' &&' if (!$first);
         $paramCheck .= " !defined($_)";
         $first = 0;
         $loopBoundDef .= '# define _PB_'.$_.' POLYBENCH_LOOP_BOUND('.$_.','.lc $_.')'."\n";
      }
   }

   my $kernelPath = "$OUTDIR/$category/$name";
   if (!(-e $kernelPath)) {
       mkpath $kernelPath;
   }

   open HFILE, ">$kernelPath/$name.h";
print HFILE << "EOF";
#ifndef $headerDef
# define $headerDef

/* Default to LARGE_DATASET. */
# if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(MEDIUM_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#  define LARGE_DATASET
# endif

$paramCheck
/* Define sample dataset sizes. */
$paramDefs
#endif /* !(@params) */

$loopBoundDef

/* Default data type */
# if !defined(DATA_TYPE_IS_INT) && !defined(DATA_TYPE_IS_FLOAT) && !defined(DATA_TYPE_IS_DOUBLE)
#  define DATA_TYPE_IS_$datatypeUC
# endif

#ifdef DATA_TYPE_IS_INT
#  define DATA_TYPE int
#  define DATA_PRINTF_MODIFIER "%d "
#endif 

#ifdef DATA_TYPE_IS_FLOAT
#  define DATA_TYPE float
#  define DATA_PRINTF_MODIFIER "%0.2f "
#  define SCALAR_VAL(x) x##f
#  define SQRT_FUN(x) sqrtf(x)
#  define EXP_FUN(x) expf(x)
#  define POW_FUN(x,y) powf(x,y)
# endif

#ifdef DATA_TYPE_IS_DOUBLE
#  define DATA_TYPE double
#  define DATA_PRINTF_MODIFIER "%0.2lf "
#  define SCALAR_VAL(x) x
#  define SQRT_FUN(x) sqrt(x)
#  define EXP_FUN(x) exp(x)
#  define POW_FUN(x,y) pow(x,y)
# endif

#endif /* !$headerDef */

EOF
     close HFILE;
}

