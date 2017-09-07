#!/usr/bin/perl -s
#*************************************************************************
#
#   Program:    
#   File:       corephilic.pl
#   
#   Version:    V1.1
#   Date:       07.03.12
#   Function:   Core Philic plugin for the SAAP server
#   
#   Copyright:  (c) UCL / Dr. Andrew C. R. Martin 2011-2012
#   Author:     Dr. Andrew C. R. Martin
#   Address:    Biomolecular Structure & Modelling Unit,
#               Department of Biochemistry & Molecular Biology,
#               University College,
#               Gower Street,
#               London.
#               WC1E 6BT.
#   EMail:      andrew@bioinf.org.uk
#   Web:        http://www.bioinf.org.uk/
#               
#*************************************************************************
#
#   This program is not in the public domain, but it may be copied
#   according to the conditions laid out in the accompanying file
#   COPYING.DOC
#
#   The code may be modified as required, but any modifications must be
#   documented so that the person responsible can be identified. If 
#   someone else breaks this code, I don't want to be blamed for code 
#   that does not work! 
#
#   The code may not be sold commercially or included as part of a 
#   commercial product except as described in the file COPYING.DOC.
#
#*************************************************************************
#
#   Description:
#   ============
#
#*************************************************************************
#
#   Usage:
#   ======
#
#*************************************************************************
#
#   Revision History:
#   =================
#   V1.0  16.12.11 Original
#   V1.1  07.03.12 Generates relative accessibility in the same way that
#                  SurfacePhobic does
#
#*************************************************************************
use strict;
use FindBin;
use Cwd qw(abs_path);
use lib abs_path("$FindBin::Bin/../lib");
use config;
use SAAP;
use XMAS;

# Information string about this plugin
$::infoString = "Checking whether a hydrophilic residue has been introduced in the core";

my $relaccess    = (-1);
my $relaccessMol = (-1);

my $result = "OK";
my($residue, $mutant, $pdbfile) = SAAP::ParseCmdLine("CorePhilic");

# See if the results are cached
my $json = SAAP::CheckCache("CorePhilic", $pdbfile, $residue, $mutant);
if($json ne "")
{
    print "$json\n";
    exit 0;
}

my($chain, $resnum, $insert) = SAAP::ParseResSpec($residue);
my $native = SAAP::GetNative($pdbfile, $residue);

my $status;
($relaccess, $relaccessMol, $status) = SAAP::GetRelativeAccess($pdbfile, $residue);
if($status != 0)
{
    if($status < 0)
    {
        SAAP::PrintJsonError("CorePhilic", "Residue not found");
        exit 1;
    }
    my $message = $XMAS::ErrorMessage[$status];
    SAAP::PrintJsonError("CorePhilic", $message);
    exit 1;
}

if($relaccessMol < $SAAP::buried)
{
   # If it's a change from hydrophobic to hydrophilic...
    if(($SAAP::hydrophobicity{$native} > (-0.1)) && 
       ($SAAP::hydrophobicity{$mutant} < (-0.1)))
    {
        $result = "BAD";
    }
}

$json = SAAP::MakeJson("CorePhilic", ('BOOL'=>$result, 'RELACCESS'=>$relaccessMol, 'NATIVE-HPHOB'=>$SAAP::hydrophobicity{$native}, 'MUTANT-HPHOB'=>$SAAP::hydrophobicity{$mutant}));
print "$json\n";
SAAP::WriteCache("CorePhilic", $pdbfile, $residue, $mutant, $json);



sub UsageDie
{
    print STDERR <<__EOF;

corephilic.pl V1.1 (c) 2011-2012, UCL, Dr. Andrew C.R. Martin
Usage: corephilic.pl [chain]resnum[insert] newaa pdbfile
       (newaa maybe 3-letter or 1-letter code)

Does core hydrophobilic calculations for the SAAP server.
       
__EOF
   exit 0;
}
