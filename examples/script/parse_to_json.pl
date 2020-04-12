#!/usr/bin/env perl

use Mojo::JSON qw(decode_json encode_json to_json from_json j);
use Data::Dumper;
use feature 'say';

my @all_res;
my $per_res = {};
while(<>) {
  # say $.;
  chomp;
  if ($. % 3 == 1) {
    $per_res->{iteration} = $_;
  }
  if ($. % 3 == 2) {
    $per_res->{param} = from_json $_;
  }
  if ($. % 3 == 0) {
    $per_res->{obj_info} = from_json $_;
    push @all_res, $per_res;
    # say Dumper $per_res;
    $per_res = {};
  }

}

say encode_json \@all_res;


