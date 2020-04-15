#!/usr/bin/env perl

use JSON::XS;
use List::Util qw(max);
use feature say;

my $one_row;
my @vs;
while (<>) {
  $one_row = {};
  $data = decode_json $_;
  $one_row->{method1} = $data->{params}->{b1}->{prune_method};
  $one_row->{method2} = $data->{params}->{b2}->{prune_method};
  $one_row->{method3} = $data->{params}->{b3}->{prune_method};

  my @params = ($data->{params}->{b1}->{amount},
                $data->{params}->{b2}->{amount},
                $data->{params}->{b3}->{amount});
  $one_row->{param} = \@params;
  $one_row->{value} = -$data->{obj_info}->{value};
  $one_row->{top1} = $data->{obj_info}->{top1};
  $one_row->{sparsity} = $data->{obj_info}->{sparsity};
  if (exists $data{iteration}) {
    $one_row->{iteration} = $data->{iteration};
  } else {
    $one_row->{iteration} = $.;
  }

  push @vs, $one_row;

}

say encode_json \@vs;

