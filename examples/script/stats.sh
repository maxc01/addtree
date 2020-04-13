#!/usr/bin/env bash

echo ==========
echo SMAC single
perl ./script/parse_line.pl $(fd hists smac-single)

echo ==========
echo ADDTREE single
perl ./script/parse_line.pl $(fd json addtree-single)

echo ==========
echo ADDTREE multiple
perl ./script/parse_line.pl $(fd json addtree-multiple)

echo ==========
echo RANDOM single
perl ./script/parse_line.pl $(fd txt random-single)

echo ==========
echo RANDOM multiple
perl ./script/parse_line.pl $(fd txt random-multiple)

echo ==========
echo TPE multiple
perl ./script/parse_line.pl $(fd hists tpe-multiple)
