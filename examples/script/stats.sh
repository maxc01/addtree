#!/usr/bin/env bash

echo ==========
echo SMAC single
perl ./script/find_best.pl $(fd hists smac-single)

echo ==========
echo ADDTREE single
perl ./script/find_best.pl $(fd json addtree-single)

echo ==========
echo RANDOM single
perl ./script/find_best.pl $(fd txt random-single)

echo ==========
echo ADDTREE multiple
perl ./script/find_best.pl $(fd json addtree-multiple)

echo ==========
echo RANDOM multiple
perl ./script/find_best.pl $(fd txt random-multiple) $(fd json random-multiple)

echo ==========
echo TPE multiple
perl ./script/find_best.pl $(fd hists tpe-multiple)
