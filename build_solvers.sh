#!/usr/bin/env bash
set -euo pipefail

cd solvers/glucose/simp
make clean
make rs
cd ../../..

cd solvers/glucose_weighted/simp
make clean
make r
cd ../../..
