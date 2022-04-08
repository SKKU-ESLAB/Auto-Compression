#!/bin/bash
g++ -o test test.cc -Wall -O3 -fPIC -std=c++11 -mclflushopt
g++ -o test_pmem test_pmem.cc -Wall -O3 -fPIC -std=c++11 -lpmem
