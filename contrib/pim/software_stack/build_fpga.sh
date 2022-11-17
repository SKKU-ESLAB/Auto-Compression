#!/bin/sh

echo "Assume that ../fpga_pim.c exists (if not, error causes)"

make clean && make fpga && make
