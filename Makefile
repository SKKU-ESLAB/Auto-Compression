CC=g++ -pthread
CFLAGS= -Wall -O3 -fPIC -std=c++11
OBJS= test.o pim_blas.o pim_runtime.o pim_config.o pim_func_sim/pim_func_sim.o pim_func_sim/pim_unit.o pim_func_sim/pim_utils.o pim_func_sim/pim_func_config.h
TARGET=test

all: $(TARGET)

clean:
	rm -f *.o
	rm -f $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS)
