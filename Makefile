CC=g++ -pthread
CFLAGS= -Wall -O3 -fPIC -std=c++11 
CXXFLAGS= -Wall -O3 -fPIC -std=c++11
OBJS= test.o pim_blas.o pim_runtime.o pim_config.o pim_func_sim/pim_func_sim.o pim_func_sim/pim_unit.o pim_func_sim/pim_utils.o pim_func_sim/pim_func_config.h fpga_pim.o
SH_OBJS= pim_blas.o pim_runtime.o pim_config.o pim_func_sim/pim_func_sim.o pim_func_sim/pim_unit.o pim_func_sim/pim_utils.o pim_func_sim/pim_func_config.h fpga_pim.o
TARGET=test

all: $(TARGET)

clean:
	rm -f *.o
	rm -f pim_func_sim/*.o
	rm -f libpimss.so
	rm -f $(TARGET)

fpga:
	gcc -c ../fpga_pim.c
	$(CC) -o $@ $(OBJS)

so:	$(OBJS)
	$(CC) -shared -o libpimss.so $(SH_OBJS)

$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS)


