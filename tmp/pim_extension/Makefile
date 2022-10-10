CC=g++ -pthread
CFLAGS= -Wall -O3 -fPIC -std=c++11
OBJS= pim_blas.o pim_runtime.o pim_config.o
TARGET=yey

all: $(TARGET)

clean:
	rm -f *.o
	rm -f $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS)
