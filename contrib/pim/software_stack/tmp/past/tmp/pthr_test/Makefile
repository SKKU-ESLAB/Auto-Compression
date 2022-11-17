CC=g++ -pthread
CFLAGS= -Wall -O3 -fPIC -std=c++11
OBJS= test1.o
TARGET=test

all: $(TARGET)

clean:
	rm -f *.o
	rm -f $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS)
