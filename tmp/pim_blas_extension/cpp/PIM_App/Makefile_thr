CC=g++ -pthread
CFLAGS= -Wall -O3 -fPIC -std=c++11
OBJS=pim_app.o transaction_generator.o
TARGET=pim-app

all: $(TARGET)

clean:
	rm -f *.o
	rm -f $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS)
