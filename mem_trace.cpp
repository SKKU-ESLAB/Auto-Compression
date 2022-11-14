#include <iostream>
#include <fstream>
#include <cstring>
#include <sstream>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <chrono>

#define LEN_PIM 0x100000000

int fd;
std::ifstream fm;
std::string line;
uint8_t* pim_mem;
uint8_t* buffer;
uint64_t pim_base;

uint32_t burstSize = 32;

void set_trace_file(char **argv, char option) {
	std::cout << " > set trace file\n";
	fm.open("./mem_trace/"+std::string(argv[1])+option+".txt");
}

void set_pim_device() {
	std::cout << " > set pim device\n";
	fd = open("/dev/PIM", O_RDWR|O_SYNC);
	
	if (fd < 0)
		std::cout << "Open /dev/PIM failed...\n";
	else
		std::cout << "Opened /dev/PIM !\n";

	pim_mem = (uint8_t*)mmap(NULL, LEN_PIM, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	pim_base = (uint64_t)pim_mem;
}

void trace_and_send() {
	std::cout << " > trace and send\n";

	typedef std::chrono::high_resolution_clock Time;
	typedef std::chrono::milliseconds ms;
	typedef std::chrono::duration<float> fsec;

	buffer = (uint8_t*)calloc(64, sizeof(uint8_t));
	
	auto start = Time::now();
	while(std::getline(fm, line)) {
		std::stringstream linestream(line);
		int is_write;
		uint64_t hex_addr;

		linestream >> is_write >> hex_addr;

		if (is_write == 0) {  // read
			// std::memcpy(buffer, pim_mem + hex_addr, burstSize);
			for (int i=0; i<burstSize; i++)
				buffer[i] = (pim_mem + hex_addr)[i];
		} else if (is_write == 1) {  // write
			// std::memcpy(pim_mem + hex_addr, buffer, burstSize);
			for (int i=0; i<burstSize; i++)
				(pim_mem + hex_addr)[i] = buffer[i];
		} else if (is_write == 2) {  // preprocess end
			start = Time::now();
			system("sudo m5 dumpstats");
		} else {
			std::cout << "This should not be done... Somethings wrong\n";
		}
	}
	system("sudo m5 dumpstats");
	
	auto end = Time::now();
	std::cout << "All trace ended\n";
	fsec time = end - start;
	std::cout << time.count() << "s\n";
}

int main(int argc, char **argv) {
	char option;
	std::cout << "option : 1 / 2 / 3\nenter option :";
	std::cin >> option;

	if (argc <= 1) {
		std::cout << "add, mul, mac, bn, gemv, lstm\n";
		return -1;
	}

	set_trace_file(argv, option);

	system("sudo m5 checkpoint");
	system("echo CPU Switched!");

	set_pim_device();

	trace_and_send();

	return 0;
}
