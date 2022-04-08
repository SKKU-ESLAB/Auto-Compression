
#include <sys/mman.h>
#include <time.h>
#include <stdlib.h>
#include <string>
#include <cstdint>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <iostream>
#include <unistd.h>
#include "./pim_config.h"
#include "./half.hpp"
#include "./PIM-DD.h"
#include <x86intrin.h>
#include <libpmem.h>

#define SIZE 1024

void clflush(uint8_t *mem, size_t size) {
    for (size_t i=0; i<size; i+=64)
        _mm_clflushopt(&mem[i]);
}

int main(void) {
    int fd_;
    uint8_t *pmemAddr_;
    uint8_t tmp[SIZE];

    fd_ = open("/dev/PIM", O_RDWR|O_SYNC);
    if (fd_ < 0)
    {
        printf("open fail %s\n", strerror(errno));
        exit(1);
    }
    printf("fd %d\n", fd_);

    pmemAddr_ = (uint8_t *) mmap(NULL, LEN_PIM,
                                 PROT_READ | PROT_WRITE,
                                 MAP_SHARED,
                                 fd_, 0);
    if (pmemAddr_ == (uint8_t*) MAP_FAILED)
        perror("mmap");

    printf("finish mmap at %llx\n", pmemAddr_);

    std::memcpy(pmemAddr_, tmp, SIZE);
    clflush(pmemAddr_, SIZE);

    std::memcpy(pmemAddr_, tmp, SIZE);
    clflush(pmemAddr_, SIZE);

    std::memcpy(pmemAddr_, tmp, SIZE);
    clflush(pmemAddr_, SIZE);

    std::memcpy(tmp, pmemAddr_, SIZE);
    clflush(pmemAddr_, SIZE);

    std::memcpy(tmp, pmemAddr_, SIZE);
    clflush(pmemAddr_, SIZE);
    
    std::memcpy(tmp, pmemAddr_, SIZE);
    clflush(pmemAddr_, SIZE);
    
    std::memcpy(pmemAddr_, tmp, SIZE);
    clflush(pmemAddr_, SIZE);
    
    std::memcpy(pmemAddr_, tmp, SIZE);
    clflush(pmemAddr_, SIZE);
    
    std::memcpy(pmemAddr_, tmp, SIZE);
    clflush(pmemAddr_, SIZE);
    
    std::memcpy(tmp, pmemAddr_, SIZE);
    clflush(pmemAddr_, SIZE);
    
    std::memcpy(tmp, pmemAddr_, SIZE);
    clflush(pmemAddr_, SIZE);
    
    std::memcpy(tmp, pmemAddr_, SIZE);
    clflush(pmemAddr_, SIZE);


    return 0;
}
