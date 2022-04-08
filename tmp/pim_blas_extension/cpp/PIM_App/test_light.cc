
#include <sys/mman.h>
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
#include <libpmem.h>

#define SIZE 64

int main(void) {
    int fd_;
    uint8_t *pmemAddr_;
    uint8_t tmp[SIZE];

    fd_ = open("/dev/PIM", O_RDWR|O_SYNC);
    
    printf("fd %d\n", fd_);

    pmemAddr_ = (uint8_t *) mmap(NULL, LEN_PIM,
                                 PROT_READ | PROT_WRITE,
                                 MAP_SHARED,
                                 fd_, 0);
    
    printf("finish mmap at %llx\n", pmemAddr_);

    std::memcpy(pmemAddr_, tmp, SIZE);

    return 0;
}
