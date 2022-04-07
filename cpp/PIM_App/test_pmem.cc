
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

int main(void) {
    int fd_;
    uint8_t *pmemAddr_;
    uint8_t tmp[SIZE];
    int is_pmem;

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
    if (pmemAddr_ == (uint8_t*) MAP_FAILED) {
        perror("mmap");
        exit(1);
    }
    printf("finish mmap at %llx\n", pmemAddr_);

    /* determine if range is true pmem */
    is_pmem = pmem_is_pmem(pmemAddr_, LEN_PIM);
    printf("pmem_is_pmem: %d\n", is_pmem);

    if (is_pmem) {
        //pmem_persist(pmemAddr_, LEN_PIM);
        pmem_memcpy_persist(pmemAddr_, tmp, SIZE);
        pmem_memcpy_persist(pmemAddr_, tmp, SIZE);

        pmem_memcpy_persist(tmp, pmemAddr_, SIZE);
        pmem_memcpy_persist(tmp, pmemAddr_, SIZE);

        pmem_memcpy_persist(pmemAddr_, tmp, SIZE);
        pmem_memcpy_persist(pmemAddr_, tmp, SIZE);

        pmem_memcpy_persist(tmp, pmemAddr_, SIZE);
        pmem_memcpy_persist(tmp, pmemAddr_, SIZE);
    }
    else {
        memcpy(pmemAddr_, tmp, SIZE);
        pmem_msync(pmemAddr_, SIZE);
        memcpy(pmemAddr_, tmp, SIZE);
        pmem_msync(pmemAddr_, SIZE);

        memcpy(tmp, pmemAddr_, SIZE);
        pmem_msync(pmemAddr_, SIZE);
        memcpy(tmp, pmemAddr_, SIZE);
        pmem_msync(pmemAddr_, SIZE);

        memcpy(pmemAddr_, tmp, SIZE);
        pmem_msync(pmemAddr_, SIZE);
        memcpy(pmemAddr_, tmp, SIZE);
        pmem_msync(pmemAddr_, SIZE);

        memcpy(tmp, pmemAddr_, SIZE);
        pmem_msync(pmemAddr_, SIZE);
        memcpy(tmp, pmemAddr_, SIZE);
        pmem_msync(pmemAddr_, SIZE);
    }

    /*
    std::memcpy(pmemAddr_, tmp, SIZE);
    clflush(pmemAddr_, SIZE);

    std::memcpy(pmemAddr_, tmp, SIZE);
    clflush(pmemAddr_, SIZE);

    std::memcpy(tmp, pmemAddr_, SIZE);
    clflush(pmemAddr_, SIZE);

    std::memcpy(tmp, pmemAddr_, SIZE);
    clflush(pmemAddr_, SIZE);
    
    std::memcpy(pmemAddr_, tmp, SIZE);
    clflush(pmemAddr_, SIZE);
    
    std::memcpy(pmemAddr_, tmp, SIZE);
    clflush(pmemAddr_, SIZE);
    
    std::memcpy(tmp, pmemAddr_, SIZE);
    clflush(pmemAddr_, SIZE);
    
    std::memcpy(tmp, pmemAddr_, SIZE);
    clflush(pmemAddr_, SIZE);
    */

    close(fd_);

    return 0;
}
