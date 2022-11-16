#include <torch/extension.h>
#include <iostream>
#include <cstdint>
#include <sys/mman.h>
#include <x86intrin.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void flush(torch::Tensor T, size_t size) {
	uint8_t *data = (uint8_t*)T.data_ptr();
	for (int i=0; i<size; i+=32)
		_mm_clflush(&data[i]);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flush", &flush, "flush data of ptr address");
}
