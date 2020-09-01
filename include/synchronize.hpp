#pragma once

#include "residency.hpp"
#include "error.hpp"
#include <cstdlib>
#include <cstring>

namespace mufflon { // There is no memory namespace on purpose

// Functions for unloading a handle from the device
template < class T >
void unload(ArrayDevHandle_t<Device::CPU, T>& hdl) {
	delete[] hdl.handle;
	hdl.handle = nullptr;
}
template < class T >
void unload(ArrayDevHandle_t<Device::CUDA, T>& hdl) {
	if(hdl.handle != nullptr) {
		check_error(cudaFree(hdl.handle));
		hdl.handle = nullptr;
	}
}

// A number of copy primitives which call the internal required methods.
// This relies on CUDA UVA
template < typename T >
inline void copy(T* dst, const T* src, std::size_t size) {
	static_assert(std::is_trivially_copyable<T>::value,
				  "Must be trivially copyable");
	check_error(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
}

template < Device dev >
inline void mem_set(void* mem, int value, std::size_t size) {
	std::memset(mem, value, size);
}

template <>
inline void mem_set<Device::CUDA>(void* mem, int value, std::size_t size) {
	check_error(::cudaMemset(mem, value, size));
}


} // namespace mufflon