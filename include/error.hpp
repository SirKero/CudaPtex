#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace mufflon {

class CudaException : public std::exception {
public:
	CudaException(cudaError_t error) :
		m_errCode(error) {
		fprintf(stderr, "[Unknown function] CUDA exception: %s: %s", cudaGetErrorName(m_errCode), cudaGetErrorString(m_errCode));
	}

	virtual const char* what() const noexcept override {
		return (std::string(cudaGetErrorName(m_errCode)) + std::string(": ") + std::string(cudaGetErrorString(m_errCode))).c_str();
	}

private:
	cudaError_t m_errCode;
};

inline void check_error(cudaError_t err) {
	if(err != cudaSuccess)
		throw CudaException(err);
}

} // namespace mufflon