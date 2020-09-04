/*
 PTEX SOFTWARE
 Copyright 2014 Disney Enterprises, Inc.  All rights reserved
 */

#include "ptex.hpp"
#include <iostream>
#include <math.h>
#include <memory>
#include <vector>
#include <type_traits>
#include <stdint.h>
//CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// Calculates the power for an int base and an uin8_t power.
__device__ 
int powI(int base, uint8_t pow) {
	int result = base;
	for (int i = 1; i < pow; i++) {
		result *= base;
	}
	return result;
}

//Cuda functions
__device__ 
void PtexelFetch(float* res, int faceIdx, float u, float v, int numChannels, const float* texArr, 
	const uint32_t* texOffsetArr, const uint8_t* ResLog2U, const uint8_t* ResLog2V, bool isTriangle) {
	//calc Res U and Res V from the log2 variants from the array
	int ResU = powI(2, ResLog2U[faceIdx]);
	int ResV = powI(2, ResLog2V[faceIdx]);

	int offset = texOffsetArr[faceIdx];
	int index;
	if (!isTriangle) {
		 index = offset + ResU * numChannels * (u + v * ResV);
	}
	//For triangles: texture fetch after http://ptex.us/tritex.html
	else {
		float resf = __int2float_rz(ResU);		
		float ut = u * resf;
		float vt = v * resf;
		float uIdx = floorf(ut);
		float vIdx = floorf(vt);
		int tmpIndex;
		if ((ut - uIdx) + (vt - vIdx) <= 1.0f) {
			tmpIndex = __float2int_rz(uIdx + vIdx * resf);
		}
		else {
			tmpIndex = __float2int_rz((resf * resf - 1.0f) - (vIdx + uIdx * resf));
		}

		int iU = tmpIndex % ResU;
		int iV = tmpIndex / ResV;

		index = offset + numChannels * (iU + iV * ResU);

	}

	for (int i = 0; i < numChannels; i++) {
		res[i] = texArr[index + i];
	}
}



__device__
void PtexelFetch(float* res, int faceIdx, float u, float v, cudaPtexture tex) {
	PtexelFetch(res, faceIdx, u, v,tex.numChannels, tex.data, tex.offset, tex.ResLog2U, tex.ResLog2V, tex.isTriangle);
}

void cudaPtex::loadFile(const char* filepath, bool premultiply) {
	
	//Load texture from file
	Ptex::PtexTexture* texture;
	Ptex::String ptexErr;
	texture = Ptex::PtexTexture::open(filepath, ptexErr, premultiply);
	//In experience this only triggers if the file is not a ptex file
	if (texture == nullptr) {
		std::cerr << "Ptex Error: " << ptexErr.c_str() << '\n';
		return;
	}

	//Get info about the texture
	m_numFaces = texture->numFaces();
	m_numChannels = texture->numChannels();
	
	if (texture->meshType() == Ptex::MeshType::mt_triangle) {
		m_isTriangle = true;
	}
	
	//Create CPU side Buffers
	uint64_t totalDataSize = 0;
	auto offsetBuf = std::make_unique<uint32_t[]>(m_numFaces);
	auto resUBuf = std::make_unique<uint8_t[]>(m_numFaces);
	auto resVBuf = std::make_unique<uint8_t[]>(m_numFaces);
	
	//Fill helpers from the Ptex FaceInfo
	offsetBuf[0] = 0;	//first one has no offset
	for (int i = 0; i < m_numFaces; i++) {
		Ptex::FaceInfo faceInfo = texture->getFaceInfo(i);
		resUBuf[i] = faceInfo.res.ulog2;
		resVBuf[i] = faceInfo.res.vlog2;
		totalDataSize += faceInfo.res.size() * m_numChannels;
		if (i < m_numFaces - 1) 
			offsetBuf[i + 1] = totalDataSize;	//offset for the data
		
	}

	//Copy data to GPU
	m_offsets = make_udevptr_array < Device::CUDA, uint32_t, false>(m_numFaces);
	m_ResLog2U = make_udevptr_array < Device::CUDA, uint8_t, false>(m_numFaces);
	m_ResLog2V = make_udevptr_array < Device::CUDA, uint8_t, false>(m_numFaces);
	
	cudaMemcpy(m_offsets.get(), offsetBuf.get(), sizeof(uint32_t) * m_numFaces, cudaMemcpyDefault);
	cudaMemcpy(m_ResLog2U.get(), resUBuf.get(), sizeof(uint8_t) * m_numFaces, cudaMemcpyDefault);
	cudaMemcpy(m_ResLog2V.get(), resVBuf.get(), sizeof(uint8_t) * m_numFaces, cudaMemcpyDefault);

	//Release tmpBuffers
	offsetBuf.release();
	resUBuf.release();
	resVBuf.release();
	
	//tmpBuffer for dataArray
	auto dataBuf = std::make_unique<float[]>(totalDataSize);

	Ptex::DataType dataType = texture->dataType();

	//Check in which Data type the ptex file is in an read accordingly
	switch (texture->dataType()) {
	case Ptex::DataType::dt_uint8 :
		readPtexture<uint8_t>(dataBuf.get(), texture);
		break;
	case Ptex::DataType::dt_uint16 :
		readPtexture<uint16_t>(dataBuf.get(), texture);
		break;
	case Ptex::DataType::dt_float :
		readPtexture<float>(dataBuf.get(), texture);
		break;
	//TODO: support half data type
	default:
		std::cerr << "Ptex Error: half Data Type is not supported";
	}

	//Copy data to gpu
	m_totalDataSize = totalDataSize;
	m_data = make_udevptr_array<Device::CUDA, float, false>(totalDataSize);

	cudaMemcpy(m_data.get(), dataBuf.get(), totalDataSize * sizeof(float), cudaMemcpyDefault);

	//release Ptex texture, it is not needed anymore
	texture->release(); 
}


//TODO: Support half type
template <typename T>
void cudaPtex::readPtexture<T>(float* desArr, Ptex::PtexTexture (*texture)) {
	static_assert(std::is_same<T,uint8_t>::value || std::is_same<T, uint16_t>::value || std::is_same<T, float>::value, "Ptex has a not supported type");

	uint64_t offset = 0; //Data offset for desArr
	for (int i = 0; i < texture->numFaces(); i++) {
		std::vector<T> faceDataBuffer;
		int texSize = texture->numChannels() * texture->getFaceInfo(i).res.size();
		faceDataBuffer.resize(texSize);
		texture->getData(i, faceDataBuffer.data(), 0);

		//if it is a float it can be copied
		if (std::is_same<T, float>::value) {	
			for (int j = 0; j < texSize; j++) 
				desArr[j + offset] = faceDataBuffer[j];
		}
		//uint8 and uint16 has to be converted to the range from 0 to 1
		else {
			T max = std::numeric_limits<T>::max();	//max val of uint8 and uint16 for division
			float maxf = static_cast<float>(max);
			for (int j = 0; j < texSize; j++) 
				desArr[j + offset] = static_cast<float>(faceDataBuffer[j]) / maxf;
		}
		//add to offset
		offset += texSize;
	}
}




