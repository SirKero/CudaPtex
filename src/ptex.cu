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
#include <cuda_fp16.h>


// Calculates the power for an int base and an uin8_t power.
__device__ 
int pow2(uint8_t pow) {
	int result = 1 << pow;
	return result;
}

//Cuda functions
__device__ 
void PtexelFetch(void* res, int faceIdx, float u, float v, int numChannels, void* texArr, 
	const uint32_t* texOffsetArr, const uint8_t* ResLog2U, const uint8_t* ResLog2V, cudaPtex::TextureType texType, bool isTriangle) {
	//calc Res U and Res V from the log2 variants from the array
	int ResU = pow2(ResLog2U[faceIdx]);
	int ResV = pow2(ResLog2V[faceIdx]);

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
	
	//Sample Data depending on type
	switch (texType)
	{
	case cudaPtex::dt_uint8:
		for (int i = 0; i < numChannels; i++) {
			reinterpret_cast<uint8_t*>(res)[i] = reinterpret_cast<uint8_t*>(texArr)[index + i];
		}
		break;
	case cudaPtex::dt_uint16:
		for (int i = 0; i < numChannels; i++) {
			reinterpret_cast<uint16_t*>(res)[i] = reinterpret_cast<uint16_t*>(texArr)[index + i];
		}
		break;
	case cudaPtex::dt_half:
		for (int i = 0; i < numChannels; i++) {
			reinterpret_cast<half*>(res)[i] = reinterpret_cast<half*>(texArr)[index + i];
		}
		break;
	case cudaPtex::dt_float:
		for (int i = 0; i < numChannels; i++) {
			reinterpret_cast<float*>(res)[i] = reinterpret_cast<float*>(texArr)[index + i];
		}
		break;
	}

	
}



__device__
void PtexelFetch(void* res, int faceIdx, float u, float v, cudaPtexture tex) {
	PtexelFetch(res, faceIdx, u, v,tex.numChannels, tex.data, tex.offset, tex.ResLog2U, tex.ResLog2V,tex.texType, tex.isTriangle);
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
	uint32_t totalDataSize = 0;
	auto offsetBuf = std::make_unique<uint32_t[]>(m_numFaces);
	auto resUBuf = std::make_unique<uint8_t[]>(m_numFaces);
	auto resVBuf = std::make_unique<uint8_t[]>(m_numFaces);
	
	//Fill helpers from the Ptex FaceInfo
	offsetBuf[0] = 0;	//first one has no offset
	for (unsigned int i = 0; i < m_numFaces; i++) {
		Ptex::FaceInfo faceInfo = texture->getFaceInfo(i);
		resUBuf[i] = faceInfo.res.ulog2;
		resVBuf[i] = faceInfo.res.vlog2;
		totalDataSize += faceInfo.res.size() * m_numChannels;
		if (i < m_numFaces - 1) 
			offsetBuf[i + 1] = totalDataSize;	//offset for the data
		
	}

	uint32_t extraBufferSize = m_numFaces * sizeof(uint32_t) + 2 * m_numFaces * sizeof(uint8_t);

	
	uint32_t totalDataByteSize = 0;	//DataSize in Bytes (is depending on the Texture Type)

	//Check in which Data type the ptex file is in an read accordingly
	switch (texture->dataType()) {
	case Ptex::DataType::dt_uint8: 
		readPtexture<uint8_t>(texture, totalDataSize, extraBufferSize);
		m_DataType = TextureType::dt_uint8;
		totalDataByteSize += totalDataSize * sizeof(uint8_t);
		break;

	case Ptex::DataType::dt_uint16: 
		readPtexture<uint16_t>(texture, totalDataSize, extraBufferSize);
		m_DataType = TextureType::dt_uint16;
		totalDataByteSize += totalDataSize * sizeof(uint16_t);
		break;

	case Ptex::DataType::dt_float: 
		readPtexture<float>(texture, totalDataSize, extraBufferSize);
		m_DataType = TextureType::dt_float;
		totalDataByteSize += totalDataSize * sizeof(float);
		break;

	case Ptex::DataType::dt_half: 
		readPtexture<half>(texture, totalDataSize, extraBufferSize);
		m_DataType = TextureType::dt_half;
		totalDataByteSize += totalDataSize * sizeof(half);
		break;
	}

	//Copy Data to the GPU
	
	m_offsetPtr = reinterpret_cast<uint32_t*>(& m_dataArr[totalDataByteSize]);
	cudaMemcpy(m_offsetPtr, offsetBuf.get(), m_numFaces * sizeof(uint32_t), cudaMemcpyDefault);
	m_resLog2UPtr = reinterpret_cast<uint8_t*>(&m_dataArr[totalDataByteSize  + m_numFaces * sizeof(uint32_t)]);
	cudaMemcpy(m_resLog2UPtr, resUBuf.get(), m_numFaces * sizeof(uint8_t), cudaMemcpyDefault); 
	m_resLog2VPtr = reinterpret_cast<uint8_t*>(&m_dataArr[totalDataByteSize + m_numFaces * sizeof(uint32_t) + m_numFaces * sizeof(uint8_t)]);
	cudaMemcpy(m_resLog2VPtr, resVBuf.get(), m_numFaces * sizeof(uint8_t), cudaMemcpyDefault);

	//Copy data to gpu
	m_totalDataSize = totalDataSize;

	//release Ptex texture, it is not needed anymore
	texture->release(); 
}

cudaPtexture cudaPtex::getTexture() {
	return cudaPtexture{ getDataPointer(), getOffsetPointer(), getResLog2U(), getResLog2V(), getNumChannels(),m_DataType, m_isTriangle };
}

//TODO: Support half type
template <typename T>
void cudaPtex::readPtexture<T>(Ptex::PtexTexture (*texture),int totalDataSize, int extraBufferSize) {
	static_assert(std::is_same<T,uint8_t>::value || std::is_same<T, uint16_t>::value || std::is_same<T, float>::value || std::is_same<T, half>::value, "Type is not supportet by Ptex");
	
	auto dataBuf = std::make_unique<T[]>(totalDataSize);
	uint64_t offset = 0; //Data offset for desArr
	for (int i = 0; i < texture->numFaces(); i++) {
		std::vector<T> faceDataBuffer;
		int texSize = texture->numChannels() * texture->getFaceInfo(i).res.size();
		faceDataBuffer.resize(texSize);
		texture->getData(i, faceDataBuffer.data(), 0);

		//copyData
		for (int j = 0; j < texSize; j++)
			dataBuf[j + offset] = faceDataBuffer[j];
		
		//add to offset
		offset += texSize;
	}

	//copy to GPU
	m_dataArr = make_udevptr_array<Device::CUDA, char, false>(totalDataSize * sizeof(T) + extraBufferSize);
	cudaMemcpy(m_dataArr.get(), dataBuf.get(), totalDataSize * sizeof(T), cudaMemcpyDefault);

}




