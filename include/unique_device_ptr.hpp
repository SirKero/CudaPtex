#pragma once

#include "residency.hpp"
#include <memory>
#include "allocator.hpp"

namespace mufflon {
template < Device dev, typename T >
class unique_device_ptr : public std::unique_ptr<T, Deleter<dev>> {
	static constexpr Device DEVICE = dev;
	// inherit constructors
	using std::unique_ptr<T, Deleter<dev>>::unique_ptr;
};

template < Device dev, typename T, typename... Args >
inline unique_device_ptr<dev, T> make_udevptr(Args... args) {
	return unique_device_ptr<dev, T>(
		Allocator<dev>::template alloc<T>(std::forward<Args>(args)...),
		Deleter<dev>(1)
		);
}

template < Device dev, typename T, bool Init = true, typename... Args >
inline unique_device_ptr<dev, T[]> make_udevptr_array(std::size_t n, Args... args) {
	return unique_device_ptr<dev, T[]>(
		Allocator<dev>::template alloc_array<T, Init>(n, std::forward<Args>(args)...),
		Deleter<dev>(n)
		);
}
}
