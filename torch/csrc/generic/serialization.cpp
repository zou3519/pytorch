#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/serialization.cpp"
#else

#define SYSCHECK(call) { ssize_t __result = call; if (__result < 0) throw std::system_error((int) __result, std::system_category()); }

// Only define the following once
// #ifndef SERIALIZATION_CPP
// #define SERIALIZATION_CPP

template <>
ssize_t THPStorage_(doRead)<int>(int fildes, void* buf, size_t nbyte) {
  return read(fildes, buf, nbyte);
}

template <>
ssize_t THPStorage_(doRead)<PyObject*>(PyObject* fildes, void* buf, size_t nbyte) {
  // PyMemoryView_FromMemory doesn't exist in Python 2.7, so we manually
  // create a Py_buffer that describes the memory and create a memoryview from it.
  Py_buffer pyBuf;
  pyBuf.buf = buf;
  pyBuf.obj = nullptr;
  pyBuf.len = (Py_ssize_t)nbyte;
  pyBuf.itemsize = 1;
  pyBuf.readonly = 0;
  pyBuf.ndim = 0;
  pyBuf.format = nullptr;
  pyBuf.shape = nullptr;
  pyBuf.strides = nullptr;
  pyBuf.suboffsets = nullptr;
  pyBuf.internal = nullptr;

  THPObjectPtr pyMemoryView(PyMemoryView_FromBuffer(&pyBuf));
  if (!pyMemoryView) throw python_error();
  THPObjectPtr r(PyObject_CallMethod(fildes, "readinto", "O", pyMemoryView.get()));
  if (!r) throw python_error();
  return PyLong_AsSsize_t(r.get());
}

// #endif


void THPStorage_(writeFileRaw)(THStorage *self, int fd)
{
  real *data;
  int64_t size = self->size;
#ifndef THC_GENERIC_FILE
  data = self->data;
#else
  std::unique_ptr<char[]> cpu_data(new char[size * sizeof(real)]);
  data = (real*)cpu_data.get();
  THCudaCheck(cudaMemcpy(data, self->data, size * sizeof(real), cudaMemcpyDeviceToHost));
#endif
  ssize_t result = write(fd, &size, sizeof(int64_t));
  if (result != sizeof(int64_t))
    throw std::system_error(result, std::system_category());
  // fast track for bytes and little endian
  if (sizeof(real) == 1 || THP_nativeByteOrder() == THPByteOrder::THP_LITTLE_ENDIAN) {
    char *bytes = (char *) data;
    int64_t remaining = sizeof(real) * size;
    while (remaining > 0) {
      // we write and read in 1GB blocks to avoid bugs on some OSes
      ssize_t result = write(fd, bytes, THMin(remaining, 1073741824));
      if (result < 0)
        throw std::system_error(result, std::system_category());
      bytes += result;
      remaining -= result;
    }
    if (remaining != 0)
      throw std::system_error(result, std::system_category());
  } else {
    int64_t buffer_size = std::min(size, (int64_t)5000);
    std::unique_ptr<uint8_t[]> le_buffer(new uint8_t[buffer_size * sizeof(real)]);
    for (int64_t i = 0; i < size; i += buffer_size) {
      size_t to_convert = std::min(size - i, buffer_size);
      if (sizeof(real) == 2) {
        THP_encodeInt16Buffer((uint8_t*)le_buffer.get(),
            (const int16_t*)data + i,
            THPByteOrder::THP_LITTLE_ENDIAN,
            to_convert);
      } else if (sizeof(real) == 4) {
        THP_encodeInt32Buffer((uint8_t*)le_buffer.get(),
            (const int32_t*)data + i,
            THPByteOrder::THP_LITTLE_ENDIAN,
            to_convert);
      } else if (sizeof(real) == 8) {
        THP_encodeInt64Buffer((uint8_t*)le_buffer.get(),
            (const int64_t*)data + i,
            THPByteOrder::THP_LITTLE_ENDIAN,
            to_convert);
      }
      SYSCHECK(write(fd, le_buffer.get(), to_convert * sizeof(real)));
    }
  }
}

template <class T> // Added the template...
THStorage * THPStorage_(readFileRaw)(T fd, THStorage *_storage)
{
  real *data;
  int64_t size;
  ssize_t result = THPStorage_(doRead<T>)(fd, &size, sizeof(int64_t));
  if (result == 0)
    throw std::runtime_error("unexpected EOF. The file might be corrupted.");
  if (result != sizeof(int64_t))
    throw std::system_error(result, std::system_category());
  THStoragePtr storage;
  if (_storage == nullptr) {
    storage = THStorage_(newWithSize)(LIBRARY_STATE size);
  } else {
    THPUtils_assert(_storage->size == size,
        "storage has wrong size: expected %ld got %ld",
        size, _storage->size);
    storage = _storage;
  }

#ifndef THC_GENERIC_FILE
  data = storage->data;
#else
  std::unique_ptr<char[]> cpu_data(new char[size * sizeof(real)]);
  data = (real*)cpu_data.get();
#endif

  // fast track for bytes and little endian
  if (sizeof(real) == 1 || THP_nativeByteOrder() == THPByteOrder::THP_LITTLE_ENDIAN) {
    char *bytes = (char *) data;
    int64_t remaining = sizeof(real) * storage->size;
    while (remaining > 0) {
      // we write and read in 1GB blocks to avoid bugs on some OSes
      ssize_t result = THPStorage_(doRead)<T>(fd, bytes, THMin(remaining, 1073741824));
      if (result == 0) // 0 means EOF, which is also an error
        throw std::runtime_error("unexpected EOF. The file might be corrupted.");
      if (result < 0)
        throw std::system_error(result, std::system_category());
      bytes += result;
      remaining -= result;
    }
    if (remaining != 0)
      throw std::system_error(result, std::system_category());
  } else {
    int64_t buffer_size = std::min(size, (int64_t)5000);
    std::unique_ptr<uint8_t[]> le_buffer(new uint8_t[buffer_size * sizeof(real)]);
    for (int64_t i = 0; i < size; i += buffer_size) {
      size_t to_convert = std::min(size - i, buffer_size);
      SYSCHECK(THPStorage_(doRead<T>)(fd, le_buffer.get(), sizeof(real) * to_convert));
      if (sizeof(real) == 2) {
        THP_decodeInt16Buffer((int16_t*)data + i,
            le_buffer.get(),
            THPByteOrder::THP_LITTLE_ENDIAN,
            to_convert);
      } else if (sizeof(real) == 4) {
        THP_decodeInt32Buffer((int32_t*)data + i,
            le_buffer.get(),
            THPByteOrder::THP_LITTLE_ENDIAN,
            to_convert);
      } else if (sizeof(real) == 8) {
        THP_decodeInt64Buffer((int64_t*)data + i,
            le_buffer.get(),
            THPByteOrder::THP_LITTLE_ENDIAN,
            to_convert);
      }
    }
  }

#ifdef THC_GENERIC_FILE
  THCudaCheck(cudaMemcpy(storage->data, data, size * sizeof(real), cudaMemcpyHostToDevice));
#endif
  return storage.release();
}

#undef SYSCHECK

#endif
