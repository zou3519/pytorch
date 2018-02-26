#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/serialization.h"
#else

// Only define the following once (instead of once per storage class)
#ifndef _SERIALIZATION_H
#define _SERIALIZATION_H

template <class T>
ssize_t doRead(T fildes, void* buf, size_t nbytes);

template <class T>
off_t doSeek(T fildes, off_t offset, int whence);

template <>
inline ssize_t doRead<int>(int fildes, void* buf, size_t nbyte) {
  return read(fildes, buf, nbyte);
}

template <>
inline off_t doSeek(int fildes, off_t offset, int whence) {
  return lseek(fildes, offset, whence);
}

template <>
inline ssize_t doRead<PyObject*>(PyObject* fildes, void* buf, size_t nbyte) {
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

template <>
inline off_t doSeek<PyObject*>(PyObject* fildes, off_t offset, int whence) {
  printf("seek\n");
  THPObjectPtr r(PyObject_CallMethod(fildes, "seek", "ii", offset, whence));
  if (!r) throw python_error();
  printf("seek2\n");
  return PyLong_AsLong(r.get());
}

#endif // _SERIALIZATION_H

void THPStorage_(writeFileRaw)(THStorage *self, int fd);

template <class T>
THStorage * THPStorage_(readFileRaw)(T fd, THStorage *storage);

#endif
