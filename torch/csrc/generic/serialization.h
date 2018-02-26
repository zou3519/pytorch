#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/serialization.h"
#else

void THPStorage_(writeFileRaw)(THStorage *self, int fd);

template <class T>
THStorage * THPStorage_(readFileRaw)(T fd, THStorage *storage);

template <class T>
ssize_t THPStorage_(doRead)(T fildes, void* buf, size_t nbytes);

// Only define the following once
#ifndef SERIALIZATION_H
#define SERIALIZATION_H

#endif

#endif
