FROM ubuntu:16.04 AS build

# Install fasttext
# Run `docker build --no-cache .` to update dependencies
RUN apt-get update; apt-get install -y apt-utils software-properties-common; add-apt-repository ppa:ubuntu-toolchain-r/test -y; add-apt-repository ppa:acooks/libwebsockets6 -y; apt-get update; apt-get install build-essential -y && apt-get install --fix-missing gcc-8 g++-8 -y &&  update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 60 --slave /usr/bin/g++ g++ /usr/bin/g++-8 && update-alternatives --config gcc; apt-get install git git-flow -y; apt-get install wget -y; apt-get install libblas-dev liblapack-dev ccache libssl-dev zlib1g-dev pkg-config libuv1.dev libomp-dev -y

# Install packages
WORKDIR /tmp
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.14.5/cmake-3.14.5-Linux-x86_64.sh; sh ./cmake-3.14.5-Linux-x86_64.sh --skip-license --include-subdir; cd cmake-3.14.5-Linux-x86_64; cp -r bin /usr/; cp -r share /usr/; cp -r doc /usr/share/; cp -r man /usr/share/; cd ../; rm -rf cmake-3.14.5-Linux-x86_64; rm -f cmake-3.14.5-Linux-x86_64.sh; git clone https://github.com/google/googletest && cd googletest; mkdir build; cd build; cmake .. && make -j 4; make install && cd /tmp; rm -rf googletest; git clone https://github.com/facebookresearch/fastText.git; cd fastText; mkdir build; cd build; cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..; make; make install; cd /tmp; rm -rf fastText
