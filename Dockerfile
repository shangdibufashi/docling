FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN \
    echo "Running apt update." && \
    apt update  && \
    echo "Installing dependencies with apt."  && \
    apt install -y cmake libgtk-3-dev libgtkmm-3.0-dev liblensfun-dev librsvg2-dev \
        liblcms2-dev libfftw3-dev libiptcdata0-dev libtiff5-dev libcanberra-gtk3-dev \
        liblensfun-bin libexpat1-dev libbrotli-dev zlib1g-dev libinih-dev \
        adwaita-icon-theme-full gettext libarchive-tools zstd libgif-dev \
        libwebp-dev libwebpdemux2 \
        cmake libomp-dev libjpeg-dev libopencv-contrib-dev \
        libopencv-dev zlib1g-dev libinih-dev gettext libarchive-tools zstd \
        build-essential pkg-config autoconf libtool libcairo2-dev libcairo-gobject-dev \
        libharfbuzz-dev libxml2-dev libpango1.0-dev libglib2.0-dev  \
        build-essential fakeroot  gawk  lsb-release  curl ca-certificates git bash wget curl sudo
RUN \
    echo "Installing makedeb..." && \
    curl -Ss -qgb "" -fLC - --retry 3 --retry-delay 3 -o makedeb.deb \
    "https://github.com/makedeb/makedeb/releases/download/v16.1.0-beta1/makedeb-beta_16.1.0-beta1_amd64_focal.deb"  && \
    dpkg -i makedeb.deb

RUN \
    MAIN_VERSION='5.11' && \
    echo "Cloning RawTherapee $MAIN_VERSION." && \
    git clone --depth 1 --branch "$MAIN_VERSION" https://github.com/RawTherapee/RawTherapee.git ./main

RUN \
    useradd -m -s /bin/bash builder && \
    echo 'builder ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R builder:builder ./main

USER builder

RUN cd ./main/tools/makedeb  && \
    echo "Building and installing libjxl..." && \
    makedeb -si --no-confirm -p PKGBUILD.libjxl

USER root

RUN cd ./main && \
    EXIV2_VERSION='v0.28.3' && \
    echo "Cloning Exiv2 $EXIV2_VERSION." && \
    git clone --depth 1 --branch "$EXIV2_VERSION" https://github.com/Exiv2/exiv2.git ext/exiv2 && \
    \
    echo "Configuring build." && \
    mkdir ext/exiv2/build && \
    cd ext/exiv2/build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DEXIV2_ENABLE_BMFF=ON .. && \
    \
    echo "Building and installing." && \
    make -j$(nproc) install

RUN cd ./main && \
    LIBRSVG2_VERSION='2.52.2' && \
    echo "Cloning Librsvg2 $LIBRSVG2_VERSION." && \
    git clone --depth 1 --branch "$LIBRSVG2_VERSION" https://gitlab.gnome.org/GNOME/librsvg.git ext/librsvg2 && \
    \
    echo "Installing required dependencies with apt." && \
    apt install -y rustc cargo gtk-doc-tools libgirepository1.0-dev && \
    \
    echo "Updating PATH." && \
    export PATH="$PATH:/usr/lib/x86_64-linux-gnu/gdk-pixbuf-2.0" && \
    \
    echo "Configuring build." && \
    cd ext/librsvg2 && \
    sh autogen.sh && \
    \
    echo "Building and installing." && \
    make install
          
