import struct
import zlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib import image as mpimg
from ChunksFactory import *


class IlovePng:

    def __init__(self, image):
        self.chunks = []  #all chunks and all info about them
        self.types = []  # only the chunk`s type
        self.idat_data = []
        try:
            self.image = open(image, 'rb')
        except FileNotFoundError:
            print('File not found.')
        self.png_id = b'\x89PNG\r\n\x1a\n'
        if self.image.read(len(self.png_id)) != self.png_id:
            raise Exception("{} is not a PNG file".format(self.image.name))
        self.make_chunks()
        self.check()
        self.idat_make()
        self.show()
        self.transformers(image)
        self.anonymization()

    def anonymization(self):
        """
        image anonymization
        new image with only critical chunks is saved as newpng.png
        """
        print(f"Original image chunks: {self.types[::]} ")
        compulsory_chunks = [b'IHDR', b'PLTE', b'IDAT', b'IEND']
        with open('pics/newpng.png', 'wb') as f_out:
            f_out.write(self.png_id)
            for chunk in self.chunks:
                if chunk.type in compulsory_chunks:
                    f_out.write(chunk.length.to_bytes(4, byteorder='big'))
                    f_out.write(struct.pack('4s', chunk.type))
                    f_out.write(chunk.data)
                    f_out.write(chunk.crc.to_bytes(4, byteorder='big'))
        with open('pics/newpng.png', 'rb') as f_out:
            if f_out.read(len(self.png_id)) != self.png_id:
                raise Exception("{} is not a PNG file".format(f_out.name))
            new_types = []
            while 1:
                chunk_length, = struct.unpack('>I', f_out.read(4))  # '>' big endian, 'I' unsigned int,
                chunk_type, = struct.unpack('4s', f_out.read(4))  # '4s' four chars
                chunk_data = f_out.read(chunk_length)
                chunk_crc, = struct.unpack('>I', f_out.read(4))
                new_types.append(chunk_type)
                if chunk_type == b'IEND':
                    break
            print(f"After anonymization: {new_types[::]} ")

    def transformers(self, image):
        """
        fourier transformation and magnitude
        """
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        # Wykonaj transformatę Fouriera
        fourier = np.fft.fft2(img)
        fourier_shifted = np.fft.fftshift(fourier)
        # Oblicz moduł transformaty Fouriera
        magnitude_spectrum = 20 * np.log(np.abs(fourier_shifted))
        # magnitude_spectrum = np.asarray(20 * np.log10(np.abs(fourier_shifted)), dtype=np.uint8)
        inverse_shifted = np.fft.ifftshift(fourier_shifted)
        inverse_fourier = np.fft.ifft2(inverse_shifted)
        inverse_fourier = np.abs(inverse_fourier)

        # Wyświetl obraz oryginalny i jego widmo
        f1 = plt.figure(1)
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Obraz oryginalny'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Widmo amplitudowe'), plt.xticks([]), plt.yticks([])
        plt.show()

        # sprawdzenie czy jest poprawne
        f2 = plt.figure(2)
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Obraz oryginalny'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(inverse_fourier, cmap='gray')
        plt.title('IFFT'), plt.xticks([]), plt.yticks([])
        plt.show()

    def idat_make(self):
        """
        get IDAT`s data and store it in self.idat_data
        shows IDAT data
        based on : https://pyokagan.name/blog/2019-10-14-png/
        """

        def find_a(r, c):
            return self.idat_data[r * stride + c - idhr_bitd] if c >= idhr_bitd else 0

        def find_b(r, c):
            return self.idat_data[(r - 1) * stride + c] if r > 0 else 0

        def find_c(r, c):
            return self.idat_data[(r - 1) * stride + c - idhr_bitd] if r > 0 and c >= idhr_bitd else 0

        def paeth(a, b, c):
            p = a + b - c
            pa = abs(p - a)
            pb = abs(p - b)
            pc = abs(p - c)
            if pa <= pb and pa <= pc:
                return a
            elif pb <= pc:
                return b
            else:
                return c

        bitd = self.chunks[0].bitd
        bytes_per_pixel = {0: (bitd + 7) // 8, 2: 3 * (bitd // 8), 3: 1, 4: 2 * (bitd // 8), 6: 4 * (bitd // 8)}
        idat = b''.join(x.data for x in self.chunks if x.type == b'IDAT')
        idat = zlib.decompress(idat)
        idhr_bitd = bytes_per_pixel[self.chunks[0].colort]
        stride = self.chunks[0].width * idhr_bitd
        i = 0
        for r in range(self.chunks[0].height):
            filter_type = idat[i]
            i += 1
            for c in range(stride):
                Filt_x = idat[i]
                i += 1
                if filter_type == 0:  #none
                    Recon_x = Filt_x
                elif filter_type == 1:  #sub
                    Recon_x = Filt_x + find_a(r, c)
                elif filter_type == 2:  #ip
                    Recon_x = Filt_x + find_b(r, c)
                elif filter_type == 3:  #average
                    Recon_x = Filt_x + (find_a(r, c) + find_b(r, c)) // 2
                elif filter_type == 4:  #paeth
                    pass
                    Recon_x = Filt_x + paeth(find_a(r, c), find_b(r, c), find_c(r, c))
                else:
                    raise Exception('unknown filter type: ' + str(filter_type))
                self.idat_data.append(Recon_x & 0xff)  # truncation to byte
        if idhr_bitd == 1:
            # greyscale
            plt.imshow(np.array(self.idat_data).reshape((self.chunks[0].height, self.chunks[0].width)), cmap='gray',
                       vmin=0,
                       vmax=255)
        elif idhr_bitd == 2:
            # greyscale with alpha channel
            self.idat_data = np.array(self.idat_data).reshape(
                (self.chunks[0].height, self.chunks[0].width, idhr_bitd))
            grayscale = self.idat_data[:, :, 0]
            alpha = self.idat_data[:, :, 1]
            rgb_img = np.dstack((grayscale, grayscale, grayscale, alpha))
            plt.imshow(rgb_img)
        else:
            # truecolor, truecolor with alpha channel, pallette
            plt.imshow((np.array(self.idat_data).reshape((self.chunks[0].height, self.chunks[0].width, idhr_bitd))))

        plt.title("IDAT")
        plt.show()

    def crc_zlib(self, img_crc):
        """
        check if crc is correct
        """
        return zlib.crc32(img_crc) & 0xffffffff  #ensure that the result is a 32-bit unsigned integer value

    def make_chunks(self):
        """
        read all chunks one by one, create Chunk object and save it to self.chunks
        """
        while (1):
            chunk_length, = struct.unpack('>I', self.image.read(4))  # '>' big endian, 'I' unsigned int,
            chunk_type, = struct.unpack('4s', self.image.read(4))  # '4s' four chars
            chunk_data = self.image.read(chunk_length)
            chunk_crc, = struct.unpack('>I', self.image.read(4))
            cal_crc = self.crc_zlib(chunk_type + chunk_data)
            if chunk_crc != cal_crc:
                raise Exception("Problem with crc in chunk {}".format(chunk_type))
            x = Chunk(chunk_length, chunk_type, chunk_data, chunk_crc)
            self.chunks.append(x)
            self.types.append(chunk_type)
            if chunk_type == b'IEND':
                break

    def show(self):
        """
        show all chunks excl. idat, iend
        """

        def idhr_show(self):
            idhr = self.chunks[0]
            dic = {'width': idhr.width, "height": idhr.height, "bitd": idhr.bitd, "colort": idhr.colort,
                   "compm": idhr.compm, "filterm": idhr.filterm, "interlace": idhr.interlacem}
            print("IDHR INFO")
            print(dic)

        def text_show(self):
            if b'tEXt' in self.types:
                print("tEXt INFO")
                for chunk in self.chunks:
                    if chunk.type == b'tEXt':
                        print("{}: {}".format(chunk.keyword, chunk.text))

        def bkgd_show(self):
            for chunk in self.chunks:
                if chunk.type == b'bKGD':
                    print("bKGD INFO")
                    colort = self.chunks[0].colort
                    if colort == 3:  #indexed color
                        palette_idx, = struct.unpack('>B', chunk.data)
                        print(palette_idx)
                        plte_id = self.types.index(b'PLTE')
                        palette = self.chunks[plte_id].colors
                        color = palette[palette_idx]
                        image_data = np.array([color] * (self.chunks[0].width * self.chunks[0].height),
                                              dtype=np.uint8)
                        image_data = image_data.reshape((self.chunks[0].height, self.chunks[0].width, 3))
                        plt.imshow(image_data)
                        plt.title("bKGD")
                        plt.show()
                    elif colort == 0 or colort == 4:  #grayscale (alpha/no alpha)
                        gray_scale, = struct.unpack('>H', chunk.data)
                        print(gray_scale)
                        image_data = np.full((self.chunks[0].height, self.chunks[0].width, 3), gray_scale,
                                             dtype=np.uint8)
                        plt.imshow(image_data, cmap='gray')
                        plt.title("bKGD")
                        plt.show()
                    elif colort == 2 or colort == 6:  #truecolor (alpha/no alpha)
                        r, g, b = struct.unpack('>HHH', chunk.data)
                        print(r, g, b)
                        image_data = np.full((self.chunks[0].height, self.chunks[0].width, 3), (r, g, b),
                                             dtype=np.uint8)
                        plt.imshow(image_data)
                        plt.title("bKGD")
                        plt.show()
                    else:
                        raise Exception("Colortype out of possible range")

        def hist_show(self):
            for x in self.chunks:
                if x.type == b'hIST':
                    print("hIST INFO")
                    #print(x.color_frequencies)
                    for index, frequency in enumerate(x.hist_data):
                        print(f"{index}: {frequency}")

        def palate_show(self):
            for x in self.chunks:
                if x.type == b'PLTE':
                    print("PLTE INFO")
                    print(x.colors)
                    num_colors = len(x.colors)
                    fig, ax = plt.subplots(1, num_colors, figsize=(num_colors, 1))
                    if num_colors == 1:
                        ax = [ax]
                    for i, color in enumerate(x.colors):
                        ax[i].imshow([[color]], extent=[0, 1, 0, 1])
                        ax[i].axis('off')
                    plt.title("PLTE")
                    plt.show()

        def gama_show(self):
            for x in self.chunks:
                if x.type == b'gAMA':
                    print("gAMA INFO")
                    print(f"Gamma value: {x.value}")

        def chrm_show(self):
            for x in self.chunks:
                if x.type == b'cHRM':
                    print("cHRM INFO")
                    print(x.data_chrm)

        def time_show(self):
            for x in self.chunks:
                if x.type == b'tIME':
                    print("tIME INFO")
                    print(x.data_time)

        idhr_show(self)
        text_show(self)
        bkgd_show(self)
        hist_show(self)
        palate_show(self)
        gama_show(self)
        chrm_show(self)
        time_show(self)

    def check(self):
        """
        checks the validity of chunks
        """

        def check_ihdr(self):
            assert self.chunks[0].type == b'IHDR', print('IHDR is not a first chunk or doesnt exists')
            assert self.types.count(b'IHDR') == 1, "There are either zero or multiple IHDR chunks. Should be only one"
            if self.chunks[0].colort == 3:
                assert self.types.count(b'PLTE') == 1, "PLTE chunk must appear"
            if self.chunks[0].colort == 0 or self.chunks[0].colort == 4:
                assert self.types.count(b'PLTE') == 0, "PLTE chunk must not appear"

        def check_idat(self):
            assert self.types.count(b'IDAT') >= 1, "IDAT chunk must appear"
            idat_count = self.types.count(b'IDAT')
            first_idat_id = self.types.index(b'IDAT')
            assert [self.types[x] == (b'IDAT') for x in
                    range(first_idat_id, first_idat_id + idat_count)], " Multiple IDATs must be consecutive"

        def check_iend(self):
            assert self.chunks[-1].type == b'IEND', print('IEND is not the last chunk or doesnt exists')
            assert self.types.count(b'IEND') == 1, "IEND chunk must appear only once"

        def check_plte(self):
            if b'PLTE' not in self.types:
                return
            plte_idx = self.types.index(b'PLTE')
            assert self.chunks[plte_idx].length % 3 == 0, "Invalid PLTE chunk length. Length should be a multiple of 3"
            assert plte_idx < self.types.index(b'IDAT'), "PLTE chunk should appear before IDAT chunk"
            assert self.types.count(b'PLTE') == 1, "There are either zero or multiple PLTE chunks. Should be only one"
            assert len(self.chunks[plte_idx].data) / 3 <= pow(2, self.chunks[
                0].bitd), "The number of palette entries exceed the range that can be represented in the image bit depth"

        def check_gamma(self):
            if b'gAMA' not in self.types:
                return
            if b'PLTE' in self.types:
                assert self.types.index(b'gAMA') < self.types.index(b'PLTE'), "gAMA chunk must precede the PLTE chunk"
            assert self.types.index(b'gAMA') < self.types.index(b'IDAT'), "gAMA chunk must precede the first IDAT chunk"
            assert self.types.count(b'gAMA') == 1, "There are either zero or multiple gAMA chunks. Should be only one"

        def check_time(self):
            if b'tIME' not in self.types:
                return
            assert self.types.count(b'tIME') == 1, "There are either zero or multiple tIME chunks. Should be only one"

        def check_chrm(self):
            if b'cHRM' not in self.types:
                return
            assert self.types.count(b'cHRM') == 1, "There are either zero or multiple cHRM chunks. Should be only one"
            assert self.types.index(b'cHRM') < self.types.index(b'IDAT'), "cHRM chunk must precede the first IDAT chunk"
            if b'PLTE' in self.types:
                assert self.types.index(b'cHRM') < self.types.index(b'PLTE'), "cHRM chunk must precede the PLTE chunk"

        def check_text(self):
            pass

        def check_bkgd(self):
            if b'bKGD' not in self.types:
                return
            bkgd_indx = self.types.index(b'bKGD')
            assert self.types.count(b'bKGD') == 1, "There are either zero or multiple bKGD chunks. Should be only one"
            assert bkgd_indx < self.types.index(b'IDAT'), "bKGD chunk must precede the first IDAT chunk"
            if b'PLTE' in self.types:
                assert bkgd_indx > self.types.index(b'PLTE'), "bKGD chunk must be after the PLTE chunk"
            assert self.chunks[0].colort in [0, 2, 3, 4,
                                             6], "bKGD chunk is not applicable for the color type of the image."
            if self.chunks[0].colort in [0, 4]:  #grayscale
                assert len(
                    self.chunks[self.types.index(
                        b'bKGD')].data) == 2, "Invalid bKGD chunk length for grayscale images. Expected length: 2 bytes."
            elif self.chunks[0].colort in [2, 6]:  #truecolor
                assert len(
                    self.chunks[self.types.index(
                        b'bKGD')].data) == 6, "Invalid bKGD chunk length for truecolor images. Expected length: 6 bytes."
            elif self.chunks[0].colort == 3:  #indexed-color
                assert len(
                    self.chunks[self.types.index(
                        b'bKGD')].data) == 1, "Invalid bKGD chunk length for indexed-color images. Expected length: 1 byte."

        def check_hist(self):
            if b'hIST' not in self.types:
                return
            assert self.types.index(b'hIST') < self.types.index(b'IDAT'), "hIST chunk must precede the first IDAT chunk"
            assert b'PLTE' in self.types, "PLTE chunk must appear"
            assert self.types.index(b'hIST') > self.types.index(b'PLTE'), "hIST chunk must be after the PLTE chunk"
            assert self.types.count(b'hIST') == 1, "There are either zero or multiple hIST chunks.Should be only one"

        check_ihdr(self)
        check_plte(self)
        check_idat(self)
        check_iend(self)
        check_gamma(self)
        check_chrm(self)
        check_text(self)
        check_time(self)
        check_hist(self)
        check_bkgd(self)


if __name__ == '__main__':
    image = 'pics/cat.png'
    image1 = mpimg.imread(image)
    plt.imshow(image1)
    plt.title("Oryginalne zdjęcie")
    plt.show()
    img_decoded = IlovePng(image)
