import struct
import zlib
import matplotlib.pyplot as plt
import numpy as np
import cv2


class Chunk:

    def __init__(self, length, type, data, crc):
        self.length = length
        self.type = type
        self.data = data
        self.crc = crc
        if self.type == b'IHDR':
            ihdr_data = struct.unpack('>IIBBBBB', self.data)
            self.width, self.height, self.bitd, self.colort, self.compm, self.filterm, self.interlacem = ihdr_data
            allowed_bit_depths = {
                0: [1, 2, 4, 8, 16],  #grayscale
                2: [8, 16],  #truecolor
                3: [1, 2, 4, 8],  #indexed-color
                4: [8, 16],  #grayscale with alpha
                6: [8, 16]  #truecolor with alpha
            }
            assert len(self.data) == 13, print("Invalid IHDR chunk length")
            assert self.width > 0 and self.height > 0, "Invalid image dimensions"
            assert self.bitd in [1, 2, 4, 8, 16], "Invalid bit depth"
            assert self.colort in [0, 2, 3, 4, 6], "Invalid color type"
            assert self.bitd in allowed_bit_depths.get(self.colort,
                                                       []), "Invalid bit depth for the specified color type"
            assert self.compm == 0, "Invalid compression method"
            assert self.filterm == 0, "Invalid filter method"
            assert self.interlacem in [0, 1], "Invalid interlace method"
        if self.type == b'PLTE':
            """
            sluza do opisywania koloru za pomoca numerow"""
            self.colors = []
            for i in range(0, len(self.data), 3):
                r = self.data[i]
                g = self.data[i + 1]
                b = self.data[i + 2]
                assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255, "Invalid palette entry values. RGB values should be in the range 0-255."
                self.colors.append((r, g, b))
        if self.type == b'IEND':
            assert len(self.data) == 0, "Invalid IEND chunk length. Expected length: 0."
        if self.type == b'gAMA':
            data_gamma, = struct.unpack('>I', self.data)
            self.value = data_gamma / 100000
            assert 0.1 <= self.value <= 1.0, "Invalid gamma value. Gamma should be in the range 0.1 to 1.0."
            assert len(self.data) == 4, "Invalid gAMA chunk length. Expected length: 4 bytes."
            """
            The value is encoded as a 4-byte unsigned integer, representing gamma times 100000. 
            For example, a gamma of 0.45 would be stored as the integer 45000.
            """
        if self.type == b'tEXt':
            self.keyword, self.text = self.data.decode('utf-8').split('\x00')

            def is_latin_character(char):
                return (32 <= ord(char) <= 126) or (161 <= ord(char) <= 255)

            assert 0 < len(
                self.keyword) < 80, "The keyword must be at least one character and less than 80 characters long."
            assert all(is_latin_character(char) for char in self.keyword), "Invalid character in keyword."
            self.string = (f"Keyword: {self.keyword}, Text: {self.text}")
        if self.type == b'cHRM':
            """
            okresla polozenie punktow bieli,czerwonego,zielonego,niebieskiego.sluzy do umiejscowienia obrazu w calym ekosystemie przestrzeni kolorystycznych
            """
            assert len(self.data) == 32, "Invalid cHRM chunk length. Expected length: 32 bytes"
            self.white_x, self.white_y, self.red_x, self.red_y, self.green_x, self.green_y, self.blue_x, self.blue_y = struct.unpack(
                '>IIIIIIII', self.data)
            self.white_x /= 100000
            self.white_y /= 100000
            self.red_x /= 100000
            self.red_y /= 100000
            self.green_x /= 100000
            self.green_y /= 100000
            self.blue_x /= 100000
            self.blue_y /= 100000
            self.data_chrm = {"white_x": self.white_x, "white_y": self.white_y, "red_x": self.red_x,
                              "red_y": self.red_y, "green_x": self.green_x, "green_y": self.green_y,
                              "blue_x": self.blue_x, "blue_y": self.blue_y}
        if self.type == b'tIME':
            """
            The tIME chunk gives the time of the last image modification (not the time of initial image creation).
            """
            year, month, day, hour, minute, second, = struct.unpack('>HBBBBB', self.data)
            assert 0 < month < 13 and 0 < day < 32 and 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 60, "Something wrong with date in the tIME chunk"
            assert len(self.data) == 7, "Invalid tIME chunk length. Expected length: 7 bytes."
            if int(month) < 10: month = "0{}".format(month)
            if int(day) < 10: day = "0{}".format(day)
            if int(hour) < 10: hour = "0{}".format(hour)
            if int(minute) < 10: minute = "0{}".format(minute)
            if int(second) < 10: second = "0{}".format(second)
            self.data_time = "{}/{}/{}\n{}:{}:{}".format(day, month, year, hour, minute, second)

        if self.type == b'hIST':
            self.hist_data = struct.unpack(f'>{self.length}B', self.data)
            self.color_frequencies = [(index, frequency) for index, frequency in enumerate(self.hist_data)]
        if self.type == b'bKGD':
            assert len(self.data) in [1, 2, 6], "Invalid bKGD chunk length."
