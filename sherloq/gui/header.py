import os
from subprocess import run, PIPE
from odonto.sherloq.gui.utility import exiftool_exe


class HeaderWidget():
    def __init__(self, filename):
        pass
        name = filename.split('/')[-1].split('.')[0]
        temp_file = os.path.join('../data/metadados/', name + '.json')
        p = run([exiftool_exe(), "-htmldump0", filename], stdout=PIPE)
        with open(temp_file, "w") as file:
            file.write(p.stdout.decode("utf-8"))


