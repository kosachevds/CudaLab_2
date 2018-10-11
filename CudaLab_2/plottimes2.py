import os
from matplotlib import pyplot as pp


def main():
    current_dir = os.path.dirname(__file__)
    filename = os.path.join(current_dir, "times.txt")
    with open(filename, "rt") as times_file:
        text = times_file.read()
    parts = text.split(";")
    for p in parts:
        pp.plot([float(x) for x in p.split()])
    pp.ylabel("times, ms")
    pp.show()

if __name__ == '__main__':
    main()
