import os
from matplotlib import pyplot as pp


def main():
    current_dir = os.path.dirname(__file__)
    filename = os.path.join(current_dir, "times.txt")
    with open(filename, "rt") as times_file:
        text = times_file.read()
    parts = text.split(";")
    labels = ["With coalescing", "without coalescing"]
    for p, l in zip(parts, labels):
        pp.plot([float(x) for x in p.split()], label=l)
    pp.ylabel("times, ms")
    pp.legend()
    pp.show()

if __name__ == '__main__':
    main()
