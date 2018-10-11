import os
from matplotlib import pyplot as pp


def main():
    current_dir = os.path.dirname(__file__)
    filename = os.path.join(current_dir, "times3.txt")
    with open(filename, "rt") as times_file:
        text = times_file.read()
    parts = text.split(";")
    pp.subplot(1, 2, 1)
    pp.title("CPU")
    pp.plot([float(x) for x in parts[0].split()])
    pp.ylabel("times, ms")
    pp.subplot(1, 2, 2)
    for pair in zip(parts[1:], ["Global", "Shared"]):
        pp.plot([float(x) for x in pair[0].split()], label=pair[1])
    pp.legend()
    pp.title("GPU")
    pp.ylabel("times, ms")
    pp.show()

if __name__ == '__main__':
    main()
