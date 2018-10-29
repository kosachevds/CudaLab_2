import os
from matplotlib import pyplot as pp


def main():
    current_dir = os.path.dirname(__file__)
    filename = os.path.join(current_dir, "times2.txt")
    with open(filename, "rt") as times_file:
        text = times_file.read()
    values = [float(x) for x in text.split()]
    pp.plot(range(1, len(values) + 1), values)
    pp.ylabel("times, ms")
    pp.xlabel("streams")
    pp.show()

if __name__ == '__main__':
    main()
