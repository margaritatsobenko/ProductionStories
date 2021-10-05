import argparse
import math
import os
from typing import List, Tuple

import scipy.stats as ss


def get_data(path: str) -> List[Tuple[float, float]]:
    data = []
    with open(path, 'r') as file:
        for line in file:
            a = line.split()
            data.append((float(a[0]), float(a[1])))
    return data


def write_data(dif_R: int, st_e: int, conj: float, path: str):
    ans = [dif_R, st_e, conj]
    with open(path, "w") as text_file:
        text_file.write(' '.join(str(x) for x in ans))


def mon_conj_test(data: List[Tuple[float, float]]) -> Tuple[int, int, float]:
    sort_data = sorted(data)
    N = len(data)
    y = [i[1] for i in sort_data]
    rank_data = (N + 1) - ss.rankdata(y, method='average')
    p = round(N / 3)

    R_1 = sum(rank_data[:p])
    R_2 = sum(rank_data[-p:])
    dif_R = int(round(R_1 - R_2))

    st_e = (N + 0.5) * math.sqrt(p / 6)
    st_e = int(round(st_e))

    conj = round(dif_R / (p * (N - p)), 2)
    return dif_R, st_e, conj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="in.txt", help="input file")
    parser.add_argument("--output", type=str, default="out.txt", help="output file")
    args = parser.parse_args()

    data = get_data(args.input)

    if len(data) < 9:
        raise ValueError("Number of examples < 9!")

    path = os.path.dirname(args.output)
    if path and not os.path.exists(path):
        os.makedirs(path)

    dif_R, st_e, conj = mon_conj_test(data)
    write_data(dif_R, st_e, conj, args.output)


if __name__ == '__main__':
    main()
