"""
Jakub Susoł 274300
Zadanie 1 zaliczenie
Algorytmika

Algorytm MedianString
(Bioinformatics Algorithms - An Active Learning Approach vol. I, strona 82 (plus poprawki) 107)
"""
from MotifEnumeration import readSequences

def patternToNumber(pattern: str):
    sym2num = {"A": 0, "C": 1, "G": 2, "T": 3}
    if not pattern:
        return 0
    symbol = pattern[-1]
    prefix = pattern[:-1]
    return 4 * patternToNumber(prefix) + sym2num[symbol]


def numberToPattern(index: int, k: int):
    def genstr(index, k):
        num2sym = {0: "A", 1: "C", 2: "G", 3: "T"}
        pattern = ''
        if k == 1:
            return pattern.join(num2sym[index % 4])
        prefixIndex = index // 4
        r = index % 4
        pattern += num2sym[r]
        pattern += genstr(prefixIndex, k - 1)
        return pattern

    pattern = ''.join(reversed(list(genstr(index, k))))
    return pattern


def distancePatternString(pattern: str, dna: list):
    def hammingDistance(text, compare):
        distance = 0
        for a, b in zip(text, compare):
            if a != b:
                distance += 1
        return distance
    k = len(pattern)
    distance = 0
    for sequence in dna:
        hammingDist = 1e100
        for i in range(len(sequence) - k):
            patternprime = sequence[i:i + k]
            if hammingDist > hammingDistance(pattern, patternprime):
                hammingDist = hammingDistance(pattern, patternprime)
        distance += hammingDist
    return distance

def medianString(dna: list, k: int):
    distance = 1e100
    median = ''
    for i in range(4**k):
        print(f'Comparing pattern {i+1}/{4**k}')
        pattern = numberToPattern(i, k)
        patternDist = distancePatternString(pattern, dna)
        if distance > patternDist:
            distance = patternDist
            median = pattern
    return median

DNA = readSequences(0, 76, False)
print(medianString(DNA, 12))
