import argparse

parser = argparse.ArgumentParser('MotifSearcher', 'MotifSearcher.py -k Length -d Mismatches -lstart Startline -lstop Stopline [-all readAll]')
parser.add_argument('-k', '--Mer_length', required=True, help='Length of mers')
parser.add_argument('-d', '--Mismatch', required=True, help='Maximum number of mismatches')
parser.add_argument('-lstart', '--Startline', required=True, help='From which line start comparing')
parser.add_argument('-lstop', '--Stopline', required=True, help='At which line stop reading')
parser.add_argument('-all', '--Read_all', action='store_true', required=False, help='Whether to read entirety of all sequences (Ignores lstart and lstop)')
args = vars(parser.parse_args())
