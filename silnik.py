import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--mer_length', required = True,
                    help = 'length of mers')

parser.add_argument('-d', '--mismatch', required = True,
                    help = 'maximum number of mismatches')

parser.add_argument('-a','-startline', required = True,
                    help = 'from which line start comparing')

parser.add_argument('-p','-stopline', required = True,
                    help = 'at which line stop reading')

parser.add_argument(action = 'store_true')

args = vars(parser.parse_args())