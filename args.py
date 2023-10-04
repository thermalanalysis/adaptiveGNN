
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--train_newgnn', action='store_true')
parser.add_argument('--train_defectnode', action='store_true')
parser.add_argument('--generate_new_sample', action='store_true')
parser.add_argument('--generate_sample_badedge', action='store_true')
parser.add_argument('--numofepoch',type=int,default=3000 )
parser.add_argument('--subblock_size_xy',type=int,default=10)
parser.add_argument('--numofsubblock_xy',type=int,default=5)
parser.add_argument('--k1',type=int,default=100)
parser.add_argument('--k2',type=int,default=10)
parser.add_argument('--inilr',type=float,required=True )

