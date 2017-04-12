import argparse

from simplevec_to_entity import get_entities
from simplevec_link import observe_links

parser = argparse.ArgumentParser(description="Calculate all meanfield vectors")
parser.add_argument('short')
parser.add_argument('long')
parser.add_argument('proc', type=int, default=4)
args = parser.parse_args()

parts = [x.replace('~','-').replace('_','.') for x in args.long.split('-')]
scale = float(parts[0])
C = int(parts[1])
Z = float(parts[2])
alpha = float(parts[3])
wrong_weight = float(parts[4])

full = args.short + '-' + args.long

get_entities('frequency',
             scale=scale,
             C=C,
             Z=Z,
             alpha=alpha,
             wrong_weight=wrong_weight,
             name=args.short,
             mean_field_kwargs={"max_iter":500},
             input_dir='simplevec_all',
             output_dir='meanfield_all',
             pred_list=False)

observe_links(full, processes=args.proc)
