import sys, argparse
p=argparse.ArgumentParser()
p.add_argument("--end", type=int, required=True)
p.add_argument("--start", type=int, default=0)
p.add_argument("--step", type=int, default=1)
args=p.parse_args(sys.argv[1:])
for i in range(args.start, args.end+1, args.step):
    print(f"This is a hello sentence with {i}")