import h5py
from .gen_app import app
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Run the app")
    parser.add_argument('--port', type=int, help='The port to run the app on', default=8823)
    parser.add_argument('--num_workers', type=int, help='Number of workers to extract xlsx in parallel', default=0)
    args = parser.parse_args()
    num_workers = args.num_workers if args.num_workers > 0 else os.cpu_count()-1
    app.run(host="0.0.0.0", port=args.port, debug=False) # jupyter_mode="external", 



if __name__=="__main__":
    main()


