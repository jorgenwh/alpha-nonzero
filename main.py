import os
import argparse

if __name__ == '__main__':
    NUM_BATCHES = 15
    INPUT_SIZE = 75_000_000

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '-num_batches', type=int, default=NUM_BATCHES, help='Number of batches')
    parser.add_argument('-s', '-input_size', type=int, default=INPUT_SIZE, help='Input size')
    parser.add_argument('-i', '-input path', type=str, help='Input path', required=True)
    parser.add_argument('-o', '-output_dir', type=str, default='batches', help='Output directory')
    args = parser.parse_args()
    
    num_batches = args.n
    input_size = args.s
    input_path = args.i
    output_dir = args.o
    lines_per_batch = input_size // num_batches

    assert os.path.exists(input_path), f'Input path {input_path} does not exist'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    print(f"Batching {input_path} into {num_batches} batches of size {input_size}...")

    with open(input_path, 'r') as inFile:
        current_batch = 1
        current_line = 0
        file_path = os.path.join(output_dir, f'batch_{current_batch}.fen')
        outFile = open(file_path, 'w')
        for line in inFile:
            if current_line != 0 and current_line % lines_per_batch == 0:
                outFile.close()
                current_batch += 1
                file_path = os.path.join(output_dir, f'batch_{current_batch}.fen')
                outFile = open(file_path, 'w')
            outFile.write(line)
            current_line += 1
        if outFile:
            outFile.close()
