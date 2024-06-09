import os

def combine_fen_files(start_index, end_index, output_file, input_directory):
    with open(output_file, 'w') as outfile:
        for i in range(start_index, end_index + 1):
            input_file = os.path.join(input_directory, f'fen_batch_{i}.fen')
            if os.path.exists(input_file):
                with open(input_file, 'r') as infile:
                    outfile.write(infile.read())
            else:
                print(f"Warning: {input_file} does not exist and will be skipped.")

if __name__ == "__main__":
    start_index = 50000
    end_index = 50100 
    output_file = 'combined_fens.fen'
    input_directory = 'training/training_data'
    combine_fen_files(start_index, end_index, output_file, input_directory)
    print(f"Files from fen_batch_{start_index}.fen to fen_batch_{end_index}.fen have been combined into {output_file}")
