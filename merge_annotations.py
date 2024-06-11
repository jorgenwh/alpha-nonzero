import os
def combine_files(input_directory, output_file):
    dataPoints = 0
    if not os.path.isdir(input_directory):
        print(f"The directory {input_directory} does not exist.")
        return    
    print(f"Combining files from directory: {input_directory}")
    
    with open(output_file, 'wb') as outfile:
        for filename in os.listdir(input_directory):
            file_path = os.path.join(input_directory, filename)
            
            if os.path.isfile(file_path):
                print(f"Processing file: {file_path}")
                
                with open(file_path, 'rb') as infile:
                    for line in infile:
                        outfile.write(line)
                        dataPoints += 1
                        print(f"lineCount: {dataPoints}", end="\r", flush=True)
    
    print(f"\nProcessed {dataPoints} data points.")
    print(f"All files have been combined into {output_file}")
# Define the input directory and output file
input_directory = 'training/annotations'
output_file = 'combined_annotations.fen'
combine_files(input_directory, output_file)
