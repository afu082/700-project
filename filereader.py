import os

# file reader for .ies files

# read in .ies file
def read_ies(filename):
    # open file
    with open(filename, "r") as f:
        # read file
        lines = f.readlines()

        
        # return all lines after the first line beginning with a number
        for i, line in enumerate(lines):
            if line[0].isdigit():
                return lines[i:]
    
    # Return an empty list if "TILT=" is not found
    return []

# main function
if __name__ == "__main__":
    # Change the input to the path of your .ies file
    input_file_path = "./TestFile1/test.ies"
    
    # Extract the filename without extension from the input file path
    input_filename = os.path.splitext(os.path.basename(input_file_path))[0]
    
    # Get the extracted lines
    extracted_lines = read_ies(input_file_path)
    
    # Create the "output" directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the output file path with the name "inputname_output.ies"
    output_file_path = os.path.join(output_dir, f"{input_filename}_output.ies")
    
    # Write the extracted lines to the output file
    with open(output_file_path, "w") as f:
        f.writelines(extracted_lines)
