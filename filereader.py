import os

# file reader for .ies files




# read in .ies file
def read_ies(filename):
    # open file
    with open(filename, "r") as f:
        # read file
        lines = f.readlines()

        
        # return all lines after the first line beginning with TILT
        index = lines.index(next(line for line in lines if line.startswith("TILT=")))
        a = lines[index:]
        return a



def splitInfo(lines):
    info = {}

    # Parse tilt information
    info["tilt_val"] = lines[0][5:-1]
    lines.pop(0)
    if info["tilt_val"] == "INCLUDE":
        info["lamp_to_luminaire"] = lines.pop(0)
        info["number_tilt_angles"] = lines.pop(0)
        info["angles"] = lines.pop(0)
        info["multiplying_factors"] = lines.pop(0)

    # Parse first line
    first_line = lines.pop(0).split()
    info["num_lamps"] = first_line[0]
    info["lumen_per_lamp"] = first_line[1]
    info["candela_multiplier"] = first_line[2]
    info["num_vertical_angles"] = first_line[3]
    info["num_horizontal_angles"] = first_line[4]
    info["photometric_type"] = first_line[5]
    info["units_type"] = first_line[6]
    info["width"] = first_line[7]
    info["length"] = first_line[8]
    info["height"] = first_line[9]

    # Parse second line
    second_line = lines.pop(0).split()
    info["ballast_factor"] = second_line[0]
    info["file_generation_type"] = second_line[1]
    info["input_watts"] = second_line[2]

    # Parse vertical and horizontal angles
    info["vertical_angles"] = lines.pop(0).split()
    info["horizontal_angles"] = lines.pop(0).split()

    # Parse candela values
    candela_values = []
    for line in lines:
        candela_values.append(line.split())
    info["candela_values"] = candela_values

    return info


def visualize_info(info):
    for key, value in info.items():
        if isinstance(value, list):
            value_str = ', '.join(str(v) for v in value)
        else:
            value_str = str(value)
        print(f"{key}: {value_str}")


def writeOutput(output_file_path, extracted_lines, info):
    with open(output_file_path, "w") as f:
        # Write back the tile
        f.write(f"TILT={info['tilt_val']}\n")


        # Write the parsed dictionary values
        if "lamp_to_luminaire" in info:
            f.write(f"{info['lamp_to_luminaire']}\n")
            f.write(f"{info['number_tilt_angles']}\n")
            f.write(f"{info['angles']}\n")
            f.write(f"{info['multiplying_factors']}\n")

        f.write(f"{info['num_lamps']} {info['lumen_per_lamp']} {info['candela_multiplier']} "
                f"{info['num_vertical_angles']} {info['num_horizontal_angles']} "
                f"{info['photometric_type']} {info['units_type']} "
                f"{info['width']} {info['length']} {info['height']}\n")

        f.write(f"{info['ballast_factor']} {info['file_generation_type']} {info['input_watts']}\n")

        f.write(' '.join(info['vertical_angles']) + '\n')
        f.write(' '.join(info['horizontal_angles']) + '\n')

        for candela_line in info['candela_values']:
            f.write(' '.join(candela_line) + '\n')






# main function
if __name__ == "__main__":
    # Change the input to the path of your .ies file
    input_file_path = "./TestFile1/test.ies"
    
    # Extract the filename without extension from the input file path
    input_filename = os.path.splitext(os.path.basename(input_file_path))[0]
    
    # Get the extracted lines
    extracted_lines = read_ies(input_file_path)
    
    # split data and write to dictionary
    split_info = splitInfo(extracted_lines)
    
    # additinal function to print dictionary
    visualize_info(split_info)
    
    # Create the "output" directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the output file path with the name "inputname_output.ies"
    output_file_path = os.path.join(output_dir, f"{input_filename}_output.ies")
    
    # Write the extracted lines to the output file
    writeOutput(output_file_path, extracted_lines, split_info)
        
        
        
        
