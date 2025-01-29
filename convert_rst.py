import os
import subprocess

def convert_rst_to_txt(input_dir):
    # Create output directory if it doesn't exist
    output_dir = "converted_txt"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Find all .rst files recursively
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.rst'):
                input_path = os.path.join(root, file)
                # Create equivalent output path with .txt extension
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.txt')
                
                # Create output subdirectories if needed
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Run pandoc conversion
                try:
                    subprocess.run([
                        'pandoc',
                        '-f', 'rst',  # Input format is RST
                        '-t', 'plain', # Output format is plain text
                        '-o', output_path, # Output file
                        input_path # Input file
                    ], check=True)
                    print(f"Converted {input_path} -> {output_path}")
                except subprocess.CalledProcessError as e:
                    print(f"Error converting {input_path}: {e}")

# Convert all RST files in collected_docs directory
convert_rst_to_txt('collected_docs')