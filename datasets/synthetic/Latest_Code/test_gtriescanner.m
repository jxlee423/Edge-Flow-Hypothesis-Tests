function run_external_program(input_file, output_file_base)
    gtrieScanner_mex(input_file, output_file_base);
end

input_file = 'G:\2024 Summer\EFHT\Synthetic Data Sets\example_graph.txt';
output_file_base = 'G:\2024 Summer\EFHT\Synthetic Data Sets\output_base';
run_external_program(input_file, output_file_base);