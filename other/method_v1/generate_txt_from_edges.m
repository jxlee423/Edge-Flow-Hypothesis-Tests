function generate_txt_from_edges(edge_data, input_file)
    fileID = fopen(input_file, 'w');
    
    if fileID == -1
        error(['Unable to create file: ', input_file]); 
    end
    
    for i = 1:size(edge_data, 1)
        fprintf(fileID, '%d %d\n', edge_data(i, 1), edge_data(i, 2));
    end
    fclose(fileID);

end