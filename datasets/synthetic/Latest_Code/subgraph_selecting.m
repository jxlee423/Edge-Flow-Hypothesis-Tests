function subgraph_edges = subgraph_selecting(output_file, edge_to_endpoints, subgraph_mode)
    switch subgraph_mode
        case '0001001101011110'
            subgraph_edges = subgraph_searching_1(output_file, edge_to_endpoints);
        case 'subgraph_mode2'
            subgraph_edges = subgraph_searching_2(output_file, edge_to_endpoints);
        case 'subgraph_mode3'
            subgraph_edges = subgraph_searching_3(output_file, edge_to_endpoints);
        otherwise
            error('Unknown subgraph mode: %s', subgraph_mode);
    end
end

%%subgraph.mode 0001001101011110
function subgraph_edges = subgraph_searching_1(output_file, edge_to_endpoints)
% Parse the output file, identify subgraph edges, and match edge positions in edge_to_endpoints
subgraph_edges = [];
used_points = [];  % List to track points already used in subgraphs
fileID = fopen(output_file, 'r');
if fileID == -1
    error('Cannot open file: %s', output_file);
end

tline = fgetl(fileID);
while ischar(tline)
    parts = strsplit(tline, ':'); % Split the line into subgraph code and points list
    if length(parts) == 2 && strcmp(strtrim(parts{1}), '0001001101011110')
        % Randomly decide whether to extract this target (80% chance)
        if rand() <= 0.8  % 80% probability
            % Remove whitespaces after the subgraph code and then split out point information
            points_str = strsplit(strtrim(parts{2}), ' ');
            points = str2double(points_str);

            % Check if any point in the subgraph has already been used
            if any(ismember(points, used_points))
               tline = fgetl(fileID); % Read the next line and continue
               continue;
            end

            % If no points are shared, match subgraph edges
            matched_edges = match_subgraph_edges_1(edge_to_endpoints, points); % Match edges
            subgraph_edges = [subgraph_edges; matched_edges]; % Add matched edge indices to the subgraph edges matrix
            used_points = [used_points, points]; % Add each point from this subgraph to the used points list
        end
    end
    tline = fgetl(fileID); % Read the next line
end
fclose(fileID);
end

function matched_edges = match_subgraph_edges_1(edge_to_endpoints, points)
    % Match the edges of the subgraph based on the position of points

    % Calculate the number of connections each point has with the other three points in edge_to_endpoints
    count_points = zeros(1, 4);
    for i = 1:4
        for j = 1:4
            if i ~= j
                count_points(i) = count_points(i) + sum(ismember(edge_to_endpoints, [points(i), points(j)], 'rows') | ...
                                                      ismember(edge_to_endpoints, [points(j), points(i)], 'rows'));
            end
        end
    end

    % Find the point with a count of 3 as point1
    point1_idx = find(count_points == 3, 1); % Ensure only the first satisfying condition is taken
    if isempty(point1_idx)
        return; 
    end
    point1 = points(point1_idx);

    % Find the point with a count of 1 as point2
    point2_idx = find(count_points == 1, 1); % Ensure only the first satisfying condition is taken
    if isempty(point2_idx)
        return; 
    end
    point2 = points(point2_idx);

    % The remaining two points are point3 and point4
    remaining_points = setdiff(points, [point1, point2]);
    if length(remaining_points) ~= 2
        return; 
    end
    point3 = remaining_points(1);
    point4 = remaining_points(2);

    % Match the corresponding subgraph edges
    edge1 = find_edge_index(edge_to_endpoints, point1, point2);
    edge2 = find_edge_index(edge_to_endpoints, point3, point4);
    edge3 = find_edge_index(edge_to_endpoints, point1, point3);
    edge4 = find_edge_index(edge_to_endpoints, point1, point4);

    % Ensure the returned edge indices have consistent dimensions
    if ~isempty(edge1) && ~isempty(edge2) && ~isempty(edge3) && ~isempty(edge4)
        matched_edges = [edge1, edge2, edge3, edge4];
    else
        warning('Failed to match edges for points: %s', mat2str(points));
    end
end


%% the shared function to support subgraph_selecting 
function edge_index = find_edge_index(edge_to_endpoints, point_a, point_b)
    % Find the row index of the edge formed by point_a and point_b in edge_to_endpoints
    % Input:
    %   edge_to_endpoints - Endpoint information of edges
    %   point_a, point_b - Two points forming an edge
    % Output:
    %   edge_index - Row index of the edge in edge_to_endpoints

    % Ensure point_a and point_b as a 1x2 row vector
    point_pair = [point_a, point_b];
    
    % Check [point_a, point_b]
    edge_index = find(ismember(edge_to_endpoints, point_pair, 'rows'));
    
    % If not found, check [point_b, point_a]
    if isempty(edge_index)
        point_pair = [point_b, point_a];
        edge_index = find(ismember(edge_to_endpoints, point_pair, 'rows'));
    end
end

