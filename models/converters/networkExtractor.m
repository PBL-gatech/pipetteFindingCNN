% Load the DAGNetwork
load('regressionNET-11-Mar-2020.mat'); % Assuming the network is in the variable `net`

% Create a folder for saving exported data
outputFolder = 'exported_network';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Iterate over layers in the network
layers = net.Layers;
connections = net.Connections;

% Metadata to store structure information
layer_metadata = {};
connection_metadata = {};

% Extract layer information
for i = 1:numel(layers)
    layer = layers(i);
    
    % Save weights to CSV if present
    if isprop(layer, 'Weights')
        weights = layer.Weights;
        csvFileName = fullfile(outputFolder, ['weights_layer_' num2str(i) '.csv']);
        csvwrite(csvFileName, weights);
    end
    
    % Save biases to CSV if present
    if isprop(layer, 'Bias')
        bias = layer.Bias;
        csvFileName = fullfile(outputFolder, ['bias_layer_' num2str(i) '.csv']);
        csvwrite(csvFileName, bias);
    end
    
    % Store metadata for layer
    layer_metadata{end+1} = sprintf('Layer %d: %s - Name: %s', i, class(layer), layer.Name);
end

% Extract connection information
for i = 1:height(connections)
    connection_metadata{end+1} = sprintf('Source: %s -> Destination: %s', ...
                                         connections.Source{i}, connections.Destination{i});
end

% Save metadata about the layers
metadataFileName = fullfile(outputFolder, 'layer_metadata.txt');
fid = fopen(metadataFileName, 'w');
for i = 1:numel(layer_metadata)
    fprintf(fid, '%s\n', layer_metadata{i});
end
fclose(fid);

% Save connection metadata
connectionFileName = fullfile(outputFolder, 'connections_metadata.txt');
fid = fopen(connectionFileName, 'w');
for i = 1:numel(connection_metadata)
    fprintf(fid, '%s\n', connection_metadata{i});
end
fclose(fid);

% Save network input and output names
inputOutputFileName = fullfile(outputFolder, 'input_output_metadata.txt');
fid = fopen(inputOutputFileName, 'w');
fprintf(fid, 'Network Input Names: %s\n', strjoin(net.InputNames, ', '));
fprintf(fid, 'Network Output Names: %s\n', strjoin(net.OutputNames, ', '));
fclose(fid);
