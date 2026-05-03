% Manuel Lozano García, 2016
%
% This function extracts signals from a data vector exported in LabChart
%
% INPUT PARAMETER:
%   pth   -> Absolute path of a .mat file that contains the following fields:
%            data
%            datastart
%            dataend
%            samplerate
%            titles
%            unittext
%            unittextmap
% OUTPUT PARAMETER:
%   sdata -> Structure that contains the following fields:
%            signals     -> n x m cell array with n number of channels and m number of blocks
%            nchannels   -> Number of channels
%            nblocks     -> Number of blocks
%            samplerate  -> n x m matrix that contains samplerates for each channel and block
%            titles      -> char vector of size n that contains channel names
%            unittext    -> char vector that contains channel units
%            unittextmap -> n x m matrix that contains indices that refer to the content of the unittext vector

function sdata = read_signals(pth)
    sdata = load(pth);
    sdata.nchannels = size(sdata.datastart,1); % number of channels
    sdata.nblocks = size(sdata.datastart,2); % number of blocks
    sdata.signals = cell(sdata.nchannels,sdata.nblocks);
    for i=1:sdata.nchannels
        for j=1:sdata.nblocks
            sdata.signals{i,j} = sdata.data(sdata.datastart(i,j):sdata.dataend(i,j));
        end
    end
    sdata = rmfield(sdata,{'data','datastart','dataend'});
end