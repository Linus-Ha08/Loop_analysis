function chanlocs = pop_chanedit(chanlocs, varargin)
% Stand-in for EEGLAB's pop_chanedit, used only by organize_neurone.m.
%
% The NeurOne plugin (readneurone.m, line 184) calls pop_chanedit(chanlocs) with
% just the channel labels, which opens EEGLAB's channel editor for every block.
% This returns the labels unchanged instead -- the same as pressing Cancel there.
% organize_neurone.m puts this folder on the path only around pop_readneurone.
if nargin > 1 || ~isstruct(chanlocs) || isfield(chanlocs, 'data')
    error(['This is organize_neurone''s pop_chanedit stand-in, left on the path by an ' ...
           'interrupted run. Remove it with: rmpath(''%s'')'], fileparts(mfilename('fullpath')));
end
end
