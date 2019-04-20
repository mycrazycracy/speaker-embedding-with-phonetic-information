function get_eer(target, imposter, output)

addpath('./DETware_v2.1/');

tar = load(target);
non = load(imposter);  

% calculate an effective prior from target prior, Cmiss, and Cfa
[h, eer, dcf08, dcf10, dcf12] = Get_DCF_Plot_DET(tar, non);

fid = fopen(output, 'a');
fprintf(fid, 'eer: %5.4f%%; mindcf08: %5.4f%%; mindcf10: %5.4f%%; mindcf12: %5.4f%%\n',eer*100, dcf08, dcf10, dcf12);
fclose(fid);

end
