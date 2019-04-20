function [h, eer, dcf08, dcf10, dcf12] = Get_DCF_Plot_DET(tar, non)

lim = [0.0001 0.95];
Set_DET_limits(lim(1), lim(2), lim(1), lim(2));

 h = figure;
 hold on;

% EER
Set_DCF(1, 1, 0.01);
[Pmiss, Pfa, eer] = Compute_DET(tar, non);
Plot_DET(Pmiss, Pfa, 'b-');
Plot_DET(eer, eer, 'r+', 2);

% DCF08
[DCF_opt, Popt_miss, Popt_fa] = Min_DCF(Pmiss, Pfa);
Plot_DET(Popt_miss, Popt_fa, 'go', 2);
dcf08 = DCF_opt * 100;

% DCF10
Set_DCF(1, 1, 0.001);
[DCF_opt, Popt_miss, Popt_fa] = Min_DCF(Pmiss, Pfa);
Plot_DET(Popt_miss, max(Popt_fa, lim(1)), 'ro', 2);
dcf10 = DCF_opt * 1000;

% DCF12
dcf12 = (dcf08 + dcf10) / 2;


% DCF08
Set_DCF(10, 1, 0.01);
[DCF_opt, Popt_miss, Popt_fa] = Min_DCF(Pmiss, Pfa);
Plot_DET(Popt_miss, Popt_fa, 'bo', 2);
dcf08 = DCF_opt;

% DCF10
Set_DCF(1, 1, 0.001);
[DCF_opt, Popt_miss, Popt_fa] = Min_DCF(Pmiss, Pfa);
Plot_DET(Popt_miss, max(Popt_fa, lim(1)), 'ro', 2);
dcf10 = DCF_opt * 1000;


text(-0.2, 1.08, strcat('EER=', num2str(100*eer, '%10.3f'), '%'), 'BackgroundColor', [.7 .9 .7]);
text(-0.2, 0.84, strcat('DCF08=', num2str(dcf08, '%10.3f') ), 'BackgroundColor', [.7 .9 .7]);
text(-0.2, 0.60, strcat('DCF10=', num2str(dcf10, '%10.3f') ), 'BackgroundColor', [.7 .9 .7]);
text(-0.2, 0.36, strcat('DCF12=', num2str(dcf12, '%10.3f') ), 'BackgroundColor', [.7 .9 .7]);

hold off;
