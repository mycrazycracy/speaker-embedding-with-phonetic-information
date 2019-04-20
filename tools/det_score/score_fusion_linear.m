function [best_eer, best_coef] = score_fusion_linear(score1, score2, trial, key, score_out, coef)
    addpath('./DETware_v2.1/');
    mode = 'fusion';
    if nargin < 6
        mode = 'find_coef';
    end
    
    s1 = textread(score1, '%f');
    s2 = textread(score2, '%f');
    
    modelSeg = containers.Map;
    
    fid = fopen(key, 'r');
    while 1
        string = textscan(fid, '%s', 2, 'delimiter', '|');
        if isempty(string{1})
            break;
        end        
        m = string{1}{1};
        s = string{1}{2};
        if modelSeg.isKey(m) == 0
            modelSeg(m) = containers.Map;
        end
        segPool = modelSeg(m);
        segPool(s) = [];
        modelSeg(m) = segPool;
    end
    fclose(fid);
    
    tarIdx = [];
    nonIdx = [];
    
    index = 1;
    fid = fopen(trial, 'r');
    while 1
        string = textscan(fid, '%s', 2, 'delimiter', '|');
        if isempty(string{1})
            break;
        end        
        m = string{1}{1};
        s = string{1}{2};
        if modelSeg.isKey(m)
            segPool = modelSeg(m);
            if segPool.isKey(s)
                tarIdx = [tarIdx; index];
            else
                nonIdx = [nonIdx; index];
            end
        else
            nonIdx = [nonIdx; index];
        end
        index = index + 1;
    end
    fclose(fid);
    
    Set_DCF(1, 1, 0.01);
    
    if strcmp(mode, 'find_coef')
        best_coef = -1;
        best_eer = 100;
        for c=0:0.1:1
            sc = c * s1 + (1-c) * s2;
            target = sc(tarIdx);
            nontarget = sc(nonIdx);
            [~, ~, eer] = Compute_DET(target, nontarget); 
            fprintf('%f %f\n',c, eer);
            if eer < best_eer
                best_eer = eer;
                best_coef = c;
            end
        end
    else
        if coef < 0 || coef > 1
            fprint('Error: coefficient is not valid: %f\n', coef);
            return;
        end
        best_coef = coef;
    end
    
    target = s1(tarIdx);
    nontarget = s1(nonIdx);
    [~, ~, eer1] = Compute_DET(target, nontarget);
    fprintf('Score1 EER: %f\n', eer1);
    
    target = s2(tarIdx);
    nontarget = s2(nonIdx);
    [~, ~, eer2] = Compute_DET(target, nontarget);
    fprintf('Score2 EER: %f\n', eer2);
    
    ss = best_coef * s1 + (1-best_coef) * s2;
    target = ss(tarIdx);
    nontarget = ss(nonIdx);
    [~, ~, best_eer] = Compute_DET(target, nontarget);  
    
    fid = fopen(score_out, 'w');
    fprintf(fid, '%f\n', ss);
    fclose(fid);
end
