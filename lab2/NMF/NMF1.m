% v = WH W：基向量矩阵 H：
% 输入数据V：n个 m 维  
% r:降维后的维度
% iterate:迭代次数
function [W,H] = NMF1(V,r,iterate)
    [n,m] = size(V);
    %n = size(V,1); %获取矩阵v的行数
    %m = size(V,2); %获取矩阵v的维数
    %随机产生非负矩阵W与H
    W = abs(rand(n,r));
    H = abs(rand(r,m));
    
    for it = 1:iterate
        %更新H
        H = H.*(W'*V)./(W'*W*H+eps);
        %更新W
        W = W.*(V*H')./(W*H*H'+eps);    
    end  
end