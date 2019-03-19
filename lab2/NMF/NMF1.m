% v = WH W������������ H��
% ��������V��n�� m ά  
% r:��ά���ά��
% iterate:��������
function [W,H] = NMF1(V,r,iterate)
    [n,m] = size(V);
    %n = size(V,1); %��ȡ����v������
    %m = size(V,2); %��ȡ����v��ά��
    %��������Ǹ�����W��H
    W = abs(rand(n,r));
    H = abs(rand(r,m));
    
    for it = 1:iterate
        %����H
        H = H.*(W'*V)./(W'*W*H+eps);
        %����W
        W = W.*(V*H')./(W*H*H'+eps);    
    end  
end