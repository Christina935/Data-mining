clear all;
clc;
%udata =load('cornell.mat');
% udata =load('texas.mat');
% udata =load('washington.mat');
udata =load('wisconsin.mat');
% data = udata.A;
data = udata.F;
%博客的理解
% r = 2;
% iterate = 100;
% [W,H] = NMF1(data',r,iterate);
% K=5;
% G = H';
% c = kmeans(G,K);

%TA的要求
r = 5;
iterate = 100;
[W,H] = NMF1(data,r,iterate);
WS = size(W,1);
c = zeros(WS,1);
for i=1:WS
    [big,pos] = max(W(i));
    c(i) = pos;
end

clu = udata.label;
ACC = ClusteringMeasure(c,clu);
disp(['聚类准确率ACC为',num2str(ACC(1))]);
disp(['标准化信息NMI为',num2str(ACC(2))]);
disp(['聚类纯度PUR为',num2str(ACC(3))]);