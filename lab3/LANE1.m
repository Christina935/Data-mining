clear all;
clc;
% udata =load('cornell.mat');
% udata =load('texas.mat');
% udata =load('washington.mat');
udata =load('wisconsin.mat');
% data = udata.F;

alpha1 = 0.3; %节点属性信息的权重
alpha2 = 0.5; %标签信息的权重
numiter = 30; %最大迭代次数
delta1 = 0.97; %用于构建测试表示H2的网络信息的权重
delta2 = 1.6; %用于构建测试表示H2的节点属性信息的权重

d = 50; %嵌入表示的维度
G = udata.A; %邻接矩阵
A = udata.F; %属性矩阵
n = size(G,1); %样本数
G(1:n+1:n^2) = 1; %让对角线为1
label = udata.label;
labelId = unique(label); % 类别
Y=[];
for i=1:length(labelId)
   Y = [Y,label==labelId(i)]; 
end
Y = Y*1;

Indices = crossvalind('Kfold',n,20); % K折交叉验证
Group1 = find(Indices <= 16); % 训练集
Group2 = find(Indices >= 17); % 测试集
%训练集
G1 = sparse(G(Group1,Group1)); %训练组中的节点网络
A1 = sparse(A(Group1,:)); %训练组中节点的节点属性
Y1 = sparse(Y(Group1,:)); %训练组中节点的标签
%测试集
A2 = sparse(A(Group2,:));%测试组中节点的节点属性
GC1 = sparse(G(Group1,:));%用于构建测试表示H2
GC2 = sparse(G(Group2,:));%用于构建测试表示H2
Y2 = sparse(Y(Group2,:));%测试集中节点的标签

H1 = LANE2(G1,A1,Y1,d,alpha1,alpha2,numiter); 
H2 = delta1*(GC2*pinv(pinv(H1)*GC1))+delta2*(A2*pinv(pinv(H1)*A1));
c1 = kmeans(H1,5);
acc = classificationACC(c1,Y1);
disp(['聚类准确率ACC为',num2str(acc)]);
c2 = kmeans(H2,5);
acc = classificationACC(c2,Y2);
disp(['测试集聚类准确率ACC为',num2str(acc)]);


%无监督
% beta1 = 43; %节点属性信息的权重
% beta2= 36; %相关性的权重
% numiter = 5; %最大迭代次数
% delta1 = 0.97; %用于构建测试表示H2的网络信息的权重
% delta2 = 1.6; %用于构建测试表示H2的节点属性信息的权重
% 
% 
% H1 = LANE2(G1,A1,d,beta1,beta2,numiter);
% H2 = delta1*(GC2*pinv(pinv(H1)*GC1))+delta2*(A2*pinv(pinv(H1)*A1));
% 
% c1 = kmeans(H1,5);
% acc = classificationACC(c1,Y1);
% disp(['训练集聚类准确率ACC为',num2str(acc)]);
% c2 = kmeans(H2,5);
% acc = classificationACC(c2,Y2);
% disp(['测试集聚类准确率ACC为',num2str(acc)]);
