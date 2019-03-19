clear all;
clc;
% udata =load('cornell.mat');
% udata =load('texas.mat');
% udata =load('washington.mat');
udata =load('wisconsin.mat');
% data = udata.F;
A = udata.A; %邻接矩阵
n = size(A,1); %样本数
s1 = A; %一阶相似度
s2 = zeros(n,n); %二阶相似度
for i=1:n
    for j=1:n
        s2(i,j) = sum(s1(i,:).*s1(j,:))/(sum(s1(i,:))*sum(s1(j,:)));
    end
end
S = s1+5*s2; %相似度矩阵

K = zeros(n,1); %节点i的度
for i=1:n
    K(i)=sum(A(i,:));
end

B = zeros(n,n);
B1 = zeros(n,n);
for i=1:n
    for j=1:n
        B(i,j)=A(i,j)-K(i)*K(j)/sum(K);
        B1(i,j) = K(i)*K(j)/sum(K);
    end
end

m = 15; %m是降维之后的维数
k = 5; %k是社区个数
M = rand(n,m); %初始化基矩阵
U = rand(n,m); %节点的初始化表示
H = rand(n,k); %初始化社区指标矩阵
C = rand(k,m); %社区的初始化表示
%参数
alpha =0.1;
beta =0.2;
lambda =1e9 ;

I = eye(k);
X = U';
for i=1:200
    %更新M
    M = M.*((S*U)./max(realmin,M*(U'*U)));
    %更新U
    X = X.*((M'*S+alpha*C'*H')./max(realmin,(M'*M+alpha*(C'*C))*X));
    U = X';
    %更新C
    C = C.*((H'*U)./max(realmin,C*U'*U));
    %更新H
    B1H = B1*H;
    HHH = H*(H'*H);
    AH = A*H;
    UC = U*C';
    sqrtDeta = sqrt((2*beta*B1H).^2+16*lambda*HHH.*(2*beta*AH+2*alpha*UC+(4*lambda-2*alpha)*H));
    H = H.*sqrt((-2*beta*B1H+sqrtDeta)./max(realmin,(8*lambda*HHH)));    
end

label = udata.label;
preY = kmeans(U,5);
%[maxU,preY] =max(U,[],2); %找出U中每一行中值最大的列数
acc = classificationACC(label,preY);
disp(['聚类准确率ACC为',num2str(acc)]);

%计算目标函数的最终值
% first = norm(S-M*U','fro')^2;
% second = alpha*norm(H-U*C','fro')^2;
% third = -beta*trace(H'*(B1-B2)*H);
% constraint = lambda*norm(H'*H-I,'fro')^2;
% L = first + second + third + constraint;
