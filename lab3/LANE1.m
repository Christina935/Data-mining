clear all;
clc;
% udata =load('cornell.mat');
% udata =load('texas.mat');
% udata =load('washington.mat');
udata =load('wisconsin.mat');
% data = udata.F;

alpha1 = 0.3; %�ڵ�������Ϣ��Ȩ��
alpha2 = 0.5; %��ǩ��Ϣ��Ȩ��
numiter = 30; %����������
delta1 = 0.97; %���ڹ������Ա�ʾH2��������Ϣ��Ȩ��
delta2 = 1.6; %���ڹ������Ա�ʾH2�Ľڵ�������Ϣ��Ȩ��

d = 50; %Ƕ���ʾ��ά��
G = udata.A; %�ڽӾ���
A = udata.F; %���Ծ���
n = size(G,1); %������
G(1:n+1:n^2) = 1; %�öԽ���Ϊ1
label = udata.label;
labelId = unique(label); % ���
Y=[];
for i=1:length(labelId)
   Y = [Y,label==labelId(i)]; 
end
Y = Y*1;

Indices = crossvalind('Kfold',n,20); % K�۽�����֤
Group1 = find(Indices <= 16); % ѵ����
Group2 = find(Indices >= 17); % ���Լ�
%ѵ����
G1 = sparse(G(Group1,Group1)); %ѵ�����еĽڵ�����
A1 = sparse(A(Group1,:)); %ѵ�����нڵ�Ľڵ�����
Y1 = sparse(Y(Group1,:)); %ѵ�����нڵ�ı�ǩ
%���Լ�
A2 = sparse(A(Group2,:));%�������нڵ�Ľڵ�����
GC1 = sparse(G(Group1,:));%���ڹ������Ա�ʾH2
GC2 = sparse(G(Group2,:));%���ڹ������Ա�ʾH2
Y2 = sparse(Y(Group2,:));%���Լ��нڵ�ı�ǩ

H1 = LANE2(G1,A1,Y1,d,alpha1,alpha2,numiter); 
H2 = delta1*(GC2*pinv(pinv(H1)*GC1))+delta2*(A2*pinv(pinv(H1)*A1));
c1 = kmeans(H1,5);
acc = classificationACC(c1,Y1);
disp(['����׼ȷ��ACCΪ',num2str(acc)]);
c2 = kmeans(H2,5);
acc = classificationACC(c2,Y2);
disp(['���Լ�����׼ȷ��ACCΪ',num2str(acc)]);


%�޼ල
% beta1 = 43; %�ڵ�������Ϣ��Ȩ��
% beta2= 36; %����Ե�Ȩ��
% numiter = 5; %����������
% delta1 = 0.97; %���ڹ������Ա�ʾH2��������Ϣ��Ȩ��
% delta2 = 1.6; %���ڹ������Ա�ʾH2�Ľڵ�������Ϣ��Ȩ��
% 
% 
% H1 = LANE2(G1,A1,d,beta1,beta2,numiter);
% H2 = delta1*(GC2*pinv(pinv(H1)*GC1))+delta2*(A2*pinv(pinv(H1)*A1));
% 
% c1 = kmeans(H1,5);
% acc = classificationACC(c1,Y1);
% disp(['ѵ��������׼ȷ��ACCΪ',num2str(acc)]);
% c2 = kmeans(H2,5);
% acc = classificationACC(c2,Y2);
% disp(['���Լ�����׼ȷ��ACCΪ',num2str(acc)]);
