clear all;
clc;
% udata =load('cornell.mat');
% udata =load('texas.mat');
% udata =load('washington.mat');
udata =load('wisconsin.mat');
% data = udata.F;
A = udata.A; %�ڽӾ���
n = size(A,1); %������
s1 = A; %һ�����ƶ�
s2 = zeros(n,n); %�������ƶ�
for i=1:n
    for j=1:n
        s2(i,j) = sum(s1(i,:).*s1(j,:))/(sum(s1(i,:))*sum(s1(j,:)));
    end
end
S = s1+5*s2; %���ƶȾ���

K = zeros(n,1); %�ڵ�i�Ķ�
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

m = 15; %m�ǽ�ά֮���ά��
k = 5; %k����������
M = rand(n,m); %��ʼ��������
U = rand(n,m); %�ڵ�ĳ�ʼ����ʾ
H = rand(n,k); %��ʼ������ָ�����
C = rand(k,m); %�����ĳ�ʼ����ʾ
%����
alpha =0.1;
beta =0.2;
lambda =1e9 ;

I = eye(k);
X = U';
for i=1:200
    %����M
    M = M.*((S*U)./max(realmin,M*(U'*U)));
    %����U
    X = X.*((M'*S+alpha*C'*H')./max(realmin,(M'*M+alpha*(C'*C))*X));
    U = X';
    %����C
    C = C.*((H'*U)./max(realmin,C*U'*U));
    %����H
    B1H = B1*H;
    HHH = H*(H'*H);
    AH = A*H;
    UC = U*C';
    sqrtDeta = sqrt((2*beta*B1H).^2+16*lambda*HHH.*(2*beta*AH+2*alpha*UC+(4*lambda-2*alpha)*H));
    H = H.*sqrt((-2*beta*B1H+sqrtDeta)./max(realmin,(8*lambda*HHH)));    
end

label = udata.label;
preY = kmeans(U,5);
%[maxU,preY] =max(U,[],2); %�ҳ�U��ÿһ����ֵ��������
acc = classificationACC(label,preY);
disp(['����׼ȷ��ACCΪ',num2str(acc)]);

%����Ŀ�꺯��������ֵ
% first = norm(S-M*U','fro')^2;
% second = alpha*norm(H-U*C','fro')^2;
% third = -beta*trace(H'*(B1-B2)*H);
% constraint = lambda*norm(H'*H-I,'fro')^2;
% L = first + second + third + constraint;
