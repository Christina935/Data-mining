clc;
data = load('train_small.txt');
 
% data = load('u.data');
% data = data(:,1:3);

% [d1,d2,d3,d4,d5,d6,d7] = textread('ratings.dat','%n%n%n%n%n%n%n','delimiter','::','headerlines',1);
% data = [d1 d3 d5];

[dif1,r1] = unique(data(:,1),'rows');%�ҳ��ж���user
user = size(dif1,1);
[dif2,r2] = unique(data(:,2),'rows');%�ҳ��ж���item
item = size(dif2,1);

score = zeros(user,item);
for i=1:user
    for j=1:item
        score(i,j)=-1;
    end
end


for i=1:size(data,1)
    score(data(i,1),data(i,2)) = data(i,3);
end


%���һ������֪���ֵ���Ʒ��Ϊ���Լ�
test = zeros(user,item);
for i=1:user
    for j=1:item
        test(i,j)=-1;
    end
end

for i=1:user
    nor = find(score(i,:)~=-1);
    sn = size(nor,2);
    if sn<4
        break;
    else 
        sn = floor(sn/4);
    end
    for j=1:sn
        test(i,nor(j)) = score(i,nor(j));
        score(i,nor(j)) = -1;
    end
end

k = 100;
%�����ʼ��P,Q
P = abs(rand(user,k))*0.1;
Q = abs(rand(k,item));
iterator = 10; %��������
itnum = 1;
a = 0.0001;

before = 100; 
while itnum <= iterator
    %������ʧ����
    [e,alle] = calE(score,P,Q,user,item,k); 
    if alle < before
        before = alle;
    end 
    %����P,Q
    [P,Q]= updatePQ(P,Q,a,e,user,item,k);
    itnum = itnum+1;
end
train = P*Q;
RMSE = calRMSE(train,test,user,item);
disp(['RMSE = ',num2str(RMSE)]);
acc = Accurancy(train,test,user);
disp(['׼ȷ�� = ',num2str(acc)]);

function [E,allE] = calE(R,P,Q,n,m,nk)
    allE = 0;
    E = zeros(n,m); 
    for i=1:n
        for j=1:m
            if R(i,j)~=-1
                sumk=0;
                for k=1:nk
                    sumk = sumk+P(i,k)*Q(k,j);
                end
                E(i,j) = (R(i,j) - sumk)*(R(i,j) - sumk);
                allE = allE + E(i,j);
            else
                E(i,j) = 0;
            end
        end      
    end 
    E = real(E);
end

function [P,Q] = updatePQ(P,Q,a,E,n,m,nk)
    for i=1:n
        for j=1:m
           for k=1:nk
               p = P(i,k);
               q = Q(k,j);
                P(i,k) = P(i,k)+2*a*E(i,j)*q;
                Q(k,j) = Q(k,j)+2*a*E(i,j)*p;
           end 
        end     
    end
end

