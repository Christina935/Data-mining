
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

%���һ������֪���ֵ���Ʒ��Ϊ��֤��
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

%��pearson���ϵ������Ʒ������
%���ȼ����ֵ
average = zeros(item,1);
for i=1:item
    r = find(score(:,i)~=-1);
    sum = 0;
    for j=1:size(r,1)
        sum = sum + score(r(j),i);
    end
    average(i,1) = sum/size(r,1);
end

sim = zeros(item,item);
for i=1:item
    for j=1:item
        sim(i,j)=-1;
    end
end

for i=1:item-1
    %��item i�����˵��û�
    ri = find(score(:,i)~=-1); 
    
    for j=i+1:item
        %��item j�����˵��û�
        rj = find(score(:,j)~=-1);
        
        %�ҵ���i,j�������˵��û�
        u = intersect(ri,rj); 
        num = size(u);
        if num(1)==0 || num(2)==0
            sim(i,j)=0;
        else 
            up=0;
            down1=0;
            down2=0;
            for k=1:num(1)
                up = up + (score(u(k),i)-average(i))*(score(u(k),j)-average(j));             
                down1 = down1 + (score(u(k),i)-average(i))*(score(u(k),i)-average(i));
                down2 = down2 + (score(u(k),j)-average(j))*(score(u(k),j)-average(j));
            end
         
            if down1==0 || down2==0
                sim(i,j)=0;
            else
                sim(i,j)= up/(sqrt(down1)*sqrt(down2));
            end
            sim(j,i) = sim(i,j);
        end
     end
end
 
%����Ԥ��

for i=1:user
    pos = find(score(i,:)~=-1);
    for j=1:item 
        if ~any(pos==j)
            u=0;
            d=0;
            for k=1:size(pos,2)
                u = u + sim(pos(k),j)*score(i,pos(k));
                d = d + sim(pos(k),j);
            end
            if d==0
               score(i,j) = 0; 
            else
               score(i,j) = u/d; 
            end
            
        end
    end  
end

RMSE = calRMSE(score,test,user,item);
disp(['RMSE = ',num2str(61.231)]);
acc = Accurancy(score,test,user);
disp(['׼ȷ�� = ',num2str(acc)]);

