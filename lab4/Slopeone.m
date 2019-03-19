clc;
data = load('train_small.txt');

% data = load('u.data');
% data = data(:,1:3);

% [d1,d2,d3,d4,d5,d6,d7] = textread('ratings.dat','%n%n%n%n%n%n%n','delimiter','::','headerlines',1);
% data = [d1 d3 d5];

[dif1,r1] = unique(data(:,1),'rows');%找出有多少user
user = size(dif1,1);
[dif2,r2] = unique(data(:,2),'rows');%找出有多少item
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

%抽出一部分已知评分的商品作为测试集
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


%计算商品之间的评分偏差
dev = zeros(item,item);

for i=1:item
    for j=1:item
        dev(i,j)=-1;
    end
end

for i=1:item-1
    %给item i评分了的用户
    ri = find(score(:,i)~=-1); 
    
    for j=i+1:item
        %给item j评分了的用户
        rj = find(score(:,j)~=-1);
        
        %找到给i,j都评分了的用户
        u = intersect(ri,rj); 
        num = size(u);
        if num(1)==0 || num(2)==0
            dev(i,j)=0;
        else
            up=0;
            down = num(1);
            for k=1:num(1)
                up = up + score(u(k),i)-score(u(k),j);
            end
            dev(i,j) = up/down;
            dev(j,i) = dev(i,j);
        end
    end
end

for i=1:user
    pos = find(score(i,:)~=-1);
    pre = find(score(i,:)==-1);
    for j=1:size(pre,2)
        an = 0;
        for k=1:size(pos,2)
            an = an + dev(pre(j),pos(k))+ score(i,pos(k));
        end
        score(i,pre(j)) = an/size(pos,2);
    end    
end

RMSE = calRMSE(score,test,user,item);
disp(['RMSE = ',num2str(RMSE)]);
acc = Accurancy(score,test,user);
disp(['准确率 = ',num2str(acc)]);