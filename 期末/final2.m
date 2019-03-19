clc;
train = load('u1.base');
test = load('u1.test');
user = 943; %用户个数
item = 1682; %项目个数
type = 19; %项目类型个数
attr = 3; %用户特征

%用户-项目评分矩阵
userItem = zeros(user,item);
for i=1:size(train,1)
    for j=1:3
        userItem(train(i,1),train(i,2)) = train(i,3);
    end
end

UI = userItem;
% RMSE = calRMSE(userItem,test);
% disp(['RMSE = ',num2str(RMSE)]);

%项目-类别矩阵
itemType = zeros(item,type);
file = fopen('u.item');
ty=textscan(file,'%n %s %s %s %s %n %n %n %n %n %n %n %n %n %n %n %n %n %n %n %n %n %n %n','delimiter','|');
fclose(file);
for i=6:length(ty)
    for j=1:item
        itemType(j,i-5) = ty{i}(j); 
    end
end

%构建用户-类别矩阵
userType = zeros(user,type);
for i=1:user
    %找出该用户评分了的项目
    r = find(userItem(i,:)~=0);
    typenum = zeros(1,type);
    for j=1:size(r,2)
        %对于每个项目，把评分加入到用户-类别矩阵中，并且在tynum中记录个数
        for k=1:type
            if itemType(r(j),k)==1
                userType(i,k) = userType(i,k) + userItem(i,r(j));
                typenum(1,k) = typenum(1,k) + 1;
            end
        end
    end
    %将用户-类别矩阵中的评分总和取平均值
    for k=1:type
        if typenum(1,k)~=0
            userType(i,k) = userType(i,k)/typenum(1,k);
        end
    end
end

%根据用户-类别矩阵计算用户之间的相似性
simuv = zeros(user,user);
for u=1:user-1 
    for v=u+1:user
        %计算uv的皮尔逊相关系数
        z = corrcoef(userItem(u,:),userItem(v,:));
        simuv(u,v) = z(1,2);
        simuv(v,u) = simuv(u,v); 
    end    
end
% simuv = real(simuv);

%用用户兴趣来填充空缺值
N = 5;
for u = 1:user
    %相似度从大到小排序
    sortsim = sort(simuv(u,:),'descend');
    %找出N个相似度最高并且给项目ic评分了的用户
    sameU = zeros(1,N); %记录N个用户
    index=1;
    sortind = 1;
    while index <= N
        bigR = find(simuv(u,:)==sortsim(1,sortind));
        sortind = sortind + 1;
        num = size(bigR,2);
        for j=1:num
            sameU(1,index) = bigR(j);
            index = index + 1;
        end
    end
    %找出用户u未评分的项目
    Ic = find(userItem(u,:)==0); 
    for i=1:size(Ic,2)     
        up = 0;
        down = 0;
        for j=1:N
            up = up + simuv(u,sameU(j))*userItem(sameU(j),Ic(i));
            down = down + simuv(u,sameU(j));
        end      
        userItem(u,Ic(i)) = real(up/down ); 
    end  
end

RMSE = calRMSE(userItem,test);
disp(['填充后RMSE = ',num2str(RMSE)]);

UIJ = userItem;

%基于用户-项目评分矩阵以及用户特征的推荐
userAttr = zeros(user,attr);
file = fopen('u.user');
ua=textscan(file,'%n %n %s %s %s','delimiter','|');
fclose(file);
for i=2:length(ua)-1
    for j=1:user
        if i==2
            userAttr(j,i-1) = ua{i}(j);
        end
        if i==3
            num = panDuan(2,ua{i}(j));
            userAttr(j,i-1) = num;
        end
        if i==4
            num = panDuan(3,ua{i}(j));
            userAttr(j,i-1) = num;
        end 
    end
end

a = 0.05; %用户特征相似性的权重
simNuv = zeros(user,user);
simMuv = zeros(user,user);
simBuv = zeros(user,user);
RU = zeros(user,1);
for u=1:user
    uc = find(userItem(u,:)~=0);
    unum = size(uc,2);
    for i=1:unum
        RU(u) = RU(u)+userItem(u,uc(i));
    end
    RU(u) = RU(u)/unum;
end

for u=1:user-1
    for v=u+1:user
        z= corrcoef(userItem(u,:),userItem(v,:));
        simNuv(u,v) = z(1,2);
        simNuv(v,u) = simNuv(u,v); 
        sameAttr = 0;
        for i=1:attr
            if userAttr(u,i)==userAttr(v,i)
                sameAttr = sameAttr + 1;
            end
        end
        simMuv(u,v) = real(sameAttr/attr);
        simMuv(v,u) = simMuv(u,v);
        simBuv(u,v) =real( a*simNuv(u,v) + (1-a)*simMuv(u,v));
        simBuv(v,u) = simBuv(u,v);
    end    
end

%用用户-项目评分矩阵来预测评分

Nu =150;
doubleSim = zeros(user,item);
for u = 1:user
    %相似度从大到小排序
    sortsim = sort(simBuv(u,:),'descend');
    %找出N个相似度最高并且给项目ic评分了的用户
    sameU = zeros(1,Nu); %记录N个用户
    index=1;
    sortind = 1;
    dsim = 0;
    while index <= Nu
        bigR = find(simBuv(u,:)==sortsim(1,sortind));
        sortind = sortind + 1;
        num = size(bigR,2);
        for j=1:num
            sameU(1,index) = bigR(j);
            index = index + 1;
            dsim = dsim + sortsim(1,sortind)*sortsim(1,sortind);
%             dsim = dsim + sortsim(1,sortind);
        end
    end
    
    %找出用户u未评分的项目
    Ic = find(UI(u,:)==0); 
    for i=1:size(Ic,2) 
        doubleSim(u,Ic(i)) = dsim;
        up = 0;
        down = 0;
        for j=1:Nu
            up = up + simBuv(u,sameU(j))*userItem(sameU(j),Ic(i));
%             up = up + simBuv(u,sameU(j))*(userItem(sameU(j),Ic(i))-RU(sameU(j)));
            down = down + simBuv(u,sameU(j));
        end 
        userItem(u,Ic(i)) =real(RU(u) + up/down );
    end  
end

RMSE = calRMSE(userItem,test);
disp(['基于用户-RMSE = ',num2str(RMSE)]);

%基于用户-项目评分矩阵以及项目属性的推荐
b=0.99;
simNij = zeros(item,item);
simMij = zeros(item,item);
simBij = zeros(item,item);
RI = zeros(item,item);
for i=1:item-1
    for j=i+1:item
%         z= corrcoef(userItem(:,i),userItem(:,j));
        z = corrcoef(UIJ(:,i),UIJ(:,j));
        if isnan(z(1,2))==1
            simNij(i,j)=0;
        else
            simNij(i,j) =z(1,2);
        end
        simNij(j,i) = simNij(i,j);
        sameType = 0;
        for k=1:type
            if itemType(i,k)==itemType(j,k)
                sameType = sameType + 1;
            end
        end
        simMij(i,j) = real(sameType/type);
        simMij(j,i) = simMij(i,j);
        simBij(i,j) =real( b*simNij(i,j) + (1-b)*simMij(i,j));
        simBij(j,i) = simBij(i,j);
    end
end

%用用户-项目评分矩阵来预测评分
Ni =15; 
neigbor = zeros(item,Ni);%用来记录N个相似度最高的邻居
douSim = zeros(user,item);
for i=1:item
    %找出未给i评分的用户
    ur = find(UI(:,i)==0);
    if size(ur,1)==0 || size(ur,2)==0
        continue;
    else
        %相似度从大到小排序
        sortsimij = sort(simBij(i,:),'descend');
        %找出N个相似度最高的项目
        sameI = zeros(1,Ni); %记录N个项目
        index=1;
        sortind = 1;
        dsim = 0;
        out=0;
        while index <= Ni
            bigR = find(simBij(i,:)==sortsimij(1,sortind));
            sortind = sortind + 1;
            num = size(bigR,2);
            for j=1:num
                sameI(1,index) = bigR(j);
                index = index + 1;
                dsim = dsim + sortsimij(1,sortind)*sortsimij(1,sortind);
%                 dsim = dsim + sortsimij(1,sortind);
                if index>Ni
                    out=1;
                end
            end
            if out==1
                break;
            end
        end
        unum = size(ur,1);
        for u = 1:unum
            douSim(ur(u),i) = dsim; 
            up=0;
            down=0;
            for j=1:Ni
                up = up + simBij(i,sameI(j))*UIJ(ur(u),sameI(j));
                down = down + simBij(i,sameI(j));
            end
            UIJ(ur(u),i) = real(up/down); 
            if isnan(UIJ(ur(u),i))==1
                UIJ(ur(u),i)=0;
            end
        end
    end
end

RMSE = calRMSE(UIJ,test);
disp(['基于项目相似性-RMSE = ',num2str(RMSE)]);


%两者结合起来
for u=1:user
    %找出u中未评分的项目
    Ic = find(UI(u,:)==0);
    for i=1:size(Ic,2)
        c1 = doubleSim(u,Ic(i))/(doubleSim(u,Ic(i))+douSim(u,Ic(i)));
        c2 = douSim(u,Ic(i))/(doubleSim(u,Ic(i))+douSim(u,Ic(i)));
        userItem(u,Ic(i)) = c1*userItem(u,Ic(i))+c2*UIJ(u,Ic(i));
    end
end
RMSE = calRMSE(userItem,test);
disp(['结合两者-RMSE = ',num2str(RMSE)]);
