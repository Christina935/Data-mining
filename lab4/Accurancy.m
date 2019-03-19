function acc = Accurancy(train,test,user)
num=0;
right = 0;
for i=1:user
    r = find(test(i,:)~=-1);
    nr = size(r,2);
    num = num + nr;
    for j=1:nr
        err = (test(i,r(j))-train(i,r(j)))*(test(i,r(j))-train(i,r(j)));
        if err < 1
            right = right+1;
        end
    end
end
acc = right/num;
end