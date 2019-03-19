function H = LANE2(Net,Attri,Label,d,alpha1,alpha2,varargin)
   n = size(Net,1);
   LG = norLap(Net); %��һ������������˹����
   LA = norLap(Attri); %��һ���ڵ�����������˹����
   UAUAT = zeros(n,n); %UA*UA^T
   opts.disp = 0;
   
   if isempty(varargin) %�޼ල
        numiter = alpha2;
        beta1 = d;
        beta2 = alpha1;
        d = Label;
        H = zeros(n,d);
        for i=1:numiter
            HHT = H*H';
            TotalLG1 = LG+beta2*UAUAT+HHT;
            [UG,~] = eigs(.5*(TotalLG1+TotalLG1'),d,'LA',opts);
            UGUGT = UG*UG';
            
            TotalLA = beta1*LA+beta2*UGUGT+HHT;
            [UA,~] = eigs(.5*(TotalLA+TotalLA'),d,'LA',opts);
            UAUAT = UA*UA';
            
            TotalLH = UAUAT + UGUGT;
            [H,~] = eigs(.5*(TotalLH+TotalLH'),d,'LA',opts);
         end
   else %�мල
       numiter = varargin{1};
       H = zeros(n,d);
       LY = norLap(Label*Label');
       UYUYT = zeros(n,n);
       for i = 1:numiter
            HHT = H*H';
            TotalLG1 = LG + alpha1*UAUAT + alpha2*UYUYT + HHT;
            [UG,~] = eigs(.5*(TotalLG1+TotalLG1'),d,'LA',opts);
            UGUGT = UG*UG';
            
            TotalLA = alpha1*(LA+UGUGT) + HHT;
            [UA,~] = eigs(.5*(TotalLA+TotalLA'),d,'LA',opts);
            UAUAT = UA*UA';
            
            TotalLY = alpha2*(LY+UGUGT)+HHT;
            [UY,~] = eigs(.5*(TotalLY+TotalLY'),d,'LA',opts);
            UYUYT = UY*UY';
            
            TotalLH = UAUAT + UGUGT+UYUYT;
            [H,~] = eigs(.5*(TotalLH+TotalLH'),d,'LA',opts);
       end
   end
end

%����inpx�Ĺ�һ��ͼ������˹����
function Lapx = norLap(Inpx)
    Inpx = Inpx';
    Inpx = bsxfun(@rdivide,Inpx,sum(Inpx.^2).^.5); %��һ��
    Inpx(isnan(Inpx))=0;
    sx = Inpx'*Inpx;
    nx = length(sx);
    sx(1:nx+1:nx^2) = 1+10^-6;
    dxInv = spdiags(full(sum(sx,2)).^(-.5),0,nx,nx);
    Lapx = dxInv*sx*dxInv;
    Lapx = .5*(Lapx+Lapx');
end
