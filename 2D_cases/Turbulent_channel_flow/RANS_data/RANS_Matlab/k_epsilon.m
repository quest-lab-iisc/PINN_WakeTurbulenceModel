function [R11, R22, R33, R12, epsilon,u_mean,nut,ustar,k] = k_epsilon(n,y,del_y)
% Given data
nu=1/395; % Viscosity
ustar=1;  % Friction velocity
%rho = 1;  % Density
kappa=0.41; 
dpdx=-1;  % Pressure gradient

% K-Omega model constants
sigma_k=1;
sigma_epsilon=1.3;
C1=1.44;
C2=1.92;
c_mu=0.09;
E=9;

% load u_dns.dat;
% Initial condition
u_mean=rand(n,1);
k=(5*10^(-2))*ones(n,1);
epsilon=(10^(-3))*ones(n,1);
%u_mean(1)=u_dns(26);


% Boundary condition at y=0
k(1)=c_mu^(-1/2)*ustar^2;
epsilon(1)=(ustar^3)/(kappa*y(2));


% Variables used for computation
dudy=zeros(n,1);
nut=zeros(n,1);
nu_f=zeros(n+1,1);
nu_fk=zeros(n+1,1);
nu_fw=zeros(n+1,1);
Pk=zeros(n,1);

for count =1:5000

    dudy=(u_mean(2)-u_mean(1))/(y(2)-y(1));
    dudy(2,1)=(3*u_mean(2)+u_mean(3)-4*u_mean(1))/(3*del_y(2));
    for i=3:n-2
        dudy(i,1)=(u_mean(i+1)-u_mean(i-1))/(y(i+1)-y(i-1));
    end
     dudy(n-1,1)=(-3*u_mean(n-1)-u_mean(n-2)+4*u_mean(n))/(3*del_y(n-1));
    dudy(n,1)=0;
    if count<=100
        nut(n,1)=0;
        for i=1:n-1
            nut(i,1)=((kappa*y(i))^2)*abs(dudy(i));
        end
    else
        nut=(c_mu*k.^2)./epsilon;
    end

    % Effective coefficient at each cell faces for all three equations
    % Harmonic mean is used to calculate the effective coefficients
    nu_eff=nut+nu;
    nu_f(1,1)=nu_eff(1,1);
    nu_f(n+1,1)=nu_eff(n,1);
    nu_k=(nut/sigma_k)+nu;
    nu_fk(1,1)=nu_k(1,1);
    nu_fk(n+1,1)=nu_k(n,1);
    nu_w=(nut/sigma_epsilon)+nu;
    nu_fw(1,1)=nu_w(1,1);
    nu_fw(n+1,1)=nu_w(n,1);
    i=2;
    nu_f(2,1)=((y(i)-y(i-1)))/(((del_y(i-1))/nu_eff(i-1))+(del_y(i)/(2*nu_eff(i))));
    nu_fk(2,1)=((y(i)-y(i-1)))/(((del_y(i-1))/nu_k(i-1))+(del_y(i)/(2*nu_k(i))));
    nu_fw(2,1)=((y(i)-y(i-1)))/(((del_y(i-1))/nu_w(i-1))+(del_y(i)/(2*nu_w(i))));
    for i=3:n-1
        nu_f(i,1)=(2*(y(i)-y(i-1)))/(((del_y(i-1))/nu_eff(i-1))+(del_y(i)/nu_eff(i)));
        nu_fk(i,1)=(2*(y(i)-y(i-1)))/(((del_y(i-1))/nu_k(i-1))+(del_y(i)/nu_k(i)));
        nu_fw(i,1)=(2*(y(i)-y(i-1)))/(((del_y(i-1))/nu_w(i-1))+(del_y(i)/nu_w(i)));
    end
    i=n;
    nu_f(i,1)=((y(i)-y(i-1)))/(((del_y(i-1))/(2*nu_eff(i-1)))+(del_y(i)/nu_eff(i)));
    nu_fk(i,1)=((y(i)-y(i-1)))/(((del_y(i-1))/(2*nu_k(i-1)))+(del_y(i)/nu_k(i)));
    nu_fw(i,1)=((y(i)-y(i-1)))/(((del_y(i-1))/(2*nu_w(i-1)))+(del_y(i)/nu_w(i)));

    %Compute U

    % Boundary cell centre at wall
    a=nu_f(2)/(y(2)-y(1));
    b=nu_f(1)/y(1);
    u_mean(1)=((a*u_mean(2))-dpdx*del_y(1))/(a+b);

    for i=2:n-2
        % Internal cell centres
        a=nu_f(i+1)/(y(i+1)-y(i));
        b=nu_f(i)/(y(i)-y(i-1));
        u_mean(i)=((a*u_mean(i+1))+(b*u_mean(i-1))-dpdx*del_y(i))/(a+b);
    end

    % Boundary cell centre at centreline
    b=nu_f(n)/(y(n)-y(n-1));
    u_mean(n-1)=((b*u_mean(n-2))-dpdx*del_y(n-1))/b;
    u_mean(n)=u_mean(n-1);

    %Compute dUdy

    %     dudy(1,1)=abs((u_mean(2)-u_mean(1))/(y(2)-y(1)));
    %     for i=2:n-1
    %         dudy(i,1)=abs(u_mean(i+1)-u_mean(i-1))/(y(i+1)-y(i-1));
    %     end
    %     dudy(n,1)=0;
    dudy=(u_mean(2)-u_mean(1))/(y(2)-y(1));
    dudy(2,1)=(3*u_mean(2)+u_mean(3)-4*u_mean(1))/(3*del_y(2));
    for i=3:n-2
        dudy(i,1)=(u_mean(i+1)-u_mean(i-1))/(y(i+1)-y(i-1));
    end
    dudy(n-1,1)=(-3*u_mean(n-1)-u_mean(n-2)+4*u_mean(n))/(3*del_y(n-1));
    dudy(n,1)=0;

    %Compute Pk

    Pk(1,1)=0;
    for i=2:n-1
        Pk(i,1)=nut(i)*(dudy(i))^2;
    end
    Pk(n,1)=0;

    % Compute k

    % Boundary cell centre at wall
    ustar=(kappa*u_mean(1))/(log(E*ustar*y(1)/nu));
    k(1)=c_mu^(-1/2)*ustar^2;
    for i=2:n-2

        % Internal cell centres
        a_k=nu_fk(i+1)/(y(i+1)-y(i));
        b_k=nu_fk(i)/(y(i)-y(i-1));
%         if epsilon(i)<0
%             epsilon(i)=abs(epsilon(i));
%         end
        k(i)=((a_k*k(i+1))+(b_k*k(i-1))+(del_y(i)*(Pk(i)-epsilon(i))))/(a_k+b_k);
    end

    % Boundary cell centre at centreline
    b_k=nu_fk(n-1)/(y(n-1)-y(n-2));
    k(n-1)=((b_k*k(n-2))+(del_y(n-1)*(Pk(n-1)-epsilon(n-1))))/(b_k);
    k(n)=k(n-1);

    %Compute epsilon

    epsilon(1)=(ustar^3)/(kappa*y(2));
    for i=2:n-2
        % Internal cell centres
        a_w=nu_fw(i+1)/(y(i+1)-y(i));
        b_w=nu_fw(i)/(y(i)-y(i-1));
%         if k(i)<0
%             k(i)=abs(k(i));
%         end
        Sp=-C2*epsilon(i)/k(i);
        epsilon(i)=((a_w*epsilon(i+1))+(b_w*epsilon(i-1))+del_y(i)*(C1*Pk(i)*epsilon(i)/k(i)))/(a_w+b_w-del_y(i)*Sp);
    end

    % Boundary cell centre at centreline

    b_w=nu_fw(n-1)/(y(n-1)-y(n-2));
    Sp=-C2*epsilon(n-1)/k(n-1);
    epsilon(n-1)=((b_w*epsilon(n-2))+del_y(n-1)*(C1*Pk(n-1)*epsilon(n-1)/k(n-1)))/(b_w-del_y(n-1)*Sp);

    epsilon(n)=epsilon(n-1);

end
%Compute dUdy

dudy(1,1)=abs((u_mean(2)-u_mean(1))/(y(2)-y(1)));
for i=2:n-1
    dudy(i,1)=abs(u_mean(i+1)-u_mean(i-1))/(y(i+1)-y(i-1));
end
dudy(n,1)=0;
R11=2*k/3;
R22=2*k/3;
R33=2*k/3;
R12=-nut.*dudy;
