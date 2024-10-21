format long
clc; close all; clear

% Read DNS data0
load y_dns.dat
load u_dns.dat
load u2_dns.dat
load v2_dns.dat
load w2_dns.dat
load uv_dns.dat
load dns_data.dat


        % Fluid viscosity
rho = 1;             % Fluid density
      % Pressure gradient
delta = 1.0;         % Channel half-width
% Given data
nu=1/590; % Viscosity
ustar=1;  % Friction velocity
%rho = 1;  % Density
kappa=0.4187; 
dpdx=-1;  % Pressure gradient

ny=131;
y_plus = 55;
y(1,1)=y_plus*nu;
dy=(1-y(1))/(ny);
y(2,1)=y(1,1)+dy;
for i= 3:ny+1
    y(i,1)=y(i-1)+dy;
end
n=length(y);

y_f(1,1)=y(1);
y_f(2,1)=y_f(1,1)+dy/2;
for i=3:n
    y_f(i,1)=y_f(i-1,1)+dy;
end
y_f(n+1,1)=y_f(n,1)+dy/2;

% Cell width
del_y(1,1)=y_f(2)-y_f(1); 
for i=2:n
    del_y(i,1)=y_f(i+1)-y_f(i);
end

% Residual error limit
residue_limit = 10^(-4);

% Under relaxation factor
urf = 0.8;

% residue
residue=1;


count = 1;

% Initialise more variables

% dudy = 0.1 * ones(n, 1);
dR12dy = 0.1 * ones(n, 1);
P11 = 0.1 * ones(n, 1);
nut_f = 0.1 * ones(n+1, 1);
f = 0.1 * ones(n, 1);
tauw = ustar^2;
%for counter=1:10


% K-eps model constants
sigma_k=1;
sigma_epsilon=1.3;
C1=1.44;
C2=1.92;
c_mu=0.09;
E=9;

% load u_dns.dat;
% Initial condition
u_mean_old=rand(n,1);
k=(5*10^(-2))*ones(n,1);
epsilon=(10^(-3))*ones(n,1);
%u_mean(1)=u_dns(26);


% Boundary condition at y=0
k(1)=c_mu^(-1/2)*ustar^2;
epsilon(1)=(ustar^3)/(kappa*y(1));
u_mean_old(1) = ustar*log(9.793*y_plus)/kappa;


% Variables used for computation
u_mean=u_mean_old;
dudy=zeros(n,1);
nut=zeros(n,1);
nu_f=zeros(n+1,1);
nu_fk=zeros(n+1,1);
nu_fw=zeros(n+1,1);
Pk=zeros(n,1);
while (residue > residue_limit)
    
    
    % for count =1:5000
    
    dudy=(u_mean_old(2)-u_mean_old(1))/(y(2)-y(1));
    dudy(2,1)=(3*u_mean_old(2)+u_mean_old(3)-4*u_mean_old(1))/(3*del_y(2));
    for i=3:n-2
        dudy(i,1)=(u_mean_old(i+1)-u_mean_old(i-1))/(y(i+1)-y(i-1));
    end
     dudy(n-1,1)=(-3*u_mean_old(n-1)-u_mean_old(n-2)+4*u_mean_old(n))/(3*del_y(n-1));
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
    % u_mean(1)=((a*u_mean_old(2))-dpdx*del_y(1))/(a+b);

    for i=2:n-2
        % Internal cell centres
        a=nu_f(i+1)/(y(i+1)-y(i));
        b=nu_f(i)/(y(i)-y(i-1));
        u_mean(i)=((a*u_mean_old(i+1))+(b*u_mean(i-1))-dpdx*del_y(i))/(a+b);
    end

    % Boundary cell centre at centreline
    b=nu_f(n)/(y(n)-y(n-1));
    u_mean(n-1)=((b*u_mean_old(n-2))-dpdx*del_y(n-1))/b;
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
    % ustar=(kappa*u_mean(1))/(log(E*ustar*y(1)/nu));
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

    epsilon(1)=(ustar^3)/(kappa*y(1));
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
    
    %Compute residue
    residue_1=sum(abs(u_mean-u_mean_old));
    residue = residue_1;

    u_mean_old = u_mean;

    count=count+1;
end
dudy(1,1)=abs((u_mean(2)-u_mean(1))/(y(2)-y(1)));
for i=2:n-1
    dudy(i,1)=abs(u_mean(i+1)-u_mean(i-1))/(y(i+1)-y(i-1));
end
dudy(n,1)=0;
R11=2*k/3;
R22=2*k/3;
R33=2*k/3;
R12=-nut.*dudy;


eps_dns=dns_data(:,2)/nu;
k=0.5*(R11+R22+R33);
% Calculate dkdy
dkdy(1,1) = (k(2) - k(1)) / (y(2) - y(1));
dkdy(n,1) = (k(n) - k(n-1)) / (y(n) - y(n-1));

for i = 2:n-1
    dkdy(i,1) = (k(i+1) - k(i-1)) / (y(i+1) - y(i-1));
end

% Calculate d2kdy2

d2kdy2(1,1) = (dkdy(2) - dkdy(1)) / (y(2) - y(1));
d2kdy2(n,1) = (dkdy(n-1) - dkdy(n)) / (y(n) - y(n-1));

for i = 2:n-1
    d2kdy2(i,1) = (dkdy(i+1) - dkdy(i-1)) / (y(i+1) - y(i-1));
end
% Total diffusion rate for DNS data
dns_diff = (dns_data(:, 5) / nu) + (dns_data(:, 6) / nu) + (dns_data(:, 4) / nu);
% Turbulent diffusion rate
turb_diff=(nut.*d2kdy2/sigma_k);

% Viscous diffusion rate
visc_diff= (nu * d2kdy2);

% Total diffusion rate
total_diff = turb_diff+visc_diff;

%ustar=1;
yplus=(ustar*y)/nu;
u_loglaw=ustar*((log(yplus)/kappa)+5.2);

figure(1)
plot(u_dns, y_dns, 'bo', u_mean, y, 'k-',u_loglaw,y,'b-', 'LineWidth', 2);
xlabel('U in m/s'); ylabel('y in m'); title('U-velocity');
legend('DNS data', 'k-eps','Log-law','Location','northwest'); legend boxoff;
h=get(gcf, "currentaxes");
set(h, "fontsize", 20);

figure(2)
plot(y_dns, 0.5 * (u2_dns + v2_dns + w2_dns), 'bo', y, k, 'k-', 'LineWidth', 2);
xlabel('y in m'); ylabel('k in m^2/s^2'); title('Turbulence kinetic energy');
legend('DNS data', 'k-eps', 'Location', 'northeast'); legend boxoff;
h=get(gcf, "currentaxes");
set(h, "fontsize", 20);

figure(3)
plot(y_dns, uv_dns, 'bo', y, R12, 'k-', 'LineWidth', 2);
xlabel('y in m'); ylabel('<uv> in m^2/s^2'); title('Reynolds shear stress');
legend('DNS data', 'k-eps', 'Location', 'northeast'); legend boxoff;
h=get(gcf, "currentaxes");
set(h, "fontsize", 20);

figure(4)
plot(y_dns, eps_dns, 'bo', y, epsilon, 'k-', 'LineWidth', 2);
xlabel('y in m'); ylabel('\epsilon in m^2/s^3'); title('Turbulent dissipation');
legend('DNS data', 'k-eps', 'Location', 'northeast'); legend boxoff;
h=get(gcf, "currentaxes");
set(h, "fontsize", 20);
