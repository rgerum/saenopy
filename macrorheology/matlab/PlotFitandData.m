rheodata=load('data_exp/rheodata.dat');
stretcherdata=load('data_exp/stretcherdata.dat');
rheosim=load('rheosim.dat');
stretchersim=load('stretchersim.dat');

figure;
hold off
plot(rheosim(:,1),rheosim(:,2),'linewidth',2.5);
hold all
plot(rheodata(:,1),rheodata(:,2),'linewidth',2.5);

grid on

axis([1e-2 1e0 1e-2 1e4])
set(gca, 'xscale', 'log', 'yscale', 'log', 'xminorgrid', 'off', 'yminorgrid', 'off','box', 'off', ...
    'fontsize',14 , 'xtick', [1e-2 1e-1 1e-0], 'ytick', [1e-2 1e0 1e2 1e4], ...
    'ticklength', [0.03 0.03], 'linewidth', 1, 'Position', [0.15 0.15 0.70 0.70]);
xlabel('Engineering shear strain', 'fontsize', 15);
ylabel('Shear stress [Pa]', 'fontsize', 15);

h=legend('Model fit', 'Data', 'Location','northwest')
set(h, 'fontsize', 12, 'box','off');
set(gcf, 'Color', 'w');

strain=rheosim(:,1);
stress=rheosim(:,2);

stiffness=gradient(stress)./gradient(strain);

%figure; plot(stress,stiffness);

%xlim([0 0.4]);
%hold off

figure;
hold off

plot(stretchersim(:,1),stretchersim(:,2),'linewidth',2.5);
%stretcherdata=load('data/stretcherdata.dat');
hold all
plot(stretcherdata(:,1),stretcherdata(:,2),'linewidth',2.5);
%plot(stretcherdata(:,1),stretcherdata(:,2).*stretcherdata(:,1));
hold off
%xlim([0.95 1])

grid on

axis([0.9 1.1 -0.5 1.1])
set(gca, 'xscale', 'lin', 'yscale', 'lin', 'xminorgrid', 'off', 'yminorgrid', 'off','box', 'off', ...
    'fontsize',14 , 'xtick', [0.9 0.95 1 1.05 1.1], 'ytick', [0 0.5 1], ...
    'ticklength', [0.03 0.03], 'linewidth', 1, 'Position', [0.15 0.15 0.70 0.70]);
xlabel('Horizontal strech', 'fontsize', 15);
ylabel('Vertical contraction', 'fontsize', 15);

h=legend('Model fit', 'Data', 'Location','west')
set(h, 'fontsize', 12, 'box','off');
set(gcf, 'Color', 'w');

% figure;
% eps=load('epsilon.dat');
% loglog(abs(eps(:,1)),abs(eps(:,2)));
% 
% 
% figure;
% epsb=load('epsbar.dat');
% loglog(abs(epsb(1:1000:end,1)),abs(epsb(1:1000:end,2)));
% 
% 
% figure;
% eps=load('epsilon.dat');
% loglog(abs(eps(:,1)),abs(eps(:,2)));
