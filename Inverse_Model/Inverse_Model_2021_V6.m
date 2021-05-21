clear all; clc; close all;
format longEng

%% READ THE FILE AND SET SPECIFICATIONS OF HYDROPHONE
fileID = fopen('mag_73_07_12_2012.txt','r');
formatSpec = '%f';
A = fscanf(fileID,formatSpec);
Fs=250;
y=A;
aux=1:length(y);
t=aux./Fs; % Generate time vector related to the signal
y=y-mean(y); % Remove mean
y=y./(10^6); % Conversion from Micropascals to Pascals

%% Spectral content
len=4000;
s = spectrogram(y,len,[],len,Fs);
figure(2)
subplot(2,1,1)
plot(t,y)
%axis([0 900 -10 10])
%xlabel('Time [s]')
ylabel('Pressure [Pa]')

subplot(2,1,2)
[~,F,T,P] =spectrogram(y,len,[],len,Fs,'yaxis');
imagesc(T, F, 10*log10(P+eps)); % add eps like pspectrogram does
axis xy
ylabel('Frequency (Hz)')
xlabel('Time (s)')
h = colorbar;
h.Label.String = 'Power/frequency (dB/Hz)';



a=findobj(gcf); % get the handles associated with the current figure
allaxes=findall(a,'Type','axes');
alllines=findall(a,'Type','line');
alltext=findall(a,'Type','text');
set(allaxes,'FontName','Arial','LineWidth',1.5,...
    'FontSize',10);
set(alllines,'Linewidth',1.2);
set(alltext,'FontName','Arial','FontSize',11);
% I RECOMMEND TO STOP HERE THE SCRIPT TO FIRST VISUALISE AND IDENTIFY
% FREQUENCIES IN THE SPECTROGRAM
%% RANGES FOR PROPERTIES BASED ON WELLS AND COPPERSMITH 1994 RELATED TO MW
% L_range=(1*10^3: 1*10^3: 201*10^3); %[m]
% b_range=(2.5*10^3 :1*10^3: 82.5*10^3); %[m]
% DISP=[0.05, 9]; % Displacement

% IF USING WELLS 1994 INTRODUCE THE MAGNITUDE
T_range=(1.25: 0.5: 20.25); %[s]  THIS VALUE REMAINS FOR BOTH APPROACHES

MW=7.2;
a=-3.22 ;b=0.69 ;
SRL=10^(a+b*MW);
L_range=(round(SRL*0.5)*10^3: 1*10^3: round(SRL*2)*10^3); %[m]
% WITH THE RANGES I INTEND TO REPRESENT THE SCATTER FOUND ON THE PARAMETERIZATIONS

a=-3.49 ;b=0.91 ;
RA=10^(a+b*MW);

BRL=RA/SRL;
b_range=(round(BRL*0.5)*10^3: 1*10^3: round(BRL*2)*10^3); %[m]

a=-5.46 ;b=0.82 ;
DISP=10^(a+b*MW);
W_range=(round((DISP*2)/max(T_range),3): 0.01: round((DISP*0.5)/min(T_range),3)); %[m/s]


stepL=1000;
stepT=0.2;
stepb=1*10^3;

%% Set up parameters for the model
% I assume I know the frequency range after looking at the spectrogram
h=4000; %potential average depth
r_dist=3050*10^3; % Distance between epicentre and hydrophone
d_step=100*10^3; % Grid step for X0 considerations
% Range of frequencies defined by spectrogram
max_f=15;
min_f=0.8;


frec=Fs;
c=1500;
ro=1000;
window_ener=4000; % Size of the short time energy analysis window in samples
n_iterations_pks=4; % Number of iterations for envelope tracking
b_sol_point=20; % Number of solutions asked for sin(kb)
maxerror=0.3; % maximun error that can have the solution, this is for T and L
percent_diff_T_sol=20; % Percentage to consider solutions from histogram for T
%% HERE I CHOSE THE START AND END OF MY SIGNAL
% Short time energy is computed
ww=y; power=[];
for i =1:length(t)/window_ener
    aux2= y(1 +window_ener*(i-1) : window_ener+window_ener*(i-1)).* y(1 +window_ener*(i-1) : window_ener+window_ener*(i-1));
    power=[power, sum(aux2)];
end
close(figure(2))%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
buttonSelections = 1; count2=0;

while buttonSelections == 1

    figure(1)
    subplot(2,1,1)
    plot(t,ww)
    title('IMPORTANT TO CLICK ON THE SIGNAL AND NOT ON THE ENERGY')
    xlabel('Time (s)')
    ylabel('Pressure (Pa)')
    axis([min(t) max(t) min(ww) max(ww)])
    hold on
    
    subplot(2,1,2)
    plot(power)
    xlabel('Time (s)')
    ylabel('Power [RMS]')
    axis([0 length(power) min(power) max(power)])
    
    x0p=300;  y0p=200;
    widthp=700;  heightp=400;
    set(gcf,'position',[x0p,y0p,widthp,heightp])

    % Here a FIRST point in the plot is defining the BEGGINING of the region-----
    wg = warndlg('Zoom and then click OK when ready', 'Zoom');
    wg.Position(1) = wg.Position(1)-300;
    waitfor(wg);
    
    [x,y] = ginput;
    aa=[];
    for i=1:length(x)    
        aa(i,:)=abs(x(i)-t); 
    end
    locc=[];
    for i=1:length(x)     
        locc(i)=find(aa(i,:)==min(aa(i,:)));
    end
    
    % Here a second point in the plot is defining the end of the region-----
    wg = warndlg('Zoom and then click OK when ready', 'Zoom');
    wg.Position(1) = wg.Position(1)-300;
    waitfor(wg);
 
    [x,y] = ginput;
    aa=[];
    for i=1:length(x)    
        aa(i,:)=abs(x(i)-t); 
    end
    locc2=[];
    for i=1:length(x)     
        locc2(i)=find(aa(i,:)==min(aa(i,:)));
    end
    
    %% THIS NEED A CHECK CAUSE IM NOT SURE IF IT WORKS FOR MORE THAN 1 AREA
    promptMessage = sprintf('Do you want more areas');
    titleBarCaption = 'Yes or No';
    button = questdlg(promptMessage, titleBarCaption, 'Yes', 'No', 'Yes');
    
      if strcmpi(button, 'Yes')
        buttonSelections= 1;
      else
        buttonSelections = -1;
      end
end

t_length=t(locc2)-t(locc);% Approximated length in seconds of the audio content
Audio_start=t(locc);% Beggining of audio content relative to the recorded signal

%% THE POTENTIAL FREQUENCY DISTRIBUTIONS RELATED TO THE POTENTIAL LOCATIONS ARE COMPUTED
t_r=t;
w=pi*c/(2*h); % Frequency for the FIRST MODE
x00=0:d_step:r_dist;
y00=sqrt((r_dist).^2-(x00).^2);
t0=0-(sqrt(y00(1).^2+x00(1).^2) ./ c);
OM=[];
for i=1:length(x00)
    t=[-t0  -t0+t_length]; %%%% THIS IS THE LENGTH OF SIGNAL WE WANT TO CONSIDER
    omega=(w ./ sqrt(1- (x00(i)./(c.*t)).^2 ) ); % Frequency
    OM=[OM; omega];
end

pos=[];
% Verify that the frequencies are in the required range
for i=1:length(x00)
    if OM(i, 1)<max_f && OM(i, 2)>min_f
        pos=[pos, i];
    end
end

omega=[];
for il=1:length(pos)  
    x0=x00(pos(il));
    y0=y00(pos(il));
    t=[-t0 : 1/frec : -t0+t_length];
    omega(il,:)=(w ./ sqrt(1- (x0./(c.*t)).^2 ) );
    hold on
    plot(t-(t(1))+Audio_start,omega(il,:),'k','Linewidth',1)
end

close(figure(1))%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(3)
subplot(2,1,1)
plot(t_r(locc:locc2), ww(locc:locc2))
ylabel('Pressure (Pa)')
xlabel('Time (s)')

t1=t_r(locc:locc2);
y1=ww(locc:locc2);
%% Tracking the envelope

pks_aux=[]; locs_aux=[];locs_aux2=[];
pks=y1;
[pks,locs_aux]=findpeaks(pks);
locs_aux2=locs_aux;
for itera=1:n_iterations_pks % n iterations +1 which is out of the loop

    [pks,locs_aux]=findpeaks(pks);
    locs_aux2=locs_aux2(locs_aux);
    
end
tims2=t1(locs_aux2);
pks2=pks;

hold on
plot(tims2, pks2, 'o')

periods=tims2(2:end)-tims2(1:end-1);
frequencies=(2*pi)./periods;

subplot(2,1,2)
plot(tims2(1:end-1), frequencies)
xlabel('Time (s)')
ylabel('Frequency [Hz]')
% I verify all frequencies are below the wanted frequency so i can force 
% the obtained distribution

time1=(1:length(omega))/ frec; % times associated to omegas
time2=tims2-t_r(locc);% times associated to pks
finaltims=time2; finalpks=pks2;

% find the location of the related omegas to the tracked envelop
aa=[];
for i=1:length(time2)    
    aa(i,:)=abs(time2(i)-time1); 
end
locc=[];
for i=1:length(time2)     
    locc=[locc, find(aa(i,:)==min(aa(i,:)))];
end
omeg=[];
for ili=1:size(omega,1)
    omeg(ili,:)=omega(ili,locc);
end

% These are the points that the model will use
figure(5)
subplot(2,1,1)
plot(t1-(t1(1)),y1)
hold on 
plot(time2, pks2, 'o')
xlabel('Time (s)')
ylabel('Pressure [Pa]')

subplot(2,1,2)
for ili=1:size(omega,1)
    plot(time1, omega(ili,:))
    hold on
    plot(time2, omeg(ili,:),'o')
end
xlabel('Time (s)')
ylabel('Frequency [Hz]')

%% NOW ORIENTATION, LOCATION AND FREQUENCY DISTRIBUTION ARE KNOWN
prompt = 'How many solutions do you want?';
nnn = input(prompt) % NUMBER OF SETS OF SOLUTIONS THAT I WANT
% Here nnn number of non repeated combinations from the available points
% are made they are also sorted getting the latest points at the beggining
% since those are the selected to compute the location and eruption time

prompt = 'How many points per solution should be used?';
npts = input(prompt) % NUMBER OF POINTS FROM THE SIGNAL THAT I WANT FOR COMPUTING EACH SET OF SOLUTIONS

nlocs=[];
tic
%% I obtain solutions for all possible omega distributions
ome_ga=omega;
W0S=[]; TS=[]; LS=[]; BS=[];
for jj=1:size(ome_ga,1)
    
    omegs=omeg(jj,:);
    for i=1:nnn
       %This part ensures that the points are not too close
       nlocs(i,:)=sort(randperm(length(finaltims),npts)) ;

       tiempo1=finaltims(nlocs(i,1));
       tiempo2=finaltims(nlocs(i,2));
       while abs(abs(tiempo1)-abs(tiempo2))<10 % TUNING PARAMETER IMPORTANT DISTANCE IN SECONDS BUT OPNLY FROM THE 2 FIRST POINTS, THEY COMPUTE LOCATION
            nlocs(i,:)=sort(randperm(length(finaltims),npts)) ;
            tiempo1=finaltims(nlocs(i,1));
            tiempo2=finaltims(nlocs(i,2));
       end
    end
    %% Properties of the slender fault
    X00=[]; Y00=[]; T00=[];q=0; bsol=[]; Tsol=[]; W00sol=[]; Lsol=[]; 
    pruebaT=[]; iteration=0; W0prue=[]; W0prue2=[]; 
    t02=sqrt(x0^2+y0^2)/c;% t0 is defined because the location is known
    for j=1:length(nlocs)
        % Points from and related to the signal are defined
        iteration=iteration+1
        tiempos=finaltims(nlocs(j,:))+t02;
        presiones=finalpks(nlocs(j,:));  
        omegas=omegs(nlocs(j,:));
        kaa=omegas.*x0./(c^2.*(tiempos+t02)); % Wavenumber

        %% b (WIDTH)

        n=[1:b_sol_point]; % I want the first X solutions for each point in the analytical solution of 1=sin(kb)
        b2=[];
        for i = 1:length(tiempos)
            b2(i,:)= ((n-1/2)*pi)/kaa(i);
        end
        bb=b_range; histo=[];

        % We track around what points is the convergence in the solutions
        for j= 1:length(tiempos)
            for i=1:length(bb)-1
                range=bb(i)+stepb;
                auxil=find(b2(j,:)<range & b2(j,:)>bb(i));
                rr=isempty(auxil);    
                if j==1
                    if rr==1
                        histo=[histo,0];
                    else
                        histo=[histo,1];
                    end
                else
                    if rr==1
                        histo(i)=histo(i);
                    else
                        histo(i)=histo(i)+1;
                    end
                end
            end
        end

        % The convergence of points is found
        B=find(histo==max(histo));
        b=bb(B)+0.5*10^3;
        % If more than a solution is found they are averaged
        if length(b)>1
            baux=sum(b)/length(b);
            b=baux;
        end

        %% HERE T IS CALCULATED EVALUATION ALL POSSIBLE SOLUTIONS FOR EACH L IN A RANGE THAT WE ESTABLISH
        maxthreshT=max(T_range); %Range of possible T
        minthreshT=min(T_range);
        maxthreshL=max(L_range); %Range of possible L
        minthreshL=min(L_range); 

        % Combinations between the chosen points in the chart because we need 2 points to compute the formulation
        combos = nchoosek(1:length(tiempos),2);
        L=[minthreshL:stepL:maxthreshL];
        TTT=[];
        % All possible T are evaluated
        for i=1:size(combos,1)
            n1=combos(i,1);
            n2=combos(i,2);
            kaa1=kaa(n1);
            kaa2=kaa(n2);
            omegas1=omegas(n1);
            omegas2=omegas(n2);
            presionn1=presiones(n1);
            presionn2=presiones(n2);
            TTT(:,:,i)=calclt(x0,y0,b,stepL,stepT,maxthreshT,minthreshT,maxthreshL,minthreshL,maxerror,kaa1,kaa2,omegas1,omegas2,presionn1,presionn2);
        end

        Ts=(minthreshT:stepT:maxthreshT);
        histo=[]; xx=[]; rr=[];
        % The convergence of solutions is evaluated
        for i=1:size(combos,1)
            for j=1:length(Ts)
                xx=find(TTT(:,:,i)>Ts(j) & TTT(:,:,i)<Ts(j)+stepT);
                rr=isempty(xx);
                histo(i,j)=length(xx);   
            end
        end
        fff=sum(histo);

        % THE UNCERTAINTY OF T IS EVALUATED AND ASSOCIATED TO THIS SOLUTION
        sum2=fff;
        val=[]; ind=[];
        [val ind] = sort(sum2,'descend');

        % I ONLY CONSIDER THE SOLUTIONS WITH LESS THAN percent_diff_T_sol DIFFERENCE TO
        % THE HIGHEST CONVERGENCE OF SOLUTIONS
        top2=[val(1)];
        TTS=[Ts(ind(1))]+stepT/2;
        if length(val)>1
            if (val(1)-val(2))/val(1)*100<percent_diff_T_sol 
                top2=[val(1),val(2)];
                TTS=[Ts(ind(1)), Ts(ind(2))]+stepT/2;
            end
            if (val(1)-val(3))/val(1)*100<percent_diff_T_sol
                top2=[val(1),val(2),val(3)];
                TTS=[Ts(ind(1)), Ts(ind(2)), Ts(ind(3))]+stepT/2;
            end
            if (val(1)-val(4))/val(1)*100<percent_diff_T_sol
                top2=[val(1),val(2),val(3),val(4)];
                TTS=[Ts(ind(1)), Ts(ind(2)), Ts(ind(3)), Ts(ind(4))]+stepT/2;
            end
        end

        pruebaT=[pruebaT, mean(TTS)];
        % Here the chosen possible solutions for T are evaluated again by
        % looking at the error that the generate using the function calct2
        cnr=[];
        T=[];
        for i=1:size(combos,1)
            n1=combos(i,1);
            n2=combos(i,2);
            kaa1=kaa(n1);
            kaa2=kaa(n2);
            omegas1=omegas(n1);
            omegas2=omegas(n2);
            presionn1=presiones(n1);
            presionn2=presiones(n2);

            for j=1:length(TTS)
               LLL=calct2(x0,y0,b,stepL,TTS(j),maxthreshL,minthreshL,maxerror,kaa1,kaa2,omegas1,omegas2,presionn1,presionn2); 
               auxill=find(LLL==1);
               cnr(j)=length(auxill);
            end
            indi=find(cnr==max(cnr));
            T=[T, TTS(indi)];
        end

        cnr=[];
        for i=1:length(TTS)
            tb=TTS(i);
            tb1=find(T==tb);
            cnr=[cnr, length(tb1)];
        end
        T=TTS(find(cnr==max(cnr)));

        if length(T)>1
            Taux=sum(T)/length(T);
            T=Taux;
        end
        Tr=T; % The convergence is found around certain period

        %% W0
        
        W_range=(round((DISP*0.5)/Tr,3): 0.01: round((DISP*2)/Tr,3)); %[m/s]

        %W_range=(round((DISP(1))/Tr,3): 0.01: round((DISP(2))/Tr,3)); %[m/s]

        LL=L_range; W00=W_range; T=Tr;
        % redefine k1, omega and t, we just need 2 points n1 and n2
        n1=2; n2=3;

        k1=[kaa(n1), kaa(n2)];
        omega=[omegas(n1), omegas(n2)];
        t=[tiempos(n1), tiempos(n2)];
        pressure=[presiones(n1), presiones(n2)];
        solucion1=[]; solucion2=[]; sol1=[];  sol2=[];

        % Two surfaces are generated which display the error for each combination of L and W0 when 
        % they generate pressure compared to the measured pressure, since the
        % solution of W0 is unique there is a line for similar values of W0
        % that is averaged in order to give a final solution for W0

        for i=1:length(LL)
            for j=1:length(W00)

                E=b/LL(i);
                l=E*LL(i); % b
                X=x0*(E^2);
                Y=y0*E;
                Y1pos= (l+Y)/2 ;
                Y1neg= (l-Y)/2 ;
                v=X./k1;
                X1=v./2;    
                a1=fcs(sqrt(2./(pi.*X1)) .*  Y1pos);
                a2=fcs(sqrt(2./(pi.*X1)) .*  Y1neg);
                a3=a1/2;
                a4=a2/2;
                a5=real(a3)+real(a4)+imag(a3)+imag(a4);
                a6=-real(a3)-real(a4)+imag(a3)+imag(a4);
                At=sqrt(a5.^2+a6.^2); % Envelope

                p1= ro.*W00(j).*At  .*   ((2.^(5./2) .* (c.^3) .*  ((t).^(1/2)))  ./  (h.* pi.^(1./2) .* w.^(3./2) .*x0 ) );
                p2= (1 - (x0./ (c.*(t))).^2).^(1./4);
                p3= sin( k1.* b  );
                p4= sin ( omega .* T   );
                aux= p1.*p2.*p3.*p4;
                press=abs(aux); % Pressure signal       

                val1=pressure(1)-press(1);
                val2=pressure(2)-press(2);
                solucion1=[solucion1, abs(val1)];
                solucion2=[solucion2, abs(val2)];
            end
            sol1=[sol1; solucion1];
            sol2=[sol2; solucion2];
            solucion1=[]; solucion2=[];
        end

        aa1=sum(sol1); aa2=sum(sol2);
        ss1=find(aa1==min(aa1));
        ss2=find(aa2==min(aa2));
        W03=(W00(ss1)+W00(ss2))/2;
        
        %%%% FIND THE MINIMUM OF EACH SURFACE
        [row,col1] = find(sol1==min(sol1(:)));
        [row,col2] = find(sol2==min(sol2(:)));
        W04=(W00(col1)+W00(col2))/2;
        %% L

        LLL=[];
        for i=1:length(kaa)
            LL=L_range;

            E=b./LL;
            l=E.*LL; % b
            X=x0.*(E.^2);
            Y=y0.*E;
            Y1pos= (l+Y)./2 ;
            Y1neg= (l-Y)./2 ;
            v=X./kaa(i);
            X1=v./2;    
            a1=fcs(sqrt(2./(pi.*X1)) .*  Y1pos);
            a2=fcs(sqrt(2./(pi.*X1)) .*  Y1neg);
            a3=a1/2;
            a4=a2/2;
            a5=real(a3)+real(a4)+imag(a3)+imag(a4);
            a6=-real(a3)-real(a4)+imag(a3)+imag(a4);
            At1=sqrt(a5.^2+a6.^2);

            p11= ro.*W04.*At1  .*   ((2.^(5./2) .* (c.^3) .*  ((tiempos(i)).^(1/2)))  ./  (h.* pi.^(1./2) .* w.^(3./2) .*x0 ) );
            p22= (1 - (x0./ (c.*(tiempos(i)))).^2).^(1./4);
            p33= sin( kaa(i).* b  );
            p44= sin ( omegas(i) .* T   );

            Ataux=p11.*p22.*p33.*p44;

            min1=(abs(abs(presiones(i))-abs(Ataux(:))));
            loc1=find(min1==min(min1));
            LLL=[LLL, LL(loc1)]; %FIND THE L WITH THE SMALLEST ERROR FOR EVERY K
        end

        L=sum(LLL)/length(LLL);

        %% W0
        frec=1; % Frequency for the signals that i generate to calculate W0
        W00=W_range;

        t=[r_dist/c+1 : 1/frec : t_length+r_dist/c+1];
        maxpres=[];
        for jj =1:length(W00)
            k1=(w.*x0)  ./  ((c.^2) .* (t) .* sqrt(1-   ((x0./  (c.*(t))).^2))); % Wave number
            omega=(w ./ sqrt(1- (x0./(c.*t)).^2 ) ); % Frequency

            E=b/L;
            l=E*L; % b
            X=x0*(E^2);
            Y=y0*E;
            Y1pos= (l+Y)/2 ;
            Y1neg= (l-Y)/2 ;
            v=X./k1;
            X1=v./2;    
            a1=fcs(sqrt(2./(pi.*X1)) .*  Y1pos); 
            a2=fcs(sqrt(2./(pi.*X1)) .*  Y1neg);
            a3=a1/2;
            a4=a2/2;
            a5=real(a3)+real(a4)+imag(a3)+imag(a4);
            a6=-real(a3)-real(a4)+imag(a3)+imag(a4);
            At=sqrt(a5.^2+a6.^2); % Envelope

            p1= ro.*W00(jj).*At  .*   ((2.^(5./2) .* (c.^3) .*  ((t).^(1/2)))  ./  (h.* pi.^(1./2) .* w.^(3./2) .*x0 ) );
            p2= (1 - (x0./ (c.*(t))).^2).^(1./4);
            p3= sin( k1.* b  );
            p4= sin ( omega .* T   );
            aux= p1.*p2.*p3.*p4;

            maxpres=[maxpres, max(abs(aux))];
        end

        difer=abs(maxpres-max(pks2));
        difer2=W00(find(difer==min(abs(maxpres-max(pks2)))));
        if length(difer2)==1
            W02=[ W00(find(difer==min(abs(maxpres-max(pks2)))))];
        else
            W02=[ W03];
        end

        bsol=[bsol, b];
        Tsol=[Tsol, T];
        W00sol=[W00sol, W02];
        Lsol=[Lsol, L];
        W0prue=[W0prue, W03];
        W0prue2=[W0prue2, W04];

    end
    
    W0S=[W0S; W0prue2];
    TS=[TS; Tsol];
    LS=[LS; Lsol];
    BS=[BS; bsol];
    
end
toc

%%  CHARTS FOR THE SOLUTIONS

LH=L_range; TH=T_range; bH=b_range;

W0_final=[];
for i=1:size(ome_ga,1)    
    W0_final=[W0_final, W0S(i,:)];
end

W0H=(round(min(W0_final),3)  : 0.02  :  round(max(W0_final),3));

figure(6)
subplot(2,2,1)
xbins=W0H(1:end-1)+((W0H(2)-W0H(1))/2);
hist(W0_final,xbins) %change
xlabel('W_0 [m/s]')
%xlabel('A*R^(2/3) (m^(8/3))', 'Interpreter', 'none')
ylabel('Number of solutions')

yl1=[0 max(hist(W0_final,xbins))];
vl=[mean(W0_final) mean(W0_final)];
hold on
plot(vl,yl1,'r')
title(['Mean value is ', num2str(vl(1)), ' m/s'])

a=findobj(gcf); % get the handles associated with the current figure
allaxes=findall(a,'Type','axes');
alllines=findall(a,'Type','line');
alltext=findall(a,'Type','text');
set(allaxes,'FontName','Arial','LineWidth',1.2,...
    'FontSize',11);
set(alllines,'Linewidth',1);
set(alltext,'FontName','Arial','FontSize',16);

    
T_final=[];
for i=1:size(ome_ga,1)    
    T_final=[T_final, TS(i,:)];
    %W0mean=mean(W0S(i,:))
end

subplot(2,2,2)
xbins=TH(1:end-1)+((TH(2)-TH(1))/2);
hist(T_final,xbins) %change
xlabel('T [s]')
%ylabel('Number of solutions')

yl1=[0 max(hist(T_final,xbins))];
vl=[mean(T_final) mean(T_final)];
hold on
plot(vl,yl1,'r')
title(['Mean value is ', num2str(vl(1)), ' s'])


a=findobj(gcf); % get the handles associated with the current figure
allaxes=findall(a,'Type','axes');
alllines=findall(a,'Type','line');
alltext=findall(a,'Type','text');
set(allaxes,'FontName','Arial','LineWidth',1.2,...
    'FontSize',11);
set(alllines,'Linewidth',1);
set(alltext,'FontName','Arial','FontSize',16);

L_final=[];
for i=1:size(ome_ga,1)    
    L_final=[L_final, LS(i,:)];
end

subplot(2,2,3)
xbins=LH(1:end-1)+((LH(2)-LH(1))/2);
hist(L_final,xbins) %change
xlabel('L [m]')
ylabel('Number of solutions')

yl1=[0 max(hist(L_final,xbins))];
vl=[mean(L_final) mean(L_final)];
hold on
plot(vl,yl1,'r')
title(['Mean value is ', num2str(vl(1)), ' m'])

a=findobj(gcf); % get the handles associated with the current figure
allaxes=findall(a,'Type','axes');
alllines=findall(a,'Type','line');
alltext=findall(a,'Type','text');
set(allaxes,'FontName','Arial','LineWidth',1.2,...
    'FontSize',11);
set(alllines,'Linewidth',1);
set(alltext,'FontName','Arial','FontSize',16);

B_final=[];
for i=1:size(ome_ga,1)    
    B_final=[B_final, BS(i,:)];
end

subplot(2,2,4)
xbins=bH(1:end-1)+((bH(2)-bH(1))/2);
hist(B_final,xbins) %change
xlabel('b [m]')
%ylabel('Number of solutions')

yl1=[0 max(hist(B_final,xbins))];
vl=[mean(B_final) mean(B_final)];
hold on
plot(vl,yl1,'r')
title(['Mean value is ',num2str(vl(1)), ' m'])
%legend('Solutions', 'Mean value')

a=findobj(gcf); % get the handles associated with the current figure
allaxes=findall(a,'Type','axes');
alllines=findall(a,'Type','line');
alltext=findall(a,'Type','text');
set(allaxes,'FontName','Arial','LineWidth',1.2,...
    'FontSize',11);
set(alllines,'Linewidth',1);
set(alltext,'FontName','Arial','FontSize',16);