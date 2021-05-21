clear all
clc
close all
format longEng


gunzip('C:\Users\User\Desktop\INVERSE MODEL 2021\1.wav.gz')
[y1,Fs] = audioread('C:\Users\User\Desktop\INVERSE MODEL 2021\1.wav');

gunzip('C:\Users\User\Desktop\INVERSE MODEL 2021\2.wav.gz')
[y2,Fs] = audioread('C:\Users\User\Desktop\INVERSE MODEL 2021\2.wav');

gunzip('C:\Users\User\Desktop\INVERSE MODEL 2021\3.wav.gz')
[y3,Fs] = audioread('C:\Users\User\Desktop\INVERSE MODEL 2021\3.wav');

y=[y1; y2; y3];
aux=1:length(y);
t=aux./Fs;

% y=y(1720529:2205000);
% t=t(1720529:2205000);

len=40000;

s = spectrogram(y,len,[],len,Fs);


% elimino la media
y=y-mean(y);

%% CALIBRATION

aa=[];
aa=importdata('calibration1.txt');
bb=aa.data;
bm=mean(bb); %dB Re counts/uPa
sens=10^((bm)/20)*1000000;  %counts/Pa

%Convert to counts  max range is 2^(24-1)  (VERIFY)
yy=y*2^(24-1);
yyy=yy/sens;
y2=y;
y=yyy;

figure(2)
subplot(2,1,1)
plot(t,y)
axis([0 900 -10 10])
xlabel('Time [s]')
ylabel('Pressure [Pa]')

subplot(2,1,2)
[~,F,T,P] =spectrogram(y2,len,[],len,Fs,'yaxis');
imagesc(T, F, 10*log10(P+eps)); % add eps like pspectrogram does
axis xy
ylabel('Frequency (Hz)')
xlabel('Time (s)')
h = colorbar;
h.Label.String = 'Power/frequency (dB/Hz)';
%axis([400 600 0 10])

%% now i try to obtain the orientation and frequency distribution
% I assume i know the frequency range from the spectrogram
% in this case 3-4 Hz
frec=300;
h=2662;
c=1500;
w=pi*c/(2*h); % Frequency for the FIRST MODE

% possible x0 and y0 that together make 9000 km
x00=0:50*10^3:8900*10^3;
y00=sqrt((9000*10^3).^2-(x00).^2);
% we also know the time of the signal is not longer than 200 secs
t0=0-(sqrt(y00(1).^2+x00(1).^2) ./ c);
OM=[];
for i=1:length(x00)
    
    t=[-t0  -t0+120]; 
    omega=(w ./ sqrt(1- (x00(i)./(c.*t)).^2 ) ); % Frequency

    OM=[OM; omega];
end

pos=[];
% Verify that the frequencies are in the required range
for i=1:length(x00)
       
    if OM(i, 1)<10 && OM(i, 2)>2.8
        pos=[pos, i];
    end
end

%%IMPORTANT FIND A WAY TO CHOSE ONE OF THOSE 2 POSSIBILITIES
%THE TWO VALID POSSIBILITIES
x0=x00(pos(1));
y0=y00(pos(1));
t=[-t0 : 1/frec : -t0+120];
omega=[];
omega(1,:)=(w ./ sqrt(1- (x0./(c.*t)).^2 ) );
%figure(4)
hold on
plot(t-(t(1))+430,omega(1,:),'k','Linewidth',1)

x0=x00(pos(2));
y0=y00(pos(2));
% THIS IS THE FREQUENCY I WANT TO USE
omega(2,:)=(w ./ sqrt(1- (x0./(c.*t)).^2 ) );
%figure(5)
hold on
plot(t-(t(1))+430,omega(2,:),'k','Linewidth',1)
axis([400 600 0 10])


x0=x00(pos(3));
y0=y00(pos(3));
% THIS IS THE FREQUENCY I WANT TO USE
omega(3,:)=(w ./ sqrt(1- (x0./(c.*t)).^2 ) );
%figure(6)
hold on
plot(t-(t(1))+430,omega(3,:),'k','Linewidth',1)

x0=x00(pos(4));
y0=y00(pos(4));
% THIS IS THE FREQUENCY I WANT TO USE
omega(4,:)=(w ./ sqrt(1- (x0./(c.*t)).^2 ) );
%figure(7)
hold on
plot(t-(t(1))+430,omega(4,:),'k','Linewidth',1)


%% HERE I CHOSE THE START AND END OF MY SIGNAL
% first i compute the power
aux=1:length(y);
t=aux./Fs;
w=y;
power=[];
for i =1:length(t)/4000
    aux2= y(1 +4000*(i-1) : 4000+4000*(i-1)).* y(1 +4000*(i-1) : 4000+4000*(i-1));
    power=[power, sum(aux2)];
end

buttonSelections = 1;
count2=0;

while buttonSelections == 1

    figure(1)
    subplot(2,1,1)
    plot(t,w)
    xlabel('Time (s)')
    ylabel('Pressure (Pa)')
    hold on
    
    subplot(2,1,2)
    plot(power)
    xlabel('Time (s)')
    ylabel('Power [RMS]')
    
    x0p=300;
    y0p=0;
    widthp=700;
    heightp=400;
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
    
    %-----------------------------------------------------------------------
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

figure(3)
plot(t(locc:locc2), w(locc:locc2))

t1=t(locc:locc2);
y1=w(locc:locc2);

%% Tracking the envelope, think about automated and efficient way to find the
% optimal iterations
[pks,locs]=findpeaks(y1);
[pks,locs2]=findpeaks(pks);
[pks,locs3]=findpeaks(pks);
[pks,locs4]=findpeaks(pks);
[pks,locs5]=findpeaks(pks);
[pks,locs6]=findpeaks(pks);
[pks,locs7]=findpeaks(pks);

tims2=t1(locs(locs2(locs3(locs4(locs5(locs6(locs7)))))));
pks2=pks;

hold on
plot(tims2, pks2, 'o')

%% now i check how scattered are the obtained points in time

periods=tims2(2:end)-tims2(1:end-1);
frequencies=1./periods;
figure(4)
plot(frequencies)
xlabel('Time (s)')
ylabel('Frequency [Hz]')
% I verify all frequencies are below the wanted frequency so i can force 
% the obtained distribution

% times associated to omegas
time1=(1:length(omega))/ frec;

% times associated to pks
time2=tims2-t(locc);

% find the location of the related omegas of the traced envelop

aa=[];
for i=1:length(time2)    
    aa(i,:)=abs(time2(i)-time1); 
end
locc=[];
for i=1:length(time2)     
    locc=[locc, find(aa(i,:)==min(aa(i,:)))];
end

omeg(1,:)=omega(1,locc);
omeg(2,:)=omega(2,locc);
omeg(3,:)=omega(3,locc);
omeg(4,:)=omega(4,locc);


% this are the points that I am using
figure(5)
subplot(2,1,1)
plot(t1-(t1(1)),y1)
hold on 
plot(time2, pks2, 'o')

subplot(2,1,2)
plot(time1, omega)
hold on
plot(time2, omeg,'o')

finaltims=time2;
finalpks=pks2;

%% NOW I KNOW THE ORIENTATION, LOCATION AND FREQUENCY DISTRIBUTION
% I CAN APPLY THE INVERSE MODEL

prompt = 'How many solutions do you want?';
nnn = input(prompt)

prompt = 'How many points per solution should be used?';
npts = input(prompt)
tic
%nnn=30; % NUMBER OF SETS OF SOLUTIONS THAT I WANT
%npts=5; % NUMBER OF POINTS FROM THE SIGNAL THAT I WANT FOR COMPUTING EACH SET OF SOLUTIONS
nlocs=[];
% Here nnn number of non repeated combinations from the available points
% are made they are also sorted getting the latest points at the beggining
% since those are the selected to compute the location and eruption time

tic
%% I obtain solutions for all possible 

W0S=[];
TS=[];
LS=[];
BS=[];
for jj=1:4
    
    omegs=omeg(jj,:);
    
    for i=1:nnn
       %I have to make this that they are never close------------------------------------------------------

       nlocs(i,:)=sort(randperm(length(finaltims),npts)) ;

       tiempo1=finaltims(nlocs(i,1));
       tiempo2=finaltims(nlocs(i,2));

       while abs(abs(tiempo1)-abs(tiempo2))<10 % TUNING PARAMETER IMPORTANT DISTANCE IN SECONDS BUT OPNLY FROM THE 2 FIRST POINTS, THEY COMPUTE LOCATION
            nlocs(i,:)=sort(randperm(length(finaltims),npts)) ;
            tiempo1=finaltims(nlocs(i,1));
            tiempo2=finaltims(nlocs(i,2));
       end

    end

    
    %% TEORIA INVERSA
    %%
    c=1500; ro=1000;
    w=pi*c/(2*h); X00=[]; Y00=[]; T00=[];%% REVIEW DESCRIPTION OF THE PRESSURE THERE ARE 2 DIFFERENT WAYS
    q=0; bsol=[]; Tsol=[]; W00sol=[]; Lsol=[]; pruebaT=[]; iteration=0; W0prue=[];
    t02=sqrt(x0^2+y0^2)/c;
    for j=1:length(nlocs)
        % IMPORTANT HERE I EXTRACT T0 CAUSE MY TIMES ARE FROM THE DIRECT
        % PROBLEM
        iteration=iteration+1
        tiempos=finaltims(nlocs(j,:));%+t0; % CHECK WITH T0---------------------------------------------------------------------------
        presiones=finalpks(nlocs(j,:));  
        omegas=omegs(nlocs(j,:));

            kaa=omegas.*x0./(c^2.*(tiempos-t02)); % wave number I ADD MY T02 cause now i know the eruption time

            %% b

            n=[1:20]; % I want the first 20 solutions for each point in the analytical solution of 1=sin(kb)
            b2=[];
            for i = 1:length(tiempos)
                b2(i,:)= ((n-1/2)*pi)/kaa(i);
            end

            % Those are 2 tuning parameters the step and the starting point
            stepb=1*10^3;
            bb=(1.5*10^3:stepb:22.5*10^3); 
            histo=[];

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

            % The convergence points is found
            B=find(histo==max(histo));
            b=bb(B)+0.5*10^3;

            % If more than a solution is found they are averaged
            if length(b)>1
                baux=sum(b)/length(b);
                b=baux;
            end

            %% HERE T IS CALCULATED EVALUATION ALL POSSIBLE SOLUTIONS FOR EACH L IN A RANGE THAT WE ESTABLISH
            % some tuning paremeters such as maximum error that we want to admit in
            % our possible solution
            maxerror=0.03; % maximun error that can have the solution 
            maxthreshT=20; %Range of possible T
            minthreshT=2;
            maxthreshL=1000*10^3; %Range of possible L
            minthreshL=200*10^3; 
            stepL=2000;
            stepT=0.2;

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

            % Those are 2 tuning parameters the step and the starting point
            stepT=0.5;
            Ts=(1.25:stepT:13.25);
            histo=[];
            xx=[];
            rr=[];
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
            val=[];
            ind=[];
            [val ind] = sort(sum2,'descend');

            % I ONLY CONSIDER THE SOLUTIONS WITH LESS THAN 40% DIFFERENCE WITH
            % THE HIGHEST CONVERGENCE OF SOLUTIONS
            top2=[val(1)];
            TTS=[Ts(ind(1))]+stepT/2;
            if length(val)>1
                if (val(1)-val(2))/val(1)*100<20
                    top2=[val(1),val(2)];
                    TTS=[Ts(ind(1)), Ts(ind(2))]+stepT/2;
                end
                if (val(1)-val(3))/val(1)*100<20
                    top2=[val(1),val(2),val(3)];
                    TTS=[Ts(ind(1)), Ts(ind(2)), Ts(ind(3))]+stepT/2;
                end
                if (val(1)-val(4))/val(1)*100<20
                    top2=[val(1),val(2),val(3),val(4)];
                    TTS=[Ts(ind(1)), Ts(ind(2)), Ts(ind(3)), Ts(ind(4))]+stepT/2;
                end
            end


            pruebaT=[pruebaT, mean(TTS)];
            % Here the chosen possible solutions for T are evaluated again by
            % looking at the error that the generate using the function calct2
            cnr=[];
            maxerror=0.05;
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
        LL=[1*10^3:  5*10^3:  51*10^3];
        W00=[0.005 : 0.005: 0.5];
        T=Tr;

        % redefine k1, omega and t, we just need 2 points that can be modificed
        % in n1 and n2
        n1=2;
        n2=3;


        k1=[kaa(n1), kaa(n2)];
        omega=[omegas(n1), omegas(n2)];
        t=[tiempos(n1), tiempos(n2)];
        pressure=[presiones(n1), presiones(n2)];


        solucion1=[];
        solucion2=[];
        sol1=[];
        sol2=[];

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
                %% AQUI HE CAMBIADO LAS COMAS POR PUNTO Y COMA Y ABAJO IGUAL
                solucion1=[solucion1, abs(val1)];
                solucion2=[solucion2, abs(val2)];

            end
            sol1=[sol1; solucion1];
            sol2=[sol2; solucion2];
            solucion1=[];
            solucion2=[];

        end

        aa1=sum(sol1);
        aa2=sum(sol2);

        ss1=find(aa1==min(aa1));
        ss2=find(aa2==min(aa2));

        W03=(W00(ss1)+W00(ss2))/2;



        %% L

        LLL=[];
        for i=1:length(kaa)
            LL=[1*10^3:  5*10^3:  51*10^3];

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

            p11= ro.*W03.*At1  .*   ((2.^(5./2) .* (c.^3) .*  ((tiempos(i)).^(1/2)))  ./  (h.* pi.^(1./2) .* w.^(3./2) .*x0 ) );
            p22= (1 - (x0./ (c.*(tiempos(i)))).^2).^(1./4);
            p33= sin( kaa(i).* b  );
            p44= sin ( omegas(i) .* T   );

            Ataux=p11.*p22.*p33.*p44;

            min1=(abs(abs(presiones(i))-abs(Ataux(:))));
            loc1=find(min1==min(min1));

            LLL=[LLL, LL(loc1)];
        end


        L=sum(LLL)/length(LLL);

        %% W0
        frec=50; % Hydrophone measuring frequency Hz
        W00=[0.005 : 0.005: 0.5];


        t=[x0/c : 1/frec : tiempos(1)+x0/c]; 

        w=pi*c/(2*h); % Frequency for the FIRST MODE
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
            a1=fcs(sqrt(2./(pi.*X1)) .*  Y1pos); % Some Fresnel integrals are computed by fcs function
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

    end
    
    W0S=[W0S; W00sol];
    TS=[TS; Tsol];
    LS=[LS; Lsol];
    BS=[BS; bsol];
    
end
toc
%%

W0H=(0: 0.005: 0.3);
LH=(1*10^3: 5*10^3: 101*10^3);
TH=(2.25: 0.5: 20.25);
bH=(1.5*10^3 :1*10^3: 102.5*10^3);

    
for i=1:4    
    
    W0mean=mean(W0S(i,:))
    figure(9+i)
    xbins=W0H(1:end-1)+((W0H(2)-W0H(1))/2);
    hist(W0S(i,:),xbins)
    xlabel('W_0 (m/s)')
    ylabel('Number of solutions')
    hold on

    a=findobj(gcf); % get the handles associated with the current figure
    allaxes=findall(a,'Type','axes');
    alllines=findall(a,'Type','line');
    alltext=findall(a,'Type','text');
    set(allaxes,'FontName','Arial','LineWidth',1.5,...
        'FontSize',15);
    set(alllines,'Linewidth',1);
    set(alltext,'FontName','Arial','FontSize',22);
end
    %%   
    
for i=1:4
    Tmean=mean(TS(i,:))
    figure(13+i)
    xbins=TH(1:end-1)+((TH(2)-TH(1))/2);
    hist(TS(i,:),xbins)
    xlabel('T (s)')
    ylabel('Number of solutions')

    a=findobj(gcf); % get the handles associated with the current figure
    allaxes=findall(a,'Type','axes');
    alllines=findall(a,'Type','line');
    alltext=findall(a,'Type','text');
    set(allaxes,'FontName','Arial','LineWidth',1.5,...
        'FontSize',15);
    set(alllines,'Linewidth',1);
    set(alltext,'FontName','Arial','FontSize',22);
end
    %%
    
for i=1:4
    Lmean=mean(LS(i,:))
    figure(17+i)
    xbins=LH(1:end-1)+((LH(2)-LH(1))/2);
    hist(LS(i,:),xbins)
    xlabel('L (m)')
    ylabel('Number of solutions')
    hold on

    a=findobj(gcf); % get the handles associated with the current figure
    allaxes=findall(a,'Type','axes');
    alllines=findall(a,'Type','line');
    alltext=findall(a,'Type','text');
    set(allaxes,'FontName','Arial','LineWidth',1.5,...
        'FontSize',15);
    set(alllines,'Linewidth',1);
    set(alltext,'FontName','Arial','FontSize',22);

end
    %%  
    
for i=1:4
    bmean=mean(BS(i,:))
    figure(21+i)
    xbins=bH(1:end-1)+((bH(2)-bH(1))/2);
    hist(BS(i,:),xbins)
    xlabel('b (m)')
    ylabel('Number of solutions')
    hold on

    a=findobj(gcf); % get the handles associated with the current figure
    allaxes=findall(a,'Type','axes');
    alllines=findall(a,'Type','line');
    alltext=findall(a,'Type','text');
    set(allaxes,'FontName','Arial','LineWidth',1.5,...
        'FontSize',15);
    set(alllines,'Linewidth',1);
    set(alltext,'FontName','Arial','FontSize',22);
    
end
