function [t_A, P, T,Omega] = semi_auto_get_frequency_func(t,w)

        tol = 40; % wiggle room in points
        
        wg = warndlg('Zoom and then click OK when ready', 'Zoom');
        wg.Position(1) = wg.Position(1)-500;
        waitfor(wg);
   
        %disp('Select centre point (or LEFT CLICK to interrupt)');
        %[xc,yc] = ginput(1);
        %plot(xc,yc,'oc');

        disp('Select left point');
        
        xp = [];
        while(isempty(xp))
            [xp,yp] = ginput(1);
            [~, xidx] = min(abs(t-xp));
        
            if(yp >= 0)
                [yp, xp] = findpeaks(w(xidx-tol:xidx+tol),t(xidx-tol:xidx+tol), 'NPeaks',1,'SortStr','descend');
            else
                [yp, xp] = findpeaks(-w(xidx-tol:xidx+tol),t(xidx-tol:xidx+tol), 'NPeaks',1,'SortStr','descend');
                yp = -yp;
            end
        end
            
        plot(xp,yp,'>m');

        disp('Select right point');
        
        xf = [];
        while(isempty(xf))
            [xf,yf] = ginput(1);
            [~, xidx] = min(abs(t-xf));
            if(yf >= 0)
                [yf, xf] = findpeaks(w(xidx-tol:xidx+tol),t(xidx-tol:xidx+tol), 'NPeaks',1,'SortStr','descend');
            else
                [yf, xf] = findpeaks(-w(xidx-tol:xidx+tol),t(xidx-tol:xidx+tol), 'NPeaks',1,'SortStr','descend');
                yf = -yf;
            end
            plot(xf,yf,'<k');
        end
        
       % t_A = (xf+xp)/2; 
        T = (xf-xp)/2; 
        Omega = 2 * pi / T;
       % P = (abs(yf)+abs(yp))/2;
        
         disp('Select MIDDLE point');
        
        xm = [];
        while(isempty(xm))
            [xm,ym] = ginput(1);
            [~, xidx] = min(abs(t-xm));
            if(ym >= 0)
                [ym, xm] = findpeaks(w(xidx-tol:xidx+tol),t(xidx-tol:xidx+tol), 'NPeaks',1,'SortStr','descend');
            else
                [ym, xm] = findpeaks(-w(xidx-tol:xidx+tol),t(xidx-tol:xidx+tol), 'NPeaks',1,'SortStr','descend');
                yf = -yf;
            end
            plot(xm,ym,'<b');
        end
        t_A = xm;
        P = abs(ym);

end