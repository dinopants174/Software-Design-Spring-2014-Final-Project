%function [Tb, Mb, Tc, Mc] = twoD(mD, mFuel, launchAngle)
function twoD(mD, mFuel,launchAngle) 
% TO CHANGE: (currently in parameters)
    %mD = -30; % dm/dt - negative rocket fuel burn rate, kg/s
    %mFuel = 1000; % kg
  
    % CONSTANT:
    G = 6.67e-11; % m^3/(kg s)
    mEmpty = 100; % kg - empty mass
    u = 3000; % m/s, in negative jhat dir
    density = 1.225; % kg/m^3
    A = pi*0.2^2; % cross-sectional area of rocket
    Cd = 0.7; % coefficient of drag
    mE = 5.97e24; % kg, mass of Earth
    rE = 6378e3; % m, radius of Earth
    
    boostOptions = odeset('Events', @boostEvent);
    function [value, isterminal, direction] = boostEvent(t, M)
        value = M(5)-100; % mass is initial mass
        isterminal = 1; % end ode
        direction = -1; % moving from + to -
    end

    [Tb, Mb] = ode45(@boostPhase, 0:10:-mFuel/mD, [rE, cos(launchAngle), 0, sin(launchAngle)])%, boostOptions);
    disp('done with first ode')
    function res = boostPhase(t, V)
        %%%VECTORIZED -- does not work
        r = [V(1), V(3)];
        v = [V(2), V(4)];
        m = mEmpty+mFuel+mD*t
        rhat = r/norm(r);
        vhat = v/norm(v);
        
        thrustC = u*mD*vhat;
        dragC = -norm(v)^2*density*Cd*A*vhat/2;
        gravC = -G*mE*m*rhat/norm(r)^2;
        calc = (thrustC + dragC + gravC)/m;
        xDD = calc(1);
        yDD = calc(2);
%         
        res = [V(2); xDD; V(4); yDD];
    end
        
        %%% RECALCULATION
%         x = V(1);
%         xD = V(2);
%         y = V(3);
%         yD = V(4);
%         m = V(5);
%         
%         thrustC = u*mD/sqrt(x^2+y^2);
%         dragC = density*Cd*A*sqrt(xD^2+yD^2)/2;
%         gravC = G*mE*m/((x^2+y^2)^(3/2));
%         
%         xDD = 1/m*(thrustC*x - dragC*xD - gravC*x);
%         yDD = -1/m*(thrustC*y - dragC*yD - gravC*y);
%         
%         res = [xD; xDD; yD; yDD; mD];
%     end
% 
%     coastOptions = odeset('Events', @coastEvent);
%     function [value, isterminal, direction] = coastEvent(t, M)
%         value = abs(sqrt(M(1)^2+M(3)^2))-rE;
%         isterminal = 1;
%         direction = -1;
%     end
% 
%     x0 = Mb(end,1);
%     xD0 = Mb(end,2);
%     y0 = Mb(end,3);
%     yD0 = Mb(end,4);
%     
%     [Tc, Mc] = ode45(@coastPhase, [0:1:300000], [x0, xD0, y0, yD0], coastOptions);
%     function res = coastPhase(t, V)
%         x = V(1);
%         xD = V(2);
%         y = V(3);
%         yD = V(4);
%         xDD = G*mE*x/(x^2+y^2)^(3/2);
%         yDD = G*mE*y/(x^2+y^2)^(3/2);
%         res = [xD; xDD; yD; yDD];
%     end

    hold all
    plot(Mb(:,1), Mb(:,3), 'r', 'linewidth', 2)
   % plot(Mc(:,1), Mc(:,3), 'b', 'linewidth', 2)
    ang=pi/2:0.01:5*pi/2;
    xp=rE*cos(ang);
    yp=rE*sin(ang);
    plot(xp, yp, 'k');
    axis square
end