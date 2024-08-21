%% UAV Simulation
function [t,y] = IntegrationPsiTaliroEMA_HA(tI,tP,EMA)
    takenOff = 0;
    startIndex = 0;
    % Define parameters
    fs = 5; % Sampling frequency in Hz
    Vg = 14; % initial battery state
    tI = tI*fs;
    tP = tP*fs;
    t_start = 0; % Start time in seconds
    t_end = 450; % End time in seconds
    pulse_start_time = 0; % Pulse starts at 0 seconds
    pulse_end_time = 5; % Pulse ends at 5 seconds
    SNR = 0.1;
    SPower = 3;
    epsilon = 10; % determining how close the altitude is to the set point as a percentage
    % Generate the time vector
    t = t_start:1/fs:t_end;
    
    % Generate the pulse signal using rectangularPulse
    pulse_signal = zeros(1,size(t,2));%rectangularPulse(0, 2, t) + 0.5*rectangularPulse(2, 4, t)+0.9*rectangularPulse(4, 6, t)+0.6*rectangularPulse(6, 8, t)+0.3*rectangularPulse(8, 10, t);
    
    U = [pulse_signal; pulse_signal; pulse_signal; pulse_signal];
    tic
    Y = zeros(16,size(pulse_signal,2));
    SP = 50;
    pulse_signal = zeros(1,size(pulse_signal,2));
    K_p = 1/10;
    K_d = 1/4;
    K_I = 1/100;
    for i = 2:size(pulse_signal,2)
        
        %if(i > 300 && i < 500)
        if(i > tI && i < tI+tP)
            adjY =  awgn(Y(3,i-1),EMA,SPower,'linear');
        else
            adjY = Y(3,i-1);
        end
        err(i-1) = SP - adjY;
        if(i == 2)
            pulse_signal(i-1) = K_p * err(i-1);
        else
            pulse_signal(i-1) = K_p * err(i-1) + K_d * fs * (err(i-1)-err(i-2)) + K_I*(sum(err(1:i-1)))/fs;
        
        end
        Vg = battery_state(Vg,fs);

        if(Vg < 0.8*14)
            SP = 20;
        end
        if(pulse_signal(i-1) > 1.2)
            pulse_signal(i-1) = 1.2;
    
        end
    
        
    
        if(pulse_signal(i-1) < 0)
            pulse_signal(i-1) = 0;
        end

        if(i > 300)
            U(:,i-1) = [pulse_signal(i-1); pulse_signal(i-1); pulse_signal(i-1); pulse_signal(i-1)];
        else
            U(:,i-1) = [pulse_signal(i-1); pulse_signal(i-1); pulse_signal(i-1); pulse_signal(i-1)];
        end
        

        Y(:,i) = Y(:,i-1) + (1/fs)*stateDiff(Y(:,i-1),U(:,i-1),fs,i-1,tI,tP, EMA,Vg);
        if(Y(3,i) > 0)
            takenOff = 1;
        end
        %% Apply boundary conditions
        

        if(Y(3,i) < 0 &&  (Y(6,i)) < 0 && takenOff)
            
            Y(:,i:end) = zeros(16,size(Y,2)-i+1);
            y = Y;
            Vg
            return
            
        end

        if(Y(3,i) < SP*(1+epsilon/100) && Y(3,i) > SP*(1-epsilon/100) && startIndex == 0)
            startIndex = i;
        end

        if(~takenOff && Y(3,i) < 0)
            Y(3,i) = 0;
            Y(6,i) = 0;
            
        end

        
%         for j = 10:12
%             if(abs(Y(j,i)) > 10000)
%                 Y(j,i) = 10000*Y(j,i)/abs(Y(j,i));
%             end
%         end

%         for j = 13:16
%             if(abs(Y(j,i)) > 100)
%                 Y(j,i) = 100*Y(j,i)/abs(Y(j,i));
%             end
% 
%         end
    
        
    
    end

    y = Y(:,startIndex+1:end);
    Vg
    %plot(Y(3,:))
end
% figure
% hold on
% %
% for i = 1:800
%     subplot(1,2,1)
% plot(YY(1,1:i),YY(2,1:i),'o')
% hold on
% stem(0,0)
% xlim([-40 40])
% ylim([-20 20])
% grid on
% 
% xlabel('X axis')
% ylabel('Y axis')
% 
% 
% subplot(1,2,2)
% plot(YY(3,1:i))
% hold on
% stem( 550,100 )
% xlim([0 800])
% ylim([0 170])
% pause(0.000008)
% end

function y = stateDiff(X,U,fs, i, tI, tP, EMA, Vg)
    m = 0.65; % kg
    g = 9.806; % m/s^2
    l = 0.232; % m
    rho = 1.293; % kg/m^3
    d = 1.5*10^(-4); % Nm s^2
    Jx = 7.5*10^(-3); % Nm s^2 / rad
    Jy = 7.5*10^(-3); % Nm s^2 / rad
    Jz = 1.3*10^(-2); % Nm s^2 / rad
    Kt = 10*10^(-15); % Nm s /rad
    Kr = 10*10^(-15); % Nm s / rad
    CT = 0.055;
    CQ = 0.024;
    R = 0.15; % m
    A = pi*R^2;
    Vg = Vg; % V
    Rm = 0.036; % Ohm
    Jm = 4*10^(-4); % kg m^2
    Jr = 6*10^(-3); % kg m^2
    km = 0.01433; % kg m / A
    
    VgMat = [Vg;Vg;Vg;Vg];
%     if((i > tI && i <tI+tP) )
%         g = gA;
%     else
%         g = 9.806;
%     end

    
    
    Rt = [cos(X(8))*cos(X(9)) (sin(X(7))*sin(X(8))*cos(X(9))-cos(X(7))*sin(X(9))) (cos(X(7))*sin(X(8))*cos(X(9))+sin(X(7))*sin(X(9))); cos(X(8))*sin(X(9)) (sin(X(7))*sin(X(8))*sin(X(9))-cos(X(7))*cos(X(9))) (cos(X(7))*sin(X(8))*sin(X(9))+sin(X(7))*cos(X(9))); sin(X(8)) -(sin(X(7))*cos(X(8))) -(cos(X(7))*cos(X(8)))];
    
    KtM = Kt*eye(3);
    
    KrM = Kr*eye(3);

    y = zeros(16,1);
    
    y(1:3) = X(4:6);
    
    
    g0 = [(cos(X(7))*sin(X(8))*cos(X(9))+sin(X(7))*sin(X(9)))*rho*CT*A*R^2*(X(13)^2+X(14)^2+X(15)^2+X(16)^2); (cos(X(7))*sin(X(8))*sin(X(9))-sin(X(7))*cos(X(9)))*rho*CT*A*R^2*(X(13)^2+X(14)^2+X(15)^2+X(16)^2); cos(X(7))*cos(X(8))*rho*CT*A*R^2*(X(13)^2+X(14)^2+X(15)^2+X(16)^2)];
    
    y(4:6) = -[0; 0; g] - (1/m)*Rt*KtM*(Rt')*(X(4:6)) + g0;
    
    y(7:9) = X(10:12);
    
    %Jx = 0.4*Msphere*r^2 + 2*(l^2)*Mrotor;
    
    %Jy = 0.4*Msphere*r^2 + 2*(l^2)*Mrotor;
    
    %Jz = 0.4*Msphere*r^2 + 4*(l^2)*Mrotor;
    
    J = [Jx 0 0; 0 Jy 0; 0 0 Jz];
    
    Rr = [1 0 -sin(X(8)); 0 cos(X(7)) sin(X(7))*cos(X(8)); 0 -sin(X(7)) cos(X(7))*cos(X(8))];
    
    delRr_delPhi = [0 0 0; 0 -sin(X(7)) cos(X(7))*cos(X(8)); 0 -cos(X(7)) -sin(X(7))*cos(X(8))];
    
    delRr_delTheta = [0 0 -cos(X(8)); 0 0 -sin(X(7))*sin(X(8)); 0 0 -cos(X(7))*sin(X(8))];
    
    g1 = inv(J*Rr)*[(l/sqrt(2))*rho*CT*A*R^2*(X(13)^2-X(14)^2-X(15)^2+X(16)^2); (l/sqrt(2))*rho*CT*A*R^2*(-X(13)^2-X(14)^2+X(15)^2+X(16)^2); rho*CQ*A*R^3*(-X(13)^2+X(14)^2-X(15)^2 + X(16)^2)];
    
    y(10:12) = -inv(J*Rr)*(cross(Rr*X(10:12),J*Rr*X(10:12))+Kr*Rr*X(10:12)+J*(delRr_delPhi*X(10)+delRr_delTheta*X(11))*X(10:12))+g1;
    
    y(13:16) = (-km^2/(Rm*(Jm+Jr)))*X(13:16) - (d/(Jm+Jr))*(X(13:16).*(X(13:16))) + ((km*VgMat)/(Rm*(Jm+Jr))).*(U);



end


function noisy_signal = add_white_noise(signal, Sp, SNR)
    % ADD_WHITE_NOISE Adds white noise to a signal
    %   signal - Input signal
    %   Sp - Signal power
    %   SNR - Signal to Noise Ratio in dB

    % Calculate the signal power
    signal_power = Sp;
    
    % Calculate the noise power from the SNR
    SNR_linear = 10^(SNR / 10);
    noise_power = signal_power / SNR_linear;
    
    % Generate white noise with the calculated noise power
    noise = sqrt(noise_power) * randn(size(signal));
    
    % Add the noise to the signal
    noisy_signal = signal + noise;
end

function y = battery_state(V,fs)
charge_rate = 0.001;
y = V*(1-charge_rate/fs);
end

