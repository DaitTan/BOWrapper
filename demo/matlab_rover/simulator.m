
function [time,final_states] = simulator(a,b,r)

    

    states = [0.00; 0.00; 0.0; 0.0; 0.0; 0.0]; %initial states
    wheels = [0; 0; 0; 0]; %initial orientation of the wheels (only for plot)
    
    g = 9.81; %gravity acceleration
    
    
    % Select the robot type
    robot = 'EspeleoRobo';
    % robot = 'PioneerP3AT';
    
    
    if strcmp(robot,'EspeleoRobo')
        %Espeleorobo
%         a = 0.215; %foward/backward distance of the weels (from the robot's center)
%         b = 0.18; %lateral distance of the weels (from the robot's center)
%         r = 0.151; %radius of the wheels
        epsilon = 0.005; %"velocity of the maximum static friction"
        m = 27.4; %mass
        J = 0.76; %moment of inertia
        kf = 0.48; %coefficient of the kinetic friction (N/N)
        %Center of mass with respect to the body's center (x, y) and floor 
        CM = [0.0; 0.0; 0.12];
        
        %Size of the robot (x, y, z) - only for animation
        size_body = [0.55 0.35 0.12];
        
    elseif('PioneerP3AT')
        %Pioneer
        a = 0.135; %foward/backward distance of the weels (from the robot's center)
        b = 0.2; %lateral distance of the weels (from the robot's center)
        r = 0.098; %radius of the wheels
        epsilon = 0.005; %"velocity of the maximum static friction"
        J = 0.58; %moment of inertia
        m = 26.8; %mass
        kf = 0.42; %experimental; %coefficient of the kinetic friction (N)
        %Center of mass with respect to the body's center (x, y) and floor 
        CM = [0.0; 0.0; 0.15];
        
        %Size of the robot (x, y, z) - only for animation
        size_body = [0.4 0.34 0.18];
        
    else
        error('Select the robot')
    end
    
    
    %Compute the vectors that go from the robot's center of mass to the wheels
    c1 = [a; -b]-CM(1:2);
    c2 = [-a; -b]-CM(1:2);
    c3 = [-a; b]-CM(1:2);
    c4 = [a; b]-CM(1:2);
    
    %Simulaion times definitions
    T = 100;
    dt = 0.001;
    t = 0:dt:T;
    
    
    %Binary flag to plot simulated data
    DO_PLOTS = 0;
    
    %Flag to run an animation
    % 0 - no animation
    % 1 - 3D animation
    % 2 - 2D animation
    RUN_ANIMATION = 0;
    
    %Simulation speed factor
    SPEED = 4.0;


    %% Simulator of a skeed-steering robot
    
    %Load parameters
    % parameters
    
    
    
    %Initial weels rotation speeds
    Omega_1 = 0;
    Omega_2 = 0;
    Omega_3 = 0;
    Omega_4 = 0;
    
    F_last = [0; 0];
    
    %Logs initialization
    
    vel_log = zeros(4,length(t));
    
    F_tot_log = zeros(2,length(t));
    F1_log = zeros(2,length(t));
    F2_log = zeros(2,length(t));
    F3_log = zeros(2,length(t));
    F4_log = zeros(2,length(t));
    
    %Estimated torque on the wheels
    TORQUE_1 = zeros(1,length(t));
    TORQUE_2 = zeros(1,length(t));
    TORQUE_3 = zeros(1,length(t));
    TORQUE_4 = zeros(1,length(t));
    
    %Estimated power on the wheels
    POWER_1 = zeros(1,length(t));
    POWER_2 = zeros(1,length(t));
    POWER_3 = zeros(1,length(t));
    POWER_4 = zeros(1,length(t));
    
    
    
    
    %% Simulation loop
    tic
    for k = 1:1:(length(t)-1)
    
        %Get current states
        x = states(1,k);
        y = states(2,k);
        psi = states(3,k);
        vx = states(4,k);
        vy = states(5,k);
        wz = states(6,k);
        
        %Compute the wheels reference velocities
        [Omega_r,Omega_l] = open_loop_reference(t(k),a,b,r); % open loop reference
    
        %Filter command to simulate a ramp time of the wheels
        ALPHA = dt/(0.5/2);
        Omega_1 = (1-ALPHA)*Omega_1 + ALPHA*Omega_r;
        Omega_2 = (1-ALPHA)*Omega_2 + ALPHA*Omega_r;
        Omega_3 = (1-ALPHA)*Omega_3 + ALPHA*Omega_l;
        Omega_4 = (1-ALPHA)*Omega_4 + ALPHA*Omega_l;
        
        %Compute relative velocities of the contact points of the wheels with respect to the floor;
        u1 = compute_relative_vel(Omega_1,c1,vx,vy,wz,r);
        u2 = compute_relative_vel(Omega_2,c2,vx,vy,wz,r);
        u3 = compute_relative_vel(Omega_3,c3,vx,vy,wz,r);
        u4 = compute_relative_vel(Omega_4,c4,vx,vy,wz,r);
    
    %     %Compute friction forces (using the last value of F)
    %     normals = compute_normals(CM,m,g,a,b,acc);
    %     F_1 = friction_model(u1,kf,normals(1),epsilon);
    %     F_2 = friction_model(u2,kf,normals(2),epsilon);
    %     F_3 = friction_model(u3,kf,normals(3),epsilon);
    %     F_4 = friction_model(u4,kf,normals(4),epsilon);
        
        %Compute friction forces (using the linearity of friction  force has on the normal force)
        u_all = [u1,u2,u3,u4];
        [F_1, F_2, F_3, F_4, normals] = friction_joined_model(CM,m,g,a,b,kf,u_all,epsilon);
        
        
        %Compute total force
        F_tot = F_1 + F_2 + F_3 + F_4;
        
        %Compute the torque (around the center of mass and in the z axis)
        Tau_1 = c1(1)*F_1(2) - c1(2)*F_1(1);
        Tau_2 = c2(1)*F_2(2) - c2(2)*F_2(1);
        Tau_3 = c3(1)*F_3(2) - c3(2)*F_3(1);
        Tau_4 = c4(1)*F_4(2) - c4(2)*F_4(1);
        
        %Compute total torque
        Tau_tot = Tau_1 + Tau_2 + Tau_3 + Tau_4;
    
        % Compute the derivative of the states
        states_dot = [cos(psi)*vx-sin(psi)*vy;
                      sin(psi)*vx+cos(psi)*vy;
                      wz;
                      F_tot(1)/m + wz*vy;
                      F_tot(2)/m - wz*vx;
                      Tau_tot/J];
    
        %Save current force
        F_last = F_tot;
        
        %System integration
        states(:,k+1) = states(:,k) + states_dot*dt;
        
        %Compute the new position of the wheels (only for animation)
        wheels(:,k+1) = wheels(:,k) + [Omega_1;Omega_2;Omega_3;Omega_4]*dt;
        
        %Logs
        
        %velocities
        vel_log(:,k+1) = [norm(u1); norm(u2); norm(u3); norm(u4)];
        
        %Forces
        F_tot_log(:,k+1) = F_tot;
        F1_log(:,k+1) = F_1;
        F2_log(:,k+1) = F_2;
        F3_log(:,k+1) = F_3;
        F4_log(:,k+1) = F_4;
    
        %Normals
        N_log(k+1,:) = normals;
        
        %Wheels' torques
        TORQUE_1(k+1) = r*F_1'*[1; 0];
        TORQUE_2(k+1) = r*F_2'*[1; 0];
        TORQUE_3(k+1) = r*F_3'*[1; 0];
        TORQUE_4(k+1) = r*F_4'*[1; 0];
        %Wheels' powers
        POWER_1(k+1) = TORQUE_1(k+1) * Omega_1;
        POWER_2(k+1) = TORQUE_2(k+1) * Omega_2;
        POWER_3(k+1) = TORQUE_3(k+1) * Omega_3;
        POWER_4(k+1) = TORQUE_4(k+1) * Omega_4;
        
    end
    toc
    
    %Compute the position of the robot's center
    theta_cm = atan2(CM(2),CM(1));
    geo_center = [states(1,:)-norm(CM(1:2))*cos(states(3,:)+theta_cm); states(2,:)-norm(CM(1:2))*sin(states(3,:)+theta_cm)];
    
    
    
    %% Some simple plots
    
    if(DO_PLOTS==1)
    
        figure(1)
        subplot(2,2,1)
        axis equal
        plot(states(1,:),states(2,:),'b')
        hold on
        plot(geo_center(1,:),geo_center(2,:),'r')
        hold off
        % plot(states(1,:)+0.15*cos(states(3,:)),states(2,:)+0.15*sin(states(3,:)))
        xlim([min(states(1,:)) max(states(1,:))]+[-1 1]*0.3)
        ylim([min(states(2,:)) max(states(2,:))]+[-1 1]*0.3)
        axis equal
        grid on
        title('XY (center of mass)')
        legend('center mass','center body')
    
        subplot(2,2,2)
        plot(t, states(3,:),'b')
        grid on
        title('yaw')
    
        subplot(2,2,3)
        v_world = [];
        for k = 1:1:length(t)
            v_world(:,k) = [cos(states(3,k)) -sin(states(3,k)); sin(states(3,k)) cos(states(3,k))]*states(4:5,k);
        end
        plot(t, states(4,:),'r')
        hold on
        plot(t, states(5,:),'g')
        plot(t, sqrt(states(4,:).^2+states(5,:).^2),'b')
        hold off
        grid on
        title('Body vels')
        legend('v_x','v_y','norm')
    
        subplot(2,2,4)
        plot(t, states(6,:),'b')
        grid on
        title('\omega_z')
        % hold on
        % plot([t(1) t(end)], 1*[1 1]*(1-exp(-1)),'r')
        % hold off
    
    
        % figure(2)
        % subplot(2,1,1)
        % % plot(t, bias_v,'b')
        % xlim([t(1) t(end)])
        % ylim([-0.1 2])
        % title('bias v')
        % grid on
        % 
        % subplot(2,1,2)
        % % plot(t, average_filter(bias_w,100,0),'b')
        % xlim([t(1) t(end)])
        % ylim([-0.1 5])
        % title('bias \omega')
        % grid on
        % hold on
        % % plot([t(1) t(end)],[1 1]*(a^2+b^2)/b^2,'r')
        % hold off
    
        %%
    
        figure(3)
        subplot(3,1,1)
        % plot(t,vel_log')
        plot(t,vel_log(1,:),'r')
        hold on
        plot(t,vel_log(2,:),'g')
        plot(t,vel_log(3,:),'b')
        plot(t,vel_log(4,:),'k')
        hold off
        legend('w1','w2','w3','w4')
        title('Wheels velocities with respect to the floor')
        grid on
        subplot(3,1,2)
        plot(t,TORQUE_1(1,:),'r')
        hold on
        plot(t,TORQUE_2(1,:),'g')
        plot(t,TORQUE_3(1,:),'b')
        plot(t,TORQUE_4(1,:),'k')
        hold off
        title('Torque on the weels')
        legend('p1','p2','p3','p4')
        grid on
        subplot(3,1,3)
        plot(t,POWER_1(1,:),'r')
        hold on
        plot(t,POWER_2(1,:),'g')
        plot(t,POWER_3(1,:),'b')
        plot(t,POWER_4(1,:),'k')
        hold off
        title('Power on the weels')
        legend('p1','p2','p3','p4')
        grid on
    
    end%DO_PLOTS

    time = t;
    final_states = states';
end








