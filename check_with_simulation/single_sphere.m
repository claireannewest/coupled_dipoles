clc;% clf;
op = bemoptions( 'sim', 'ret', 'waitbar', 0, 'interp', 'curv' );


%  table of dielectric functions
epstab = { epsconst( 1.0^2 ), epstable( 'gold.dat' ) };

for radius = 10:100:20
    %  diameter of sphere
    diameter = 2*radius

    %  initialize sphere
    p = comparticle( epstab, { trisphere( 144, diameter ) }, [ 2, 1 ], 1, op );

    %  set up BEM solver
    bem = bemsolver( p, op );

    %  plane wave excitation
    exc = planewave( [ 0, 1, 0 ], [ 1, 0, 0], op );

    %  light wavelength in vacuum
    enei = linspace( 400, 700, 200 );

    %  allocate scattering and extinction cross sections
    sca = zeros( length( enei ), 1 );
    ext = zeros( length( enei ), 1 );

    %  loop over wavelengths
    for ien = 1 : length( enei ) 
      %  surface charge
      sig = bem \ exc( p, enei( ien ) );
      %  scattering and extinction cross sections
      sca( ien, : ) = exc.sca( sig );
      ext( ien, : ) = exc.ext( sig );
    end

    abs = ext - sca;
    nmsqrd_to_micronsqrd = (10^(-6));
    abs_mcsqrd = reshape(abs*nmsqrd_to_micronsqrd, 1, length( enei ));
    ext_mcsqrd = reshape(ext*nmsqrd_to_micronsqrd, 1, length( enei ));

    en_ev = 1240./enei; 
    plot(enei, abs_mcsqrd); hold on;
%     filename = strcat('Spectrum_sph_ret_',num2str(radius),'nm_drude_n1.0.mat');
%     save(filename, 'en_ev', 'ext_mcsqrd', 'abs_mcsqrd');
end


%%  comparison with Mie theory
clear;clc;
op = bemoptions( 'sim', 'ret', 'waitbar', 0, 'interp', 'curv' );
enei = linspace( 400, 700, 200 );
nmsqrd_to_micronsqrd = (10^(-6));

mie_10 = miesolver( epstable( 'au_drude.dat' ), epsconst( 1.^2 ),  2*(10), op,'lmax',10);
mie_40 = miesolver( epstable( 'au_drude.dat' ), epsconst( 1.^2 ),  2*(40), op,'lmax',10);


subplot(1,3,1);
% plot(enei, (mie_10.ext( enei )-mie_10.sca( enei ))*nmsqrd_to_micronsqrd,'Linewidth',2);  hold on
% plot(enei, (mie_20.ext( enei )-mie_20.sca( enei ))*nmsqrd_to_micronsqrd,'Linewidth',2);  hold on
% plot(enei, (mie_30.ext( enei )-mie_30.sca( enei ))*nmsqrd_to_micronsqrd,'Linewidth',2);  hold on
% plot(enei, (mie_40.ext( enei )-mie_40.sca( enei ))*nmsqrd_to_micronsqrd,'Linewidth',2);  hold on
% plot(enei, (mie_50.ext( enei )-mie_50.sca( enei ))*nmsqrd_to_micronsqrd,'Linewidth',2);  hold on


mie_10 = miesolver( epstable( 'gold.dat' ), epsconst( 1.^2 ),  2*(10), op,'lmax',10);
mie_20 = miesolver( epstable( 'gold.dat' ), epsconst( 1^2 ),  2*(20), op,'lmax',10);
mie_30 = miesolver( epstable( 'gold.dat' ), epsconst( 1^2 ),  2*(30), op,'lmax',10);
mie_40 = miesolver( epstable( 'gold.dat' ), epsconst( 1.^2 ),  2*(40), op,'lmax',10);
mie_50 = miesolver( epstable( 'gold.dat' ), epsconst( 1.^2 ),  2*(50), op,'lmax',10);

plot(enei, (mie_10.ext( enei )-mie_10.sca( enei ))*nmsqrd_to_micronsqrd,':','Linewidth',2);  hold on
% plot(enei, (mie_20.ext( enei )-mie_20.sca( enei ))*nmsqrd_to_micronsqrd,':','Linewidth',2);  hold on
% plot(enei, (mie_30.ext( enei )-mie_30.sca( enei ))*nmsqrd_to_micronsqrd,':','Linewidth',2);  hold on
% plot(enei, (mie_40.ext( enei )-mie_40.sca( enei ))*nmsqrd_to_micronsqrd,':','Linewidth',2);  hold on
% plot(enei, (mie_50.ext( enei )-mie_50.sca( enei ))*nmsqrd_to_micronsqrd,':','Linewidth',2);  hold on



% ylim([0,1E-2])
% 
% subplot(1,3,2);
% plot(enei, mie_10.sca( enei )*nmsqrd_to_micronsqrd,'Linewidth',2);  hold on
% plot(enei, mie_20.sca( enei )*nmsqrd_to_micronsqrd,'Linewidth',2);  hold on
% plot(enei, mie_30.sca( enei )*nmsqrd_to_micronsqrd,'Linewidth',2);  hold on
% plot(enei, mie_40.sca( enei )*nmsqrd_to_micronsqrd,'Linewidth',2);  hold on
% plot(enei, mie_50.sca( enei )*nmsqrd_to_micronsqrd,'Linewidth',2);  hold on
% 
% subplot(1,3,3);
% plot(enei, (mie_10.ext( enei ))*nmsqrd_to_micronsqrd,'Linewidth',2);  hold on
% plot(enei, (mie_20.ext( enei ))*nmsqrd_to_micronsqrd,'Linewidth',2);  hold on
% plot(enei, (mie_30.ext( enei ))*nmsqrd_to_micronsqrd,'Linewidth',2);  hold on
% plot(enei, (mie_40.ext( enei ))*nmsqrd_to_micronsqrd,'Linewidth',2);  hold on
% plot(enei, (mie_50.ext( enei ))*nmsqrd_to_micronsqrd,'Linewidth',2);  hold on
% 


% ylim([0,1E-2])

