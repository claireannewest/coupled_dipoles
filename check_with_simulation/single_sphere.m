clc;% clf;
op = bemoptions( 'sim', 'ret', 'waitbar', 0, 'interp', 'curv' );

%  table of dielectric functions
epstab = { epsconst( 1.0^2 ), epstable( 'au_drude.dat' ) };

for radius = 70:500:100
    %  diameter of sphere
    diameter = 2*radius;

    %  initialize sphere
    p = comparticle( epstab, { trisphere( 500, diameter ) }, [ 2, 1 ], 1, op );

    %  set up BEM solver
    bem = bemsolver( p, op );

    %  plane wave excitation
    exc = planewave( [ 0, 1, 0 ], [ 1, 0, 0], op );

    %  light wavelength in vacuum
    enei = linspace( 1240./3.5, 1240./1.0, 500 );

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
    plot(en_ev, ext_mcsqrd); hold on;
%     filename = strcat('Spectrum_sph_ret_',num2str(radius),'nm_drude_n1.0.mat');
%     save(filename, 'en_ev', 'ext_mcsqrd', 'abs_mcsqrd');
end


%%  comparison with Mie theory
clear;clc
op = bemoptions( 'sim', 'ret', 'waitbar', 0, 'interp', 'curv' );
enei = linspace( 350, 1200, 200 );

radii = 140;
diameter = 2*(radii);

mie = miesolver( epstable( 'au_drude.dat' ), epsconst( 1.0^2 ), diameter, op,'lmax',1);
ext = mie.ext( enei );

sca = mie.sca( enei );
abs = ext-sca;
nmsqrd_to_micronsqrd = (10^(-6));

plot(1240./enei, ext*nmsqrd_to_micronsqrd,'Linewidth',2);  hold on

% legend( 'BEM : x-polarization', 'BEM : y-polarization', 'Mie theory' );
ext_mcsqrd = reshape(ext*nmsqrd_to_micronsqrd, 1, length( enei ));
abs_mcsqrd = reshape(abs*nmsqrd_to_micronsqrd, 1, length( enei ));
sca_mcsqrd = reshape(sca*nmsqrd_to_micronsqrd, 1, length( enei ));
xlim([1.,3.])


