clc;% clf;
op = bemoptions( 'sim', 'ret', 'waitbar', 0, 'interp', 'curv' );

%  table of dielectric functions
epstab = { epsconst( 1.0^2 ), epstable( 'au_drude.dat' ) };

radius1 = 10; 
radius2 = 30;

for gap = 40:10:71
    %  diameter of sphere
    diameter1 = 2*radius1;
    diameter2 = 2*radius2;

%      initialize sphere
    p1 = trisphere( 144, diameter1 );
    p2 = trisphere( 144, diameter2 );
    
    p1 = shift(p1, [0, 0, -radius1-gap/2] );
    p2 = shift(p2, [0, 0, radius2+gap/2] );

    p = comparticle( epstab, { p1, p2 }, [ 2, 1; 2, 1 ], 1, 2, op );

    %  set up BEM solver
    bem = bemsolver( p, op );

    %  plane wave excitation
    exc = planewave( [ 0, 0, 1 ], [ 1, 0, 0], op );

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
    filename = strcat('sphere_heterodimer/Spectrum_sph_ret_gap',num2str(gap), 'nm_',num2str(radius1), '-', num2str(radius2), 'nm_drude_n1.0.mat');
    save(filename, 'en_ev', 'ext_mcsqrd', 'abs_mcsqrd')
end
beep on;
