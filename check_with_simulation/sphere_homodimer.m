clc; clf; clearvars; clear;

op = bemoptions( 'sim', 'ret', 'waitbar', 0, 'interp', 'curv' );

%  table of dielectric functions

nback = 1.473;
diel = 'au_drude.dat';

epstab = { epsconst( nback^2 ), epstable( diel ) };
angles = [ 0, 30, 60, 90 ];  % polarization angle from x-axis [degrees]

nback_s = num2str( nback );
if ~contains( nback_s, '.' )
    nback_s = [ nback_s '.0' ];
end

if strcmp( diel, 'au_drude.dat' )
    diel_str = 'drude';
elseif strcmp( diel, 'gold.dat' )
    diel_str = 'JC';
end




radius = 25; 

% for gap = 10:10:50
gap = 30;
%  diameter of sphere
diameter = 2*radius;

%  initialize sphere
p1 = trisphere( 256, diameter );
p2 = trisphere( 256, diameter );

p1 = shift(p1, [-radius-gap/2, 0, 0, ] );
p2 = shift(p2, [radius+gap/2, 0, 0] );

p = comparticle( epstab, { p1, p2 }, [ 2, 1; 2, 1 ], 1, 2, op );

%  set up BEM solver
bem = bemsolver( p, op );
enei = linspace( 450, 650, 150 );

%  plane wave excitation
nmsqrd_to_micronsqrd = 1e-6;
ext_mcsqrd = zeros( length( angles ), length( enei ) );  %  [n_angles x n_enei]
abs_mcsqrd = zeros( length( angles ), length( enei ) );

for ia = 1 : length( angles )
    theta = angles( ia ) * pi / 180;
    pol   = [ cos( theta ), sin( theta ), 0 ];
    dir   = [ 0, 0, 1 ];  %  propagation perp to pol, in xy-plane

    exc = planewave(pol, dir, op );

    sca = zeros( length( enei ), 1 );
    ext = zeros( length( enei ), 1 );

    for ien = 1 : length( enei )
        sig           = bem \ exc( p, enei( ien ) );
        sca( ien, : ) = exc.sca( sig );
        ext( ien, : ) = exc.ext( sig );
    end

    ext_mcsqrd( ia, : ) = ext * nmsqrd_to_micronsqrd;
    abs_mcsqrd( ia, : ) = ( ext - sca ) * nmsqrd_to_micronsqrd;
end

en_ev = 1240 ./ enei;
pol_angles = angles;


plot(en_ev, ext_mcsqrd); hold on;
filename = strcat('sphere_homodimer/Spectrum_sph_ret_gap',num2str(gap), 'nm_',num2str(radius),'nm_drude_n', nback_s, '.mat');
save(filename, 'en_ev', 'ext_mcsqrd', 'abs_mcsqrd')
% end
