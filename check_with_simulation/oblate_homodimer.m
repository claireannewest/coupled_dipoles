clc;
op = bemoptions( 'sim', 'ret', 'waitbar', 0, 'interp', 'curv' );

nback = 1.5;
diel = 'gold.dat';

epstab = { epsconst( nback^2 ), epstable( diel ) };

ax = [ 20, 20, 10 ];  % [ major, major, minor ] in nm

nback_s = num2str( nback );
if ~contains( nback_s, '.' )
    nback_s = [ nback_s '.0' ];
end

if strcmp( diel, 'au_drude.dat' )
    diel_str = 'drude';
elseif strcmp( diel, 'gold.dat' )
    diel_str = 'JC';
end

angles = [ 0, 30, 60, 90 ];  % polarization angle from x-axis [degrees]

enei = linspace( 500, 1000, 200 );

gap = 100;
% for gap = 10 : 20 : 20

p1 = scale( trisphere( 144, 1 ), ax );
p2 = scale( trisphere( 144, 1 ), ax );

%  shift side by side along x (plates on a table, separated in-plane)
semi_major = ax( 1 ) / 2;
p1 = shift( p1, [ -(semi_major + gap/2), 0, 0 ] );
p2 = shift( p2, [  (semi_major + gap/2), 0, 0 ] );

p = comparticle( epstab, { p1, p2 }, [ 2, 1; 2, 1 ], 1, 2, op );

bem = bemsolver( p, op );  %  compute once per geometry

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

filename = strcat( 'oblate_homodimer/Spectrum_obl_ret_gap', num2str( gap ), ...
                   'nm_', num2str( ax(1) ), 'x', num2str( ax(3) ), ...
                   'nm_', diel_str, '_n', nback_s, '.mat' );
save( filename, 'en_ev', 'ext_mcsqrd', 'abs_mcsqrd', 'pol_angles' );
% end
