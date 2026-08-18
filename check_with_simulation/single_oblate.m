clc;
op = bemoptions( 'sim', 'ret', 'waitbar', 0, 'interp', 'curv' );

nback = 1.5;
diel = 'gold.dat';

epstab = { epsconst( nback^2 ), epstable( diel ) };

ax = [ 110, 110, 24 ];

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

for i = 1 : size( ax, 1 )
    p = scale( trisphere( 144, 1 ), ax( i, : ) );
    p = comparticle( epstab, { p }, [ 2, 1 ], 1, op );

    bem = bemsolver( p, op );  %  compute once per geometry

    nmsqrd_to_micronsqrd = 1e-6;
    ext_mcsqrd = zeros( length( angles ), length( enei ) );  %  [n_angles x n_enei]
    abs_mcsqrd = zeros( length( angles ), length( enei ) );

    for ia = 1 : length( angles )
        theta = angles( ia ) * pi / 180;
        pol   = [ cos( theta ), sin( theta ), 0 ];
        dir   = [ 0, 0, 1 ];  %  propagation perp to pol, in xy-plane

        exc = planewave( pol, dir,  op );

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

    filename = strcat( 'single_oblate/Spectrum_obl_ret_', ...
                       num2str( ax(i,1) ), 'x', num2str( ax(i,3) ), 'nm_', ...
                       diel_str, '_n', nback_s, '.mat' );
    save( filename, 'en_ev', 'ext_mcsqrd', 'abs_mcsqrd', 'pol_angles' );
end
