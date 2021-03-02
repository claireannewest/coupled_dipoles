clear;clc;
op = bemoptions( 'sim', 'ret', 'waitbar', 0 );
%  table of dielectric functions
epstab = { epsconst( 1.5^2 ), epstable( 'drude.dat' ) };

%  light wavelength in vacuum
enei = linspace( 500, 2000, 200 );


%  axes of ellipsoids
ax = [ 2*10, 2*10, 2*30];

%  nano ellipsoid
p = scale( trisphere( 144, 1 ), ax( 1, : ) );

%  set up COMPARTICLE object
p = comparticle( epstab, { p }, [ 2, 1 ], 1, op );

%  set up BEM solver
bem = bemsolver( p, op );
%  plane wave excitation
exc = planewave( [ 0, 0, 1 ], [ 1, 0, 0 ], op );

%  loop over wavelengths
for ien = 1 : length( enei )
%  surface charge
sig = bem \ exc( p, enei( ien ) );
%  scattering and extinction cross sections
sca( ien, 1 ) = exc.sca( sig );
ext( ien, 1 ) = exc.ext( sig );
end

abs = ext - sca;
nmsqrd_to_micronsqrd = (10^(-6));
size(abs)

abs_mcsqrd = reshape(abs*nmsqrd_to_micronsqrd, 1, length( enei ));
ext_mcsqrd = reshape(ext*nmsqrd_to_micronsqrd, 1, length( enei ));
sca_mcsqrd = reshape(sca*nmsqrd_to_micronsqrd, 1, length( enei ));

plot( enei, ext_mcsqrd, 'o-');  hold on;

xlabel( 'Wavelength (nm)' );
ylabel( 'Scattering cross section (nm^2)' );


write_it = [1240./enei; ext_mcsqrd; abs_mcsqrd; sca_mcsqrd];
fileID = fopen('Spectrum_10_30_n1.5','w');
fprintf(fileID,'%s %s %s %s \n', 'Energy [eV]', 'Ext Cross [um^2]', 'Abs Cross [um^2]', 'Sca Cross [um^2]');
fprintf(fileID,'%2.3f \t %2.5e \t %2.5e \t %2.5e \n',write_it);
fclose(fileID);
%%
abs0 = ext0 - sca0;
nmsqrd_to_micronsqrd = (10^(-6));
abs0_mcsqrd = reshape(abs0*nmsqrd_to_micronsqrd, 1, length( enei ));
ext0_mcsqrd = reshape(ext0*nmsqrd_to_micronsqrd, 1, length( enei ));
sca0_mcsqrd = reshape(sca0*nmsqrd_to_micronsqrd, 1, length( enei ));

write_it0 = [1240./enei; ext0_mcsqrd; abs0_mcsqrd; sca0_mcsqrd];
fileID = fopen('simulated_spectra/single_ellipsoid/Spectrum_bemretMIE_10_10_50_drude_1.0','w');
fprintf(fileID,'%s %s %s %s \n', 'Energy [eV]', 'Ext Cross [um^2]', 'Abs Cross [um^2]', 'Sca Cross [um^2]');
fprintf(fileID,'%2.3f \t %2.5e \t %2.5e \t %2.5e \n',write_it0);
fclose(fileID);


