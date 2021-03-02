% clf;clc;
op = bemoptions( 'sim', 'ret', 'waitbar', 0 );
%  table of dielectric functions
epstab = { epsconst( 1.^2 ), epstable( 'drude.dat' ) };

%  light wavelength in vacuum
enei = linspace( 1240/3., 1240/.6, 300 );

radx1 = 10;
rady1 = 10; 
radz1 = 30;

radx2 = 10; 
rady2 = 10;
radz2 = 30;

gapz = 5;
gapy = 0;
whichpol = 'z';

%  axes of ellipsoids
ax1 = [ 2*radx1, 2*rady1, 2*radz1];
ax2 = [ 2*radx2, 2*rady2, 2*radz2];

%  nano ellipsoids
p1 = scale( trisphere( 144, 1 ), ax1 );
p2 = scale( trisphere( 144, 1 ), ax2 );

% p1 = shift(p1, [0, -rady1-gapy/2, -radz1-gapz] );
% p2 = shift(p2, [0, rady2+gapy/2, radz2-gapz] );

p1 = shift(p1, [0, 0, -radz1-gapz] );
p2 = shift(p2, [0, 0, radz2-gapz] );

p1 = rot(p1, 0, [1, 0, 0]);
p2 = rot(p2, 0, [1, 0, 0]);


p = comparticle( epstab, { p1, p2 }, [ 2, 1; 2, 1], 1, 2, op );


%  set up BEM solver
bem = bemsolver( p, op );
%  plane wave excitation

if whichpol == 'y'
    exc = planewave( [ 0, 1, 0 ], [ 1, 0, 0 ], op );
end

if whichpol == 'z'
    exc = planewave( [ 0, 0, 1 ], [ 1, 0, 0 ], op );
end
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
abs_mcsqrd = reshape(abs*nmsqrd_to_micronsqrd, 1, length( enei ));
ext_mcsqrd = reshape(ext*nmsqrd_to_micronsqrd, 1, length( enei ));
sca_mcsqrd = reshape(sca*nmsqrd_to_micronsqrd, 1, length( enei ));

plot( 1240./enei, abs_mcsqrd, 'o-');  hold on;

xlabel( 'Wavelength (nm)' );
ylabel( 'Scattering cross section (nm^2)' );

write_it = [1240./enei; ext_mcsqrd; abs_mcsqrd; sca_mcsqrd];
% filename = strcat('simulated_spectra/two_ellipsoids/Spectrum_bemret_homo_',string(rady1),'_',string(radz1),'_',whichgap,string(gap),'_pol',whichpol);
fileID = fopen('Spectrum_homo10_30_g5_n1.0','w');
fprintf(fileID,'%s %s %s %s \n', 'Energy [eV]', 'Ext Cross [um^2]', 'Abs Cross [um^2]', 'Sca Cross [um^2]');
fprintf(fileID,'%2.3f \t %2.5e \t %2.5e \t %2.5e \n',write_it);
fclose(fileID);

% disp(strcat(string(rady1),'_',string(radz1),'_',whichgap,string(gap),'_pol',whichpol))
beep on
beep
%%
load handel;
sound(y,Fs);