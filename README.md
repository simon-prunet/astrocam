# astrocam
Routines for the CFHT astrometric camera
An example of how to run it:

from astropy.io import fits as pyfits
import astrocam
[hdu] = pyfits.open('images/astrometric.fits')
ast=astrocam.astrometry(hdu,num_std=5,max_stars=300,query_radius=20)
ast.match_and_filter()
ast.adjust_wcs()
raf,decf = ast.adjusted_wcs.all_pix2world(ast.xpix_orig,ast.ypix_orig,0)
figure()
plot(ast.ra_ref,ast.dec_ref,'.')
plot(raf,decf,'.')

