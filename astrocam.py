from astropy.stats import sigma_clipped_stats
#from photutils import datasets
from photutils import DAOStarFinder
import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch
from astropy.visualization import ImageNormalize
from photutils import CircularAperture


#hdu = datasets.load_star_image()
import astropy.io.fits as pyfits
import astropy.wcs as wcs
import numpy as np
from pyspherematch import spherematch
from web import query_vizier

class astrometry:

    def __init__(self,hdu,plot=True,num_std=20,max_stars=300,query_radius=5.):

        '''

        :param hdu:
        :param plot:
        :param num_std:
        :param max_stars:
        :param query_radius:
        '''

        self.data = hdu.data.copy()
        self.header = hdu.header.copy()

        # Extract star positions and magnitudes from image
        mean, med, std = sigma_clipped_stats(self.data)
        daofind = DAOStarFinder(fwhm=3,threshold=num_std*std)
        sources = daofind(self.data-med)
        self.pixel_positions = (sources['xcentroid'],sources['ycentroid'])
        self.inst_magnitudes = sources['mag']

        if (plot):
            apertures = CircularAperture(self.pixel_positions, r=4)
            norm = ImageNormalize(stretch=SqrtStretch())
            plt.imshow(self.data,cmap='Greys',origin='lower',norm=norm)
            apertures.plot(color='blue',lw=1.5,alpha=0.5)

        self.wcs_orig = wcs.WCS(self.header)
        (self.ra_orig, self.dec_orig) = self.wcs_orig.all_pix2world(self.pixel_positions[0],self.pixel_positions[1],0)
        arr_ref = query_vizier(query_radius,self.header['CRVAL1'],self.header['CRVAL2'],max_stars=max_stars)
        self.ra_ref = arr_ref[:,0]
        self.dec_ref = arr_ref[:,1]
        self.mag_ref = arr_ref[:,2]
        (self.xpix_ref,self.ypix_ref) = self.wcs_orig.all_world2pix(self.ra_ref,self.dec_ref,0)
        (self.xpix_orig,self.ypix_orig) = self.pixel_positions

        if (plot):

            ref_aper = CircularAperture((self.xpix_ref,self.ypix_ref),r=4)
            ref_aper.plot(color='red',lw=1.5,alpha=0.5)
            plt.show()

    def match_and_filter(self,exclusion=5/3600.,tol=3/3600.,plot=True):
        '''

        :param exclusion: if None, no prefiltering is done. If equal to a distance, ensures that the minimum distance between
        stars of the reference catalog is larger than that distance (in degrees)
        :param tol: maximum distance in degrees within a matched pair
        :return:
        '''

        if (exclusion is not None):
            # Performing pre-filtering on reference catalog star list. Make sure we keep only stars more distant than
            # exclusion value from each other
            idx1, idx2, ds = spherematch(self.ra_ref,self.dec_ref,self.ra_ref,self.dec_ref,nnearest=2)
            # Now exclude one of the stars from each pair where distance is smaller than exclusion
            ou = np.where((ds > exclusion))
            self.ra_ref = self.ra_ref[ou]
            self.dec_ref = self.dec_ref[ou]
            self.xpix_ref = self.xpix_ref[ou]
            self.ypix_ref = self.ypix_ref[ou]
            self.mag_reg = self.mag_ref[ou]
            # Do the same on the image star list
            idx1, idx2, ds = spherematch(self.ra_orig,self.dec_orig,self.ra_orig,self.dec_orig,nnearest=2)
            ou = np.where((ds>exclusion))
            self.ra_orig = self.ra_orig[ou]
            self.dec_orig = self.dec_orig[ou]
            self.xpix_orig = self.xpix_orig[ou]
            self.ypix_orig = self.ypix_orig[ou]
            self.inst_magnitudes = self.inst_magnitudes[ou]

        idx1, idx2, ds = spherematch(self.ra_orig,self.dec_orig,self.ra_ref,self.dec_ref,nnearest=1,tol=tol)
        self.ra_orig = self.ra_orig[idx1]
        self.xpix_orig = self.xpix_orig[idx1]
        self.dec_orig = self.dec_orig[idx1]
        self.ypix_orig = self.ypix_orig[idx1]
        self.inst_magnitudes = self.inst_magnitudes[idx1]

        self.ra_ref = self.ra_ref[idx2]
        self.dec_ref = self.dec_ref[idx2]
        self.xpix_ref = self.xpix_ref[idx2]
        self.ypix_ref = self.ypix_ref[idx2]
        self.mag_ref = self.mag_ref[idx2]

        self.distances = ds

        if (plot):

            plt.close()
            apertures = CircularAperture((self.xpix_orig,self.ypix_orig), r=4)
            ref_apertures = CircularAperture((self.xpix_ref,self.ypix_ref), r=4)
            norm = ImageNormalize(stretch=SqrtStretch())
            plt.imshow(self.data,cmap='Greys',origin='lower',norm=norm)
            apertures.plot(color='blue',lw=1.5,alpha=0.5)
            ref_apertures.plot(color='red',lw=1.5,alpha=0.5)
            plt.show()


    def adjust_wcs(self):

        '''
        This routine will start from the original astrometric (approximate) solution, and will adjust the astrometric
        parameters which consist of:
        1) A pixel space polynomial transform r' = p0 + p1*r + p2*r**2 + p3*r**3. For now, we will fix p1=1 since it is
        degenerate with the determinant of the CDij matrix of WCS
        2) A CDij matrix, consisting of 4 independent elements. These elements allow for a linear transform between the pixel
        coordinates and the intermediate world coordinate system in the tangent plane.
        3) CRVAL1,2 give the RA, DEC positions corresponding to the reference pixel of coordinates CRPIX1, CRPIX2
        For a total of 3+4+2 = 9 parameters
        :return: refined wcs solution
        '''

        def radius_transform(r,*p):
            '''
            This routine contains the non-linear transform on the radius defined in pixel space, relative to the reference
            pixel.
            '''
            res = p[0] + r + p[1]*r**2 + p[2]*r**3
            return res

        





