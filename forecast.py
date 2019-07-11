import numpy as nm
from DIET.diet import psfsnr

dark_sky_mag = {'U':21.3,'B':22.1,'V':21.3,'R':20.4}
grey_sky_mag = {'U':17.3,'B':19.5,'V':19.5,'R':19.1}
bright_sky_mag = {'U':15.0,'B':17.1,'V':18.0,'R':17.4}
skies_mag = {'dark':dark_sky_mag,'grey':grey_sky_mag,'bright':bright_sky_mag}

qsi_640_config = {'rpix':1.01,'nccd':16,'gain':0.7,'saturation':65535.0, 'zpt': {'U':17.95,'B':19.78,'V':19.57,'R':19.54}}
qsi_660_config = {'rpix':0.62,'nccd':7,'gain':0.17,'saturation':65535.0, 'zpt': {'U':19.12,'B':20.18,'V':20.10,'R':20.32}}
qsi_inter_config = {'rpix':1.01,'nccd':16,'gain':0.17,'saturation':65535.0, 'zpt': {'U':19.12,'B':20.18,'V':20.10,'R':20.32}}

class camera_psfsnr(psfsnr):
  '''
  Class for astrometric camera computations
  '''
  def sky_mag2el(self,skymag):
    '''
    Converts sky magnitudes per arcsec^2 to flux in electrons per second per pixel
    '''
    # First convert from mag to e-/s
    res = 10**(-0.4*(skymag-self.zpt))
    # Now convert to e-/s/pix
    res *= self.rpix**2
    return res

  def __init__(self,mAB=24.9,filter='R',am=1.2,trans=1.0,seeing=0.69,beta=3.0,background='dark',texp=3600.,camera_config=qsi_640_config):

    self.camera_config = camera_config
    self.filter = filter
    self.rpix = camera_config['rpix']
    self.nccd = camera_config['nccd']
    self.gain = camera_config['gain']
    self.zpt  = camera_config['zpt'][self.filter]
    self.sky  = self.sky_mag2el(skies_mag[background][filter])
    psfsnr.__init__(self,mAB=mAB,filter=self.filter,am=am,trans=trans,seeing=seeing,beta=beta,background='dark',
      nccd=self.nccd,texp=texp,rpix=self.rpix,gain=self.gain,zpt=self.zpt,sky=self.sky)


def draw_pos_err(magmin,magmax,magstep,seeing=1.3):

  from matplotlib import pyplot as plt

  mags = nm.arange(magmin,magmax,magstep)

  poserr_640 = nm.zeros(mags.size)
  poserr_660 = nm.zeros(mags.size)
  poserr_inter = nm.zeros(mags.size)

  for i in range(mags.size):
      s=camera_psfsnr(mAB=mags[i],texp=1,seeing=seeing,camera_config=qsi_640_config)
      poserr_640[i]=s.position_error()
      s=camera_psfsnr(mAB=mags[i],texp=1,seeing=seeing,camera_config=qsi_660_config)
      poserr_660[i]=s.position_error()
      s=camera_psfsnr(mAB=mags[i],texp=1,seeing=seeing,camera_config=qsi_inter_config)
      poserr_inter[i]=s.position_error()

  plt.semilogy(mags,poserr_640,label='QSI 640')
  plt.semilogy(mags,poserr_660,label='QSI 660')
  plt.semilogy(mags,poserr_inter,'.',label='QSI intermediate')
  plt.legend()

  plt.title('Centroid position error in arcsec')
  plt.xlabel('AB magnitude in R band')
  plt.show()