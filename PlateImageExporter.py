
from astropy.io import fits
from photutils.datasets import load_simulated_hst_star_image
#from photutils import 
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 800
from astropy.visualization import simple_norm
from photutils.datasets import (load_simulated_hst_star_image,
                                make_noise_image)

#data += make_noise_image(data.shape, distribution='gaussian',

                         #mean=10.0, stddev=5.0, seed=123) 

simplecos_image = '5624_02_intensity.fits'

simplecos_image = [3754, "11111_TRAILED", 12659,]
for i in simplecos_image:

    hdu = fits.open("{0}_02_intensity.fits".format(str(i)))
    data = hdu[0].data  
    data = data.astype('float64')
    

    norm = simple_norm(data, 'sqrt', percent=99.0)
    plt.imshow(data, norm=norm, origin='lower', cmap='Greys')
    plt.axis('off')
    plt.savefig("{0}_png.png".format(str(i)), bbox_inches='tight')
    hdu = fits.PrimaryHDU(data)
#hdu.writeto('sim_image.fits')